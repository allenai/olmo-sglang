"""Scoped Core arithmetic with persistent, publication-safe weight layouts."""

from __future__ import annotations

import importlib
import os
from contextlib import contextmanager
from contextvars import ContextVar

import torch
from torch import nn
from torch.nn import functional as F

ENVIRONMENT_VARIABLE = "OLMO_SGLANG_CORE_COMPAT"
_AUTO_MODE = ContextVar("olmo_core_compat_auto_mode", default="off")

# The measured 12.5B hero family, independent of checkpoint path and EMO ancestry.
# Expanding this profile requires graph/refresh and probability/performance checks.
_ROUNDING_PROFILE = {
    "hidden_size": 1024,
    "attention_hidden_size": 1024,
    "num_hidden_layers": 16,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "n_routed_experts": 512,
    "num_experts_per_tok": 16,
    "moe_intermediate_size": 1024,
    "latent_moe_dim": 512,
    "latent_moe_bias": False,
    "latent_moe_up_proj_input_norm": False,
    "shared_expert_intermediate_size": 1024,
    "dense_mlp_intermediate_size": 8192,
    "dense_layers_use_shared_expert": True,
    "hidden_act": "silu",
    "gating_function": "softmax",
    "normalize_expert_weights": 1.0,
    "restore_weight_scale": True,
    "original_num_experts_per_tok": None,
    "embed_norm": True,
    "embed_scale": 32.0,
    "scalable_softmax": True,
    "rms_norm_eps": 1e-6,
    "linear_norm_eps": 1e-5,
    "use_peri_ln": True,
    "use_head_qk_norm": True,
    "qk_norm_per_head_gains": True,
    "attention_bias": False,
    "use_rope": False,
    "attention_gate_type": "elementwise",
    "attention_gate_full_precision": True,
    "linear_num_key_heads": 8,
    "linear_num_value_heads": 8,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 256,
    "linear_conv_kernel_dim": 4,
    "linear_allow_neg_eigval": True,
}


def default_rounding_supported(config, parallel, args, quant_config) -> bool:
    """Conservative automatic selection; explicit modes still use runtime guards."""
    missing = object()
    return (
        all(
            getattr(config, key, missing) == value
            for key, value in _ROUNDING_PROFILE.items()
        )
        and tuple(getattr(config, "dense_layers_indices", ())) == (0,)
        and tuple(getattr(config, "layer_types", ()))
        == tuple(["linear_attention"] * 7 + ["full_attention"]) * 2
        and parallel.tp_size == parallel.moe_ep_size == 1
        and quant_config is None
        and args.dtype in {"bfloat16", "bf16"}
        and args.cuda_graph_backend_decode in {"disabled", "full"}
        and args.cuda_graph_backend_prefill == "disabled"
        and getattr(args, "moe_runner_backend", "auto") in {"auto", "triton"}
        and not getattr(args, "speculative_algorithm", None)
        and not getattr(args, "enable_torch_compile", False)
    )


@contextmanager
def model_mode(config, parallel, args, quant_config):
    """Resolve auto only while building this model; never mutate worker env vars.

    Modules retain their selected implementations after construction. Full-mode
    attention backends still read the explicit environment setting independently.
    Resetting the context prevents a later/different model inheriting this choice.
    """
    selected = (
        "rounding"
        if default_rounding_supported(config, parallel, args, quant_config)
        else "off"
    )
    token = _AUTO_MODE.set(selected)
    try:
        validate_runtime(config, parallel, args, quant_config)
        yield mode()
    finally:
        _AUTO_MODE.reset(token)


def mode() -> str:
    value = os.environ.get(ENVIRONMENT_VARIABLE, "auto").lower().strip()
    if value == "auto":
        return _AUTO_MODE.get()
    if value in {"1", "true", "on", "full"}:
        return "full"
    if value in {"0", "false", "off", ""}:
        return "off"
    if value == "rounding":
        return "rounding"
    raise ValueError(f"Invalid {ENVIRONMENT_VARIABLE} value: {value!r}")


def enabled() -> bool:
    """Whether to use the full, eager Core reference path."""
    return mode() == "full"


def rounding_enabled() -> bool:
    return mode() == "rounding"


def norms_enabled() -> bool:
    return mode() != "off"


def fused_rounding_enabled(component: str = "moe") -> bool:
    """Keep the original tensor implementation as a diagnostic control."""
    value = os.environ.get("OLMO_SGLANG_ROUNDING_KERNELS", "fused").lower().strip()
    if value not in {"fused", "torch", "moe", "norms"}:
        raise ValueError(f"Invalid OLMO_SGLANG_ROUNDING_KERNELS value: {value!r}")
    return value == "fused" or value == component


def validate_runtime(config, parallel, args, quant_config):
    if mode() == "off":
        return
    if parallel.tp_size != 1 or parallel.moe_ep_size != 1:
        raise ValueError("Core compatibility currently requires TP1 and EP1")
    if quant_config is not None or args.dtype not in {"bfloat16", "bf16"}:
        raise ValueError("Core compatibility currently requires unquantized BF16")
    if enabled() and (
        args.cuda_graph_backend_decode != "disabled"
        or args.cuda_graph_backend_prefill != "disabled"
    ):
        raise ValueError(
            "Core compatibility requires disabled prefill and decode CUDA graphs"
        )
    if rounding_enabled() and getattr(args, "moe_runner_backend", "auto") not in {
        "auto",
        "triton",
    }:
        raise ValueError("Rounding compatibility requires the Triton MoE backend")
    if getattr(args, "speculative_algorithm", None):
        raise ValueError("Core compatibility does not support speculative decoding")
    if getattr(config, "attention_bias", False) or getattr(config, "use_rope", True):
        raise ValueError(
            "Core compatibility currently supports bias-free, no-RoPE Olmo models"
        )
    if "sliding_attention" in config.layer_types:
        raise ValueError(
            "Core compatibility currently supports full/KDA attention only"
        )


class CoreRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, value):
        fp32 = value.float()
        normalized = fp32 * torch.rsqrt(
            fp32.square().mean(-1, keepdim=True) + self.variance_epsilon
        )
        return (normalized * self.weight.float()).to(value.dtype)


class CoreDenseMLP(nn.Module):
    """Keep HF loader names while storing Core's packed up/gate and down layouts."""

    def __init__(self, hidden_size, intermediate_size, *, quant_config=None, prefix=""):
        super().__init__()
        if quant_config is not None:
            raise ValueError("Core dense MLP does not support quantization")
        self.gate_up_proj = nn.Module()
        self.gate_up_proj.weight = nn.Parameter(
            torch.empty(hidden_size, 2 * intermediate_size).t()
        )
        self.gate_up_proj.weight.weight_loader = self.load_up_gate
        self.down_proj = nn.Module()
        self.down_proj.weight = nn.Parameter(
            torch.empty(intermediate_size, hidden_size).t()
        )

    @staticmethod
    @torch.no_grad()
    def load_up_gate(param, value, shard_id):
        # External HF/SGLang shard 0 is gate; native Core's first half is up.
        if shard_id not in (0, 1):
            raise ValueError(f"Unsupported dense projection shard: {shard_id}")
        target = param.chunk(2, dim=0)[1 - shard_id]
        if target.shape != value.shape:
            raise ValueError("Core dense projection shape mismatch")
        target.copy_(value)

    def forward(self, value):
        shape = value.shape
        up, gate = (value.reshape(-1, shape[-1]) @ self.gate_up_proj.weight.t()).chunk(
            2, -1
        )
        hidden = up * F.silu(gate)
        output = torch.bmm(
            hidden.unsqueeze(0), self.down_proj.weight.t().unsqueeze(0)
        ).squeeze(0)
        return output.reshape(shape)


class CoreExperts(nn.Module):
    """Core no-EP GEMMs/unpermutation; all publications update the used storage."""

    def __init__(self, num_experts, hidden_size, intermediate_size, **kwargs):
        super().__init__()
        self.num_experts = num_experts
        # Up then gate, unlike SGLang's external fused gate/up payload.
        self.w13_weight = nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_size, hidden_size)
        )
        # Logical HF shape [E,D,H], with Core's contiguous [E,H,D] backing layout.
        self.w2_weight = nn.Parameter(
            torch.empty(num_experts, intermediate_size, hidden_size).transpose(1, 2)
        )
        self.w13_weight.weight_loader = self.weight_loader
        self.w2_weight.weight_loader = self.weight_loader
        utils = importlib.import_module("olmo_core.nn.moe.utils")
        experts = importlib.import_module("olmo_core.nn.moe.v2.routed_experts")
        if not experts.use_torch_grouped_mm():
            raise ValueError(
                "Core compatibility requires Core's Torch grouped GEMM backend"
            )
        self.permute = utils.moe_permute_no_compile
        self.unpermute = utils.moe_unpermute_no_compile
        self.gmm = experts.gmm

    @torch.no_grad()
    def weight_loader(self, param, value, name, *, shard_id, expert_id):
        if not 0 <= expert_id < self.num_experts:
            raise ValueError("Expert ID outside Core compatibility storage")
        if shard_id == "w2" and param is self.w2_weight:
            target = param[expert_id]
        elif shard_id in {"w1", "w3"} and param is self.w13_weight:
            target = param[expert_id].chunk(2, 0)[shard_id == "w1"]
        else:
            raise ValueError(f"Invalid expert shard: {name}, {shard_id}")
        if target.shape != value.shape:
            raise ValueError("Core expert projection shape mismatch")
        target.copy_(value)

    @torch.no_grad()
    def weight_loader_fused(self, param, value, name, *, shard_id):
        if param.shape != value.shape:
            raise ValueError("Core fused expert shape mismatch")
        if shard_id == "w13" and param is self.w13_weight:
            gate, up = value.chunk(2, 1)
            up_target, gate_target = param.chunk(2, 1)
            up_target.copy_(up)
            gate_target.copy_(gate)
        elif shard_id == "w2" and param is self.w2_weight:
            param.copy_(value)
        else:
            raise ValueError(f"Invalid fused expert shard: {name}, {shard_id}")

    def forward(self, value, topk):
        if not value.numel():
            return value
        routes = topk.topk_ids.int()
        permuted, reverse = permute_on_default_stream(self.permute, value, routes)
        counts = torch.bincount(routes.reshape(-1), minlength=self.num_experts).to(
            torch.int32
        )
        up, gate = self.gmm(permuted, self.w13_weight, counts, trans_b=True).chunk(
            2, -1
        )
        hidden = up * F.silu(gate)
        output = self.gmm(hidden, self.w2_weight.transpose(1, 2), counts)
        return self.unpermute(
            inp=output,
            row_id_map=reverse,
            restore_shape=value.shape,
            map_type="index",
            merging_probs=topk.topk_weights,
        )


def permute_on_default_stream(permute, value, routes):
    """Order TE's default-stream sort with SGLang's model execution stream.

    TE 2.17's index permutation launches CUB SortPairs without a stream argument,
    then launches its gather on the current stream. Run both on the default
    stream, with dependencies and allocator lifetimes covering the handoff.
    """
    current = torch.cuda.current_stream(value.device)
    default = torch.cuda.default_stream(value.device)
    if current != default:
        default.wait_stream(current)
        value.record_stream(default)
        routes.record_stream(default)
    with torch.cuda.stream(default):
        permuted, reverse = permute(
            inp=value,
            routing_map=routes,
            num_out_tokens=routes.numel(),
            map_type="index",
        )
    if current != default:
        current.wait_stream(default)
        permuted.record_stream(current)
        reverse.record_stream(current)
    return permuted, reverse
