"""Opt-in Core arithmetic with persistent, publication-safe weight layouts."""

from __future__ import annotations

import importlib
import os

import torch
from torch import nn
from torch.nn import functional as F

ENVIRONMENT_VARIABLE = "OLMO_SGLANG_CORE_COMPAT"


def enabled() -> bool:
    value = os.environ.get(ENVIRONMENT_VARIABLE, "0").lower().strip()
    if value not in {"0", "1", "false", "true", "off", "on", ""}:
        raise ValueError(f"Invalid {ENVIRONMENT_VARIABLE} value: {value!r}")
    return value in {"1", "true", "on"}


def validate_runtime(config, parallel, args, quant_config):
    if not enabled():
        return
    if parallel.tp_size != 1 or parallel.moe_ep_size != 1:
        raise ValueError("Core compatibility currently requires TP1 and EP1")
    if quant_config is not None or args.dtype not in {"bfloat16", "bf16"}:
        raise ValueError("Core compatibility currently requires unquantized BF16")
    if (
        args.cuda_graph_backend_decode != "disabled"
        or args.cuda_graph_backend_prefill != "disabled"
    ):
        raise ValueError(
            "Core compatibility requires disabled prefill and decode CUDA graphs"
        )
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
