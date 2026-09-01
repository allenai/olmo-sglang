# SPDX-License-Identifier: Apache-2.0

"""OLMo KDA backend integration for SGLang.

The target checkpoint enables negative recurrent eigenvalues by computing
``beta = 2 * sigmoid(beta_logits)``. FLA 0.5.2 exposes that contract directly,
so this backend reuses SGLang's KDA cache/scheduler plumbing while dispatching
prefill and decode recurrence to FLA.
"""

from __future__ import annotations

import importlib.metadata
import logging
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import torch
from sglang.srt.configs.mamba_utils import (
    BaseLinearStateParams,
    Mamba2StateDType,
)
from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend
from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)

EXPECTED_FLA_VERSION = "0.5.2"
_OLMO_MODEL_ARCHITECTURE = "Olmo3MoeForCausalLM"
LOGGER = logging.getLogger(__name__)
_REGISTERED = False
_FLA_PATCHED = False


def _triton_needs_fla_patch(version: str) -> bool:
    """Return whether FLA 0.5.2 needs its KDA constexpr compatibility shim."""

    try:
        major_minor = tuple(int(part) for part in version.split(".")[:2])
    except ValueError as error:
        raise RuntimeError(f"Cannot parse Triton version {version!r}") from error
    return major_minor >= (3, 6)


def _rewrite_fla_kernel_source(source: str, *, block_width: int) -> str:
    """Replace FLA's unsupported constexpr helper with an equivalent literal."""

    old_expression = "    BK: tl.constexpr = triton.next_power_of_2(K)\n"
    new_expression = f"    BK: tl.constexpr = {block_width}\n"

    if new_expression in source and old_expression not in source:
        return source
    if old_expression not in source:
        raise RuntimeError(
            "FLA 0.5.2 KDA source did not match the expected Triton 3.6+ compatibility target"
        )
    return source.replace(old_expression, new_expression)


def _patch_fla_for_triton_3_6() -> None:
    """Resolve FLA 0.5.2's KDA block width before Triton 3.6+ compiles it."""

    global _FLA_PATCHED

    triton_version = importlib.metadata.version("triton")
    if _FLA_PATCHED or not _triton_needs_fla_patch(triton_version):
        return

    import fla.ops.kda.chunk_intra as chunk_intra_module
    import fla.ops.kda.chunk_intra_token_parallel as token_parallel_module
    import triton

    kernel = token_parallel_module.chunk_kda_fwd_kernel_intra_token_parallel
    jit_kernel = kernel
    while hasattr(jit_kernel, "fn") and not hasattr(jit_kernel, "_unsafe_update_src"):
        jit_kernel = jit_kernel.fn

    original_launcher = token_parallel_module.chunk_kda_fwd_intra_token_parallel
    patched_block_width: int | None = None

    def chunk_kda_fwd_intra_token_parallel_compat(
        q: torch.Tensor,
        k: torch.Tensor,
        gk: torch.Tensor,
        beta: torch.Tensor,
        Aqk: torch.Tensor,
        Akk: torch.Tensor,
        scale: float,
        cu_seqlens: torch.LongTensor | None = None,
        chunk_size: int = 64,
        sub_chunk_size: int = 16,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal patched_block_width

        block_width = triton.next_power_of_2(q.shape[-1])
        if patched_block_width is None:
            # Triton explicitly provides this API for controlled source
            # rewrites. This kernel has no JIT callers whose hash also needs
            # invalidating. A literal retains constexpr type in Triton 3.6.
            jit_kernel._unsafe_update_src(
                _rewrite_fla_kernel_source(
                    jit_kernel.src,
                    block_width=block_width,
                )
            )
            patched_block_width = block_width
        elif block_width != patched_block_width:
            raise RuntimeError(
                "FLA 0.5.2 KDA compatibility shim cannot mix head widths in one process: "
                f"compiled {patched_block_width}, received {block_width}"
            )

        return original_launcher(
            q=q,
            k=k,
            gk=gk,
            beta=beta,
            Aqk=Aqk,
            Akk=Akk,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            sub_chunk_size=sub_chunk_size,
        )

    token_parallel_module.chunk_kda_fwd_intra_token_parallel = (
        chunk_kda_fwd_intra_token_parallel_compat
    )
    chunk_intra_module.chunk_kda_fwd_intra_token_parallel = (
        chunk_kda_fwd_intra_token_parallel_compat
    )
    _FLA_PATCHED = True
    LOGGER.info("Applied FLA 0.5.2 compatibility shim for Triton %s", triton_version)


def require_fla_0_5_2() -> None:
    """Require the FLA release whose KDA API supports negative eigenvalues."""

    try:
        version = importlib.metadata.version("flash-linear-attention")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "OLMo KDA requires flash-linear-attention==0.5.2. Install the package in the same environment as SGLang."
        ) from error
    if version != EXPECTED_FLA_VERSION:
        raise RuntimeError(
            f"OLMo KDA requires flash-linear-attention==0.5.2; found {version}"
        )

    _patch_fla_for_triton_3_6()


@dataclass(kw_only=True, frozen=True)
class OlmoKDAStateShape:
    """SGLang recurrent-cache geometry for unequal OLMo KDA K/V widths."""

    conv: list[tuple[int, int]]
    temporal: tuple[int, int, int]
    num_heads: int
    head_dim: int
    num_k_heads: int
    head_k_dim: int
    conv_kernel: int
    num_spec: int = 0
    conv_shard_groups: list[int] | None = None
    num_k_heads_per_tp: int = 1
    disable_conv_window_dedup: bool = True
    conv_slice_axis: int = 1

    @classmethod
    def from_config(cls, config: Any, *, tp_world_size: int = 1) -> OlmoKDAStateShape:
        """Build the per-rank cache shape from an OLMo HF config."""

        key_dim = config.linear_num_key_heads * config.linear_key_head_dim
        value_dim = config.linear_num_value_heads * config.linear_value_head_dim
        conv_kernel = config.linear_conv_kernel_dim
        if config.linear_num_key_heads % tp_world_size:
            raise ValueError("linear_num_key_heads must be divisible by TP size")
        if config.linear_num_value_heads % tp_world_size:
            raise ValueError("linear_num_value_heads must be divisible by TP size")
        if key_dim % tp_world_size or value_dim % tp_world_size:
            raise ValueError(
                "linear KDA projection widths must be divisible by TP size"
            )
        return cls(
            conv=[
                (
                    conv_kernel - 1,
                    (2 * key_dim + value_dim) // tp_world_size,
                )
            ],
            temporal=(
                config.linear_num_value_heads // tp_world_size,
                config.linear_value_head_dim,
                config.linear_key_head_dim,
            ),
            num_heads=config.linear_num_value_heads,
            head_dim=config.linear_value_head_dim,
            num_k_heads=config.linear_num_key_heads,
            head_k_dim=config.linear_key_head_dim,
            conv_kernel=conv_kernel,
            conv_shard_groups=[key_dim, key_dim, value_dim],
            num_k_heads_per_tp=config.linear_num_key_heads // tp_world_size,
        )


@dataclass(kw_only=True, frozen=True)
class OlmoKDACacheParams(BaseLinearStateParams):
    """Cache parameters marking the recurrent state as per-channel KDA."""

    shape: OlmoKDAStateShape

    @property
    def is_kda(self) -> bool:
        return True


def _model_activation_dtype(config: Any) -> torch.dtype:
    """Return the checkpoint activation dtype used by the conv cache."""

    dtype = getattr(config, "dtype", None)
    if dtype is None:
        dtype = getattr(config, "torch_dtype", None)
    if isinstance(dtype, str):
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }.get(dtype)
    if dtype not in (torch.bfloat16, torch.float16):
        return torch.bfloat16
    return dtype


def _prepare_olmo_config(config: Any) -> bool:
    architectures = tuple(getattr(config, "architectures", ()) or ())
    if "Olmo3MoeForCausalLM" not in architectures:
        return False

    layer_types = tuple(getattr(config, "layer_types", ()) or ())
    linear_layer_ids = [
        index
        for index, layer_type in enumerate(layer_types)
        if layer_type == "linear_attention"
    ]
    if not linear_layer_ids:
        return False

    config.linear_layer_ids = linear_layer_ids
    config.full_attention_layer_ids = [
        index
        for index, layer_type in enumerate(layer_types)
        if layer_type != "linear_attention"
    ]
    from sglang.srt.runtime_context import get_parallel

    try:
        tp_world_size = get_parallel().attn_tp_size
    except AssertionError:
        # ModelConfig is also constructed once before distributed groups exist.
        # Every TP worker constructs a fresh copy after group initialization.
        tp_world_size = 1
    config.mamba2_cache_params = OlmoKDACacheParams(
        shape=OlmoKDAStateShape.from_config(config, tp_world_size=tp_world_size),
        layers=linear_layer_ids,
        dtype=Mamba2StateDType(
            conv=_model_activation_dtype(config),
            temporal=torch.float32,
        ),
    )
    return True


class _OlmoKDAConfigMatcher(type):
    """Adapt OLMo config matching to SGLang's type-based registry API."""

    def __instancecheck__(cls, instance: object) -> bool:
        return _prepare_olmo_config(instance)


class _OlmoKDAConfig(metaclass=_OlmoKDAConfigMatcher):
    """Virtual config type matching only supported OLMo KDA checkpoints."""


def _enable_olmo_extra_buffer_capability() -> None:
    """Bridge OLMo into SGLang's current extra-buffer architecture gate.

    ``LinearAttnModelSpec`` exposes ``support_mamba_cache_extra_buffer``, but
    SGLang 8f3d3a31f4 still validates the strategy against a private built-in
    architecture set. Extend that set in memory until the registry field is
    consumed by upstream capability resolution. No SGLang source is modified.
    """

    from sglang.srt.arg_groups import overrides

    extra_buffer_archs = getattr(overrides, "_MAMBA_EXTRA_BUFFER_ARCHS", None)
    if extra_buffer_archs is not None:
        overrides._MAMBA_EXTRA_BUFFER_ARCHS = extra_buffer_archs | {
            _OLMO_MODEL_ARCHITECTURE
        }


def register_olmo_kda_backend() -> None:
    """Register OLMo's custom KDA cache geometry and backend with SGLang."""

    global _REGISTERED
    if _REGISTERED:
        return

    from sglang.srt.configs.linear_attn_model_registry import (
        LinearAttnModelSpec,
        register_linear_attn_model,
    )

    _enable_olmo_extra_buffer_capability()
    register_linear_attn_model(
        LinearAttnModelSpec(
            config_class=_OlmoKDAConfig,
            backend_class_name="olmo_sglang.kda.backend.OlmoKDAAttnBackend",
            arch_names=[_OLMO_MODEL_ARCHITECTURE],
            uses_mamba_radix_cache=True,
            support_mamba_cache=True,
            support_mamba_cache_extra_buffer=True,
        )
    )
    _REGISTERED = True


class OlmoFLAKDAKernel(LinearAttnKernelBase):
    """FLA 0.5.2 KDA kernel with OLMo's exact beta activation."""

    supports_packed_decode = False

    _RUN_ARGUMENTS = frozenset(
        {
            "A_log",
            "dt_bias",
            "ssm_states",
            "cache_indices",
            "query_start_loc",
            "return_intermediate_states",
            "lower_bound",
        }
    )

    def __init__(self, *, allow_neg_eigval: bool) -> None:
        require_fla_0_5_2()
        self.allow_neg_eigval = allow_neg_eigval

    @staticmethod
    def _gather_initial_state(
        ssm_states: torch.Tensor, cache_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = cache_indices >= 0
        safe_indices = cache_indices.clamp_min(0).to(torch.long)
        initial_state = ssm_states.index_select(0, safe_indices).clone()
        if not valid.all():
            initial_state[~valid] = 0
        return initial_state, valid

    @staticmethod
    def _commit_final_state(
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        final_state: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        if valid.any():
            ssm_states.index_copy_(
                0,
                cache_indices[valid].to(torch.long),
                final_state[valid].to(ssm_states.dtype),
            )

    def _run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        raw_gate: torch.Tensor,
        raw_beta: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        return_intermediate_states: bool = False,
        lower_bound: float | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if lower_bound is not None:
            raise NotImplementedError(
                "OLMo KDA does not use Kimi's safe-gate lower bound"
            )

        from fla.ops.kda import chunk_kda

        initial_state, valid = self._gather_initial_state(ssm_states, cache_indices)
        raw_gate = raw_gate.reshape(q.shape[0], q.shape[1], v.shape[2], q.shape[-1])
        beta = raw_beta.reshape(q.shape[0], q.shape[1], v.shape[2]).float().sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2.0
        inference_context = (
            torch.inference_mode() if return_intermediate_states else nullcontext()
        )
        with inference_context:
            result = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=raw_gate,
                beta=beta,
                A_log=A_log.reshape(-1),
                dt_bias=dt_bias.reshape(-1),
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                return_intermediate_states=return_intermediate_states,
                transpose_state_layout=True,
                cu_seqlens=query_start_loc,
            )
        output, final_state = result[:2]
        self._commit_final_state(ssm_states, cache_indices, final_state, valid)
        if not valid.all():
            invalid_tokens = torch.repeat_interleave(
                ~valid, query_start_loc[1:] - query_start_loc[:-1]
            )
            output[:, invalid_tokens] = 0
        if return_intermediate_states:
            return output, result[2]
        return output

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        run_kwargs = {
            name: value for name, value in kwargs.items() if name in self._RUN_ARGUMENTS
        }
        return self._run(q, k, v, a, b, **run_kwargs)

    def target_verify(
        self,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        intermediate_states_buffer: torch.Tensor,
        intermediate_state_indices: torch.Tensor,
        cache_steps: int,
        retrieve_parent_token: torch.Tensor | None,
        lower_bound: float | None = None,
        **_: Any,
    ) -> torch.Tensor:
        """Verify draft chains or trees without mutating committed KDA states.

        This correctness-first path snapshots every post-token recurrent state so
        SGLang can commit the accepted step centrally. For a speculative tree,
        each node resumes from the intermediate state of its declared parent.
        """

        if lower_bound is not None:
            raise NotImplementedError(
                "OLMo KDA does not use Kimi's safe-gate lower bound"
            )
        if intermediate_states_buffer is None:
            raise RuntimeError("OLMo KDA target_verify requires intermediate states")
        if q.is_cuda:
            from olmo_sglang.kda.speculative import olmo_kda_target_verify

            return olmo_kda_target_verify(
                a_log=A_log,
                dt_bias=dt_bias,
                q=q,
                k=k,
                v=v,
                raw_gate=a,
                raw_beta=b,
                state=ssm_states,
                state_indices=cache_indices,
                query_start_loc=query_start_loc,
                scratch=intermediate_states_buffer,
                scratch_indices=intermediate_state_indices,
                cache_steps=cache_steps,
                retrieve_parent_token=retrieve_parent_token,
                allow_neg_eigval=self.allow_neg_eigval,
            )
        if q.ndim != 4 or q.shape[0] != 1:
            raise ValueError(
                "q, k, and v must use packed [1, tokens, heads, dim] layout"
            )

        num_tokens = q.shape[1]
        num_value_heads = v.shape[2]
        key_dim = q.shape[-1]
        if k.shape[:2] != q.shape[:2] or k.shape[-1] != key_dim:
            raise ValueError("q and k must have matching token and key dimensions")
        if num_value_heads % q.shape[2] or num_value_heads % k.shape[2]:
            raise ValueError("value heads must be divisible by query and key heads")

        query = q.float().repeat_interleave(num_value_heads // q.shape[2], dim=2)
        key = k.float().repeat_interleave(num_value_heads // k.shape[2], dim=2)
        query *= torch.rsqrt((query * query).sum(dim=-1, keepdim=True) + 1e-6)
        query *= key_dim**-0.5
        key *= torch.rsqrt((key * key).sum(dim=-1, keepdim=True) + 1e-6)
        value = v.float()
        raw_gate = a.reshape(1, num_tokens, num_value_heads, key_dim).float()
        raw_beta = b.reshape(1, num_tokens, num_value_heads).float()
        decay_parameter = A_log.reshape(num_value_heads).float().exp()
        gate_bias = dt_bias.reshape(num_value_heads, key_dim).float()
        beta_multiplier = 2.0 if self.allow_neg_eigval else 1.0
        output = torch.empty_like(v)

        starts = query_start_loc.detach().cpu().tolist()
        if not starts or starts[0] != 0 or starts[-1] != num_tokens:
            raise ValueError("query_start_loc must partition every packed token")
        batch_size = len(starts) - 1
        if cache_indices.numel() < batch_size:
            raise ValueError("cache_indices has fewer entries than the verify batch")
        if intermediate_state_indices.numel() < batch_size:
            raise ValueError(
                "intermediate_state_indices has fewer entries than the verify batch"
            )
        if retrieve_parent_token is not None and (
            retrieve_parent_token.ndim != 2
            or retrieve_parent_token.shape[0] < batch_size
        ):
            raise ValueError("retrieve_parent_token must have shape [batch, steps]")

        with torch.inference_mode():
            for request_index, (start, end) in enumerate(pairwise(starts)):
                request_steps = end - start
                if request_steps > cache_steps:
                    raise ValueError(
                        f"request has {request_steps} verify tokens but cache_steps={cache_steps}"
                    )
                if (
                    retrieve_parent_token is not None
                    and retrieve_parent_token.shape[1] < request_steps
                ):
                    raise ValueError(
                        "retrieve_parent_token has fewer steps than the verify request"
                    )
                state_index = int(cache_indices[request_index].item())
                scratch_index = int(intermediate_state_indices[request_index].item())
                if state_index < 0:
                    output[:, start:end].zero_()
                    intermediate_states_buffer[scratch_index, :request_steps].zero_()
                    continue

                state = ssm_states[state_index].float().clone()
                for step, token_index in enumerate(range(start, end)):
                    if retrieve_parent_token is not None and step:
                        parent_step = int(
                            retrieve_parent_token[request_index, step].item()
                        )
                        if parent_step < 0 or parent_step >= step:
                            raise ValueError(
                                "retrieve_parent_token must reference an earlier "
                                f"node: request={request_index}, step={step}, "
                                f"parent={parent_step}"
                            )
                        state = (
                            intermediate_states_buffer[scratch_index, parent_step]
                            .float()
                            .clone()
                        )
                    log_decay = -decay_parameter[
                        :, None
                    ] * torch.nn.functional.softplus(
                        raw_gate[0, token_index] + gate_bias
                    )
                    state *= log_decay.exp().unsqueeze(-2)
                    delta = value[0, token_index] - torch.einsum(
                        "hvk,hk->hv", state, key[0, token_index]
                    )
                    beta = beta_multiplier * raw_beta[0, token_index].sigmoid()
                    delta *= beta.unsqueeze(-1)
                    state += torch.einsum("hv,hk->hvk", delta, key[0, token_index])
                    token_output = torch.einsum(
                        "hvk,hk->hv", state, query[0, token_index]
                    )
                    output[0, token_index].copy_(token_output)
                    intermediate_states_buffer[scratch_index, step].copy_(state)

        return output

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        run_kwargs = {
            name: value for name, value in kwargs.items() if name in self._RUN_ARGUMENTS
        }
        return self._run(q, k, v, g, beta, **run_kwargs)


class OlmoPackedKDAKernel(OlmoFLAKDAKernel):
    """FLA prefill/reference path plus fused OLMo-semantic one-token decode."""

    supports_packed_decode = True

    def packed_decode(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        num_v_heads: int,
        head_v_dim: int,
        lower_bound: float | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if lower_bound is not None:
            raise NotImplementedError(
                "OLMo KDA does not use Kimi's safe-gate lower bound"
            )
        from olmo_sglang.kda.packed_decode import olmo_packed_kda_decode

        return olmo_packed_kda_decode(
            mixed_qkv,
            a,
            b,
            a_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            state=ssm_states,
            state_indices=cache_indices,
            num_value_heads=num_v_heads,
            value_dim=head_v_dim,
            allow_neg_eigval=self.allow_neg_eigval,
            **kwargs,
        )


class OlmoKDAAttnBackend(KDAAttnBackend):
    """SGLang KDA cache plumbing with FLA recurrence for OLMo semantics."""

    def __init__(self, model_runner: Any) -> None:
        super().__init__(model_runner)
        config = model_runner.model_config.hf_config
        model_runner.model_config.full_attention_layer_ids = (
            config.full_attention_layer_ids
        )
        model_runner.model_config.linear_layer_ids = config.linear_layer_ids
        kernel = OlmoPackedKDAKernel(
            allow_neg_eigval=bool(config.linear_allow_neg_eigval)
        )
        self.kernel_dispatcher.decode_kernel = kernel
        self.kernel_dispatcher.extend_kernel = kernel
        self.kernel_dispatcher.verify_kernel = kernel
        self.kernel_dispatcher.supports_packed_decode = True
