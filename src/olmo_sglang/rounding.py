"""Graph-compatible BF16 rounding using SGLang's existing expert GEMMs.

This deliberately scoped TP1/BF16 adapter reuses the pinned SGLang alignment,
tuning and GEMM interfaces. It does not modify process-global activation hooks,
Core layouts, attention dispatch or recurrent caches.
"""

import torch
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe
from torch.nn import functional as F
from triton import language as tl

from olmo_sglang import core_compat, rounding_kernels


def rounded_experts(value, w13, w2, weights, routes, *, fused=None):
    if fused is None:
        fused = core_compat.fused_rounding_enabled()
    if (
        value.dtype != torch.bfloat16
        or w13.dtype != value.dtype
        or w2.dtype != value.dtype
    ):
        raise ValueError("Rounding compatibility requires BF16 inputs and weights")
    if not (w13.is_contiguous() and w2.is_contiguous()):
        raise ValueError(
            "Rounding compatibility requires ordinary contiguous expert weights"
        )
    if weights.shape != routes.shape or weights.shape[0] != value.shape[0]:
        raise ValueError("Rounding compatibility route/input shape mismatch")
    if not value.numel():
        return value
    value = value.contiguous()
    config, down_config, down_tma, up_tma, sorted_ids, expert_ids, padded_count = (
        fused_moe._prepare_fused_moe_run(
            value,
            w13,
            w2,
            routes,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=None,
        )
    )
    tokens, topk = routes.shape
    experts, width, _ = w13.shape
    padding = (
        min(tokens * topk, experts + 1) * (config["BLOCK_SIZE_M"] - 1)
        if down_tma
        else 0
    )
    gate_up = torch.empty(
        (tokens * topk + padding, width), device=value.device, dtype=value.dtype
    )
    fused_moe.invoke_fused_moe_kernel(
        value,
        w13,
        None,
        gate_up,
        None,
        None,
        None,
        weights,
        routes,
        sorted_ids,
        expert_ids,
        padded_count,
        False,
        topk,
        config,
        compute_type=tl.bfloat16,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        c_sorted=down_tma,
        b_use_tma=up_tma,
        filter_expert=False,
    )
    if fused:
        activated = rounding_kernels.silu_mul(gate_up)
    else:
        gate, up = gate_up.chunk(2, dim=-1)
        activated = F.silu(gate) * up
    outputs = torch.empty(
        (tokens, topk, w2.shape[1]), device=value.device, dtype=value.dtype
    )
    fused_moe.invoke_fused_moe_kernel(
        activated,
        w2,
        None,
        outputs.unsqueeze(0),
        None,
        None,
        None,
        weights,
        routes,
        sorted_ids,
        expert_ids,
        padded_count,
        False,
        1,
        down_config or config,
        compute_type=tl.bfloat16,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        a_use_tma=down_tma,
        b_use_tma=down_tma,
        filter_expert=False,
        router_topk=topk,
    )
    # Down GEMM writes BF16 *before* weighting. Keep weighting and reduction in
    # FP32, then round once, as in Core. Captured tensors use normal graph pools.
    if fused:
        return rounding_kernels.weighted_sum(outputs, weights)
    return (outputs.float() * weights.float().unsqueeze(-1)).sum(1).to(value.dtype)


class RoundingRMSNorm(core_compat.CoreRMSNorm):
    def forward(self, value):
        if value.dtype == torch.bfloat16 and core_compat.fused_rounding_enabled(
            "norms"
        ):
            return rounding_kernels.rms_norm(value, self.weight, self.variance_epsilon)
        return super().forward(value)


class RoundingExperts(FusedMoE):
    """Keep SGLang parameter storage/loaders and publication contracts intact."""

    def forward(self, hidden_states, topk_output):
        if getattr(self.quant_method, "w13_swiglu_interleaved", False):
            raise ValueError(
                "Rounding compatibility requires non-interleaved gate/up weights"
            )
        return rounded_experts(
            hidden_states,
            self.w13_weight,
            self.w2_weight,
            topk_output.topk_weights,
            topk_output.topk_ids,
        )
