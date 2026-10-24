# SPDX-License-Identifier: Apache-2.0

"""OLMo-specific expert routing semantics."""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

ROUTING_KERNELS_VARIABLE = "OLMO_SGLANG_ROUTING_KERNELS"


def fp32_router_logits(
    hidden_states: torch.Tensor, router_weight: torch.Tensor
) -> torch.Tensor:
    """Evaluate the OLMo router projection in FP32, matching OLMo-core.

    The model stores the router weight in FP32 already (an exact widening of the
    BF16 checkpoint), so only the activations need a cast there.
    """

    if router_weight.dtype != torch.float32:
        router_weight = router_weight.float()
    return F.linear(hidden_states.float(), router_weight)


def routing_kernel_mode() -> str:
    """Select the fused SGLang top-k kernel or the original tensor operations.

    ``fused`` (default) runs SGLang's single-launch softmax/top-k/renormalize
    kernel on CUDA; ``torch`` keeps the separate tensor operations as a
    diagnostic control. CPU tensors always use the tensor operations.
    """

    value = os.environ.get(ROUTING_KERNELS_VARIABLE, "fused").lower().strip()
    if value not in {"fused", "torch"}:
        raise ValueError(f"Invalid {ROUTING_KERNELS_VARIABLE} value: {value!r}")
    return value


def _fused_topk_supported(
    gating_output: torch.Tensor, normalize_expert_weights: float | None
) -> bool:
    # The fused kernel renormalizes by the sum of the selected weights, which is
    # L1 normalization of non-negative softmax scores; other norms stay on torch.
    return (
        gating_output.is_cuda
        and normalize_expert_weights in (None, 1.0)
        and routing_kernel_mode() == "fused"
    )


def olmo3_moe_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    *,
    normalize_expert_weights: float | None,
    restore_weight_scale: bool,
    original_num_experts_per_tok: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the OLMo router's scoring, normalization, and scale contract."""

    del renormalize
    if _fused_topk_supported(gating_output, normalize_expert_weights):
        return _fused_olmo3_moe_topk(
            hidden_states,
            gating_output,
            topk,
            normalize_expert_weights=normalize_expert_weights,
            restore_weight_scale=restore_weight_scale,
            original_num_experts_per_tok=original_num_experts_per_tok,
        )
    scores = torch.softmax(gating_output.float(), dim=-1)
    topk_weights, topk_ids = torch.topk(scores, topk, dim=-1)
    if normalize_expert_weights is not None:
        topk_weights = topk_weights / torch.linalg.vector_norm(
            topk_weights,
            ord=normalize_expert_weights,
            dim=-1,
            keepdim=True,
        )
    if restore_weight_scale:
        topk_weights = topk_weights * topk
    if (
        original_num_experts_per_tok is not None
        and original_num_experts_per_tok != topk
    ):
        topk_weights = topk_weights * (original_num_experts_per_tok / topk) ** 0.5
    return topk_weights, topk_ids.to(torch.int32)


def _fused_olmo3_moe_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    *,
    normalize_expert_weights: float | None,
    restore_weight_scale: bool,
    original_num_experts_per_tok: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Softmax, top-k and L1 renormalization in one SGLang kernel launch.

    The constant scale factors are folded into one multiply afterwards. Expert
    selection matches the tensor path except for exact ties; the FP32 weights
    can differ in the last bit because the kernel sums in a different order.
    """

    # Imported here so the CPU-only suite can import this module without SGLang.
    from sglang.srt.layers.moe.topk import fused_topk

    topk_weights, topk_ids = fused_topk(
        hidden_states,
        gating_output.float(),
        topk,
        renormalize=normalize_expert_weights is not None,
    )
    scale = float(topk) if restore_weight_scale else 1.0
    if (
        original_num_experts_per_tok is not None
        and original_num_experts_per_tok != topk
    ):
        scale *= (original_num_experts_per_tok / topk) ** 0.5
    if scale != 1.0:
        topk_weights = topk_weights * scale
    return topk_weights, topk_ids
