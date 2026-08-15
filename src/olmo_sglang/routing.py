# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OLMo-specific expert routing semantics."""

from __future__ import annotations

import torch


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

    del hidden_states, renormalize
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
