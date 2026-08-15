# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration validation shared by the model and lightweight tests."""

from __future__ import annotations

from typing import Any


SUPPORTED_ATTENTION_TYPES = frozenset({"full_attention", "sliding_attention"})


def validate_olmo3_moe_config(config: Any) -> None:
    """Validate the currently executable native SGLang subset.

    The first milestone intentionally supports the attention-only Olmo3MoE
    reference checkpoints. The production model's KDA layers are detected and
    rejected explicitly until their beta-range semantics are added to SGLang's
    recurrent KDA kernels.
    """

    layer_types = tuple(config.layer_types)
    if len(layer_types) != config.num_hidden_layers:
        raise ValueError("layer_types must contain one entry per logical layer")

    unsupported = sorted(set(layer_types) - SUPPORTED_ATTENTION_TYPES)
    if unsupported:
        raise NotImplementedError(
            "Native Olmo KDA is not executable yet; unsupported layer types: "
            f"{unsupported}. Full/sliding-attention checkpoints are supported."
        )

    if getattr(config, "gating_function", "softmax") != "softmax":
        raise NotImplementedError(
            "The native SGLang path currently supports softmax routing only"
        )
    if getattr(config, "normalize_expert_weights", 1.0) != 1.0:
        raise NotImplementedError(
            "The native SGLang path requires L1-normalized expert weights"
        )
