# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration validation shared by the model and lightweight tests."""

from __future__ import annotations

from typing import Any


SUPPORTED_ATTENTION_TYPES = frozenset({"full_attention", "linear_attention", "sliding_attention"})


def validate_olmo3_moe_config(config: Any) -> None:
    """Validate the currently executable native SGLang subset.

    The native model accepts both softmax-attention and OLMo KDA layers. KDA
    execution additionally requires flash-linear-attention 0.5.2 at runtime.
    """

    layer_types = tuple(config.layer_types)
    if len(layer_types) != config.num_hidden_layers:
        raise ValueError("layer_types must contain one entry per logical layer")

    unsupported = sorted(set(layer_types) - SUPPORTED_ATTENTION_TYPES)
    if unsupported:
        raise NotImplementedError(f"Unsupported Olmo layer types: {unsupported}")

    if "linear_attention" in layer_types:
        required_kda_fields = (
            "linear_allow_neg_eigval",
            "linear_conv_kernel_dim",
            "linear_key_head_dim",
            "linear_norm_eps",
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_value_head_dim",
        )
        missing = [name for name in required_kda_fields if not hasattr(config, name)]
        if missing:
            raise ValueError(f"Olmo KDA config is missing required fields: {missing}")
        if config.linear_num_key_heads != config.linear_num_value_heads:
            raise NotImplementedError("The initial native Olmo KDA path requires matching key and value head counts")
        if not 1 <= config.linear_key_head_dim <= 256:
            raise NotImplementedError("The native Olmo KDA path requires key head dimensions in [1, 256]")

    if getattr(config, "gating_function", "softmax") != "softmax":
        raise NotImplementedError("The native SGLang path currently supports softmax routing only")
    if getattr(config, "normalize_expert_weights", 1.0) != 1.0:
        raise NotImplementedError("The native SGLang path requires L1-normalized expert weights")
