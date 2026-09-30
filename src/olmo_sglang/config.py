# SPDX-License-Identifier: Apache-2.0

"""Configuration validation shared by the model and lightweight tests."""

from __future__ import annotations

from typing import Any

SUPPORTED_ATTENTION_TYPES = frozenset(
    {"full_attention", "linear_attention", "sliding_attention"}
)


def _validate_emo_config(config: Any) -> None:
    mode = getattr(config, "emo_routing_mode", None)
    if mode not in (None, "full_pool"):
        raise NotImplementedError(f"Unsupported EMO routing mode: {mode!r}")
    fields = (
        "emo_min_document_expert_pool",
        "emo_max_document_expert_pool",
        "emo_eval_document_expert_pool",
        "emo_eos_token_id",
    )
    values = {name: getattr(config, name, None) for name in fields}
    if all(value is None for value in values.values()):
        return
    for name, value in values.items():
        if type(value) is not int or value < (0 if name == "emo_eos_token_id" else 1):
            raise ValueError(f"EMO requires an explicit valid integer {name}")
    experts = getattr(config, "n_routed_experts", None)
    top_k = getattr(config, "num_experts_per_tok", None)
    if type(experts) is not int or type(top_k) is not int or not 0 < top_k <= experts:
        raise ValueError("EMO requires 0 < num_experts_per_tok <= n_routed_experts")
    if not top_k <= values[fields[0]] <= values[fields[1]] <= experts:
        raise ValueError(
            "EMO requires top_k <= min_document_expert_pool <= max_pool <= num_experts"
        )
    if values[fields[2]] != experts:
        raise NotImplementedError(
            "EMO serving supports full-pool routing only: "
            "emo_eval_document_expert_pool must equal n_routed_experts. "
            "Select full-pool execution explicitly during RL preparation."
        )


def validate_olmo3_moe_config(config: Any) -> None:
    """Validate the currently executable native SGLang subset.

    The native model accepts both softmax-attention and OLMo KDA layers. KDA
    execution additionally requires flash-linear-attention 0.5.2 at runtime.
    """

    _validate_emo_config(config)

    if getattr(config, "qk_norm_per_head_gains", False) and not getattr(
        config, "use_head_qk_norm", False
    ):
        raise ValueError("qk_norm_per_head_gains requires use_head_qk_norm=True")

    layer_types = tuple(config.layer_types)
    if len(layer_types) != config.num_hidden_layers:
        raise ValueError("layer_types must contain one entry per logical layer")

    unsupported = sorted(set(layer_types) - SUPPORTED_ATTENTION_TYPES)
    if unsupported:
        raise NotImplementedError(f"Unsupported Olmo layer types: {unsupported}")

    if any(layer_type != "linear_attention" for layer_type in layer_types):
        if not getattr(config, "use_head_qk_norm", False):
            raise NotImplementedError(
                "The native Olmo attention path requires use_head_qk_norm=True; "
                "normalization across the full Q/K projection is not supported"
            )
        if getattr(config, "attention_gate_type", None) not in (None, "elementwise"):
            raise NotImplementedError(
                "The native Olmo attention path supports attention_gate_type=None "
                "or 'elementwise'"
            )
    if "sliding_attention" in layer_types:
        window = getattr(config, "sliding_window", None)
        # SGLang treats a zero left window as full attention. We pass window - 1.
        if not isinstance(window, int) or isinstance(window, bool) or window < 2:
            raise ValueError(
                "sliding_attention requires an integer sliding_window >= 2"
            )

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
            raise NotImplementedError(
                "The initial native Olmo KDA path requires matching "
                "key and value head counts"
            )
        if not 1 <= config.linear_key_head_dim <= 256:
            raise NotImplementedError(
                "The native Olmo KDA path requires key head dimensions in [1, 256]"
            )

    if getattr(config, "gating_function", "softmax") != "softmax":
        raise NotImplementedError(
            "The native SGLang path currently supports softmax routing only"
        )
    if getattr(config, "normalize_expert_weights", 1.0) != 1.0:
        raise NotImplementedError(
            "The native SGLang path requires L1-normalized expert weights"
        )
