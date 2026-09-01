from types import SimpleNamespace

import pytest

from olmo_sglang.config import validate_olmo3_moe_config


def _config(**overrides):
    values = {
        "layer_types": ["sliding_attention", "full_attention"],
        "num_hidden_layers": 2,
        "linear_allow_neg_eigval": True,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 8,
        "linear_norm_eps": 1e-5,
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 4,
        "linear_value_head_dim": 16,
        "gating_function": "softmax",
        "normalize_expert_weights": 1.0,
        "original_num_experts_per_tok": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_accepts_attention_reference_config():
    validate_olmo3_moe_config(_config())


def test_accepts_kda_config():
    validate_olmo3_moe_config(
        _config(layer_types=["linear_attention", "full_attention"])
    )


def test_rejects_kda_config_with_missing_fields():
    config = _config(layer_types=["linear_attention", "full_attention"])
    del config.linear_norm_eps
    with pytest.raises(ValueError, match="linear_norm_eps"):
        validate_olmo3_moe_config(config)


def test_rejects_layer_count_mismatch():
    with pytest.raises(ValueError, match="one entry per logical layer"):
        validate_olmo3_moe_config(_config(num_hidden_layers=3))
