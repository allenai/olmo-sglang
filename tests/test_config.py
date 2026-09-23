from types import SimpleNamespace

import pytest

from olmo_sglang.config import validate_olmo3_moe_config


def _config(**overrides):
    values = {
        "layer_types": ["sliding_attention", "full_attention"],
        "num_hidden_layers": 2,
        "use_head_qk_norm": True,
        "sliding_window": 128,
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


def test_rejects_full_projection_qk_norm_instead_of_using_headwise_math():
    with pytest.raises(NotImplementedError, match="use_head_qk_norm=True"):
        validate_olmo3_moe_config(_config(use_head_qk_norm=False))


@pytest.mark.parametrize("gate_type", ["headwise", "unknown"])
def test_rejects_unsupported_attention_gate_instead_of_dropping_it(gate_type):
    with pytest.raises(NotImplementedError, match="attention_gate_type"):
        validate_olmo3_moe_config(_config(attention_gate_type=gate_type))


@pytest.mark.parametrize("window", [None, 0, 1, -1, 1.5, True])
def test_rejects_invalid_sliding_window(window):
    with pytest.raises(ValueError, match="sliding_window >= 2"):
        validate_olmo3_moe_config(_config(sliding_window=window))


def test_kda_only_does_not_require_softmax_attention_settings():
    validate_olmo3_moe_config(
        _config(layer_types=["linear_attention"] * 2, use_head_qk_norm=False)
    )


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


def test_per_head_gains_require_headwise_normalization():
    with pytest.raises(ValueError, match="use_head_qk_norm"):
        validate_olmo3_moe_config(
            _config(qk_norm_per_head_gains=True, use_head_qk_norm=False)
        )
    validate_olmo3_moe_config(
        _config(
            qk_norm_per_head_gains=True, use_head_qk_norm=True, scalable_softmax=True
        )
    )
