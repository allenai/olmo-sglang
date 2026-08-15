from types import SimpleNamespace

import pytest

from olmo_sglang.config import validate_olmo3_moe_config


def _config(**overrides):
    values = {
        "layer_types": ["sliding_attention", "full_attention"],
        "num_hidden_layers": 2,
        "gating_function": "softmax",
        "normalize_expert_weights": 1.0,
        "original_num_experts_per_tok": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_accepts_attention_reference_config():
    validate_olmo3_moe_config(_config())


def test_rejects_kda_with_actionable_error():
    with pytest.raises(NotImplementedError, match="KDA"):
        validate_olmo3_moe_config(
            _config(layer_types=["linear_attention", "full_attention"])
        )


def test_rejects_layer_count_mismatch():
    with pytest.raises(ValueError, match="one entry per logical layer"):
        validate_olmo3_moe_config(_config(num_hidden_layers=3))
