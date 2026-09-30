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


def _emo_config(**overrides):
    values = dict(
        n_routed_experts=8,
        num_experts_per_tok=2,
        emo_min_document_expert_pool=2,
        emo_max_document_expert_pool=4,
        emo_eval_document_expert_pool=8,
        emo_eos_token_id=0,
    )
    values.update(overrides)
    return _config(**values)


def test_accepts_full_pool_emo_with_restricted_training_metadata():
    validate_olmo3_moe_config(_emo_config())
    validate_olmo3_moe_config(_emo_config(emo_routing_mode="full_pool"))


def test_rejects_unknown_emo_execution_mode():
    with pytest.raises(NotImplementedError, match="EMO routing mode"):
        validate_olmo3_moe_config(_emo_config(emo_routing_mode="document_pool"))


def test_accepts_null_emo_ancestry_metadata():
    validate_olmo3_moe_config(
        _config(
            emo_min_document_expert_pool=None,
            emo_max_document_expert_pool=None,
            emo_eval_document_expert_pool=None,
            emo_eos_token_id=None,
        )
    )


@pytest.mark.parametrize("pool", [1, 2, 4, 9])
def test_rejects_emo_pool_that_cannot_be_served(pool):
    with pytest.raises(NotImplementedError, match="full-pool"):
        validate_olmo3_moe_config(_emo_config(emo_eval_document_expert_pool=pool))


@pytest.mark.parametrize(
    "field",
    [
        "emo_min_document_expert_pool",
        "emo_max_document_expert_pool",
        "emo_eval_document_expert_pool",
        "emo_eos_token_id",
    ],
)
@pytest.mark.parametrize("value", [None, True, 1.5, "2", -1])
def test_rejects_incomplete_or_malformed_emo(field, value):
    with pytest.raises(ValueError, match=field):
        validate_olmo3_moe_config(_emo_config(**{field: value}))


@pytest.mark.parametrize("minimum,maximum", [(1, 4), (4, 2), (2, 9)])
def test_rejects_invalid_emo_training_range(minimum, maximum):
    with pytest.raises(ValueError, match="min_document_expert_pool"):
        validate_olmo3_moe_config(
            _emo_config(
                emo_min_document_expert_pool=minimum,
                emo_max_document_expert_pool=maximum,
            )
        )


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
