from types import SimpleNamespace

import torch

from olmo_sglang.kda_backend import (
    OlmoKDAStateShape,
    _OlmoKDAConfig,
    _prepare_olmo_config,
    _triton_needs_fla_patch,
)


def _config():
    return SimpleNamespace(
        dtype="float16",
        architectures=["Olmo3MoeForCausalLM"],
        layer_types=["linear_attention", "full_attention"],
        linear_conv_kernel_dim=4,
        linear_key_head_dim=8,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_value_head_dim=16,
    )


def test_olmo_kda_state_shape_supports_unequal_key_value_widths():
    shape = OlmoKDAStateShape.from_config(_config())
    assert shape.conv == [(3, 128)]
    assert shape.temporal == (4, 16, 8)
    assert shape.conv_shard_groups == [32, 32, 64]


def test_prepare_olmo_config_marks_hybrid_layers_and_cache():
    config = _config()
    assert _prepare_olmo_config(config)
    assert config.linear_layer_ids == [0]
    assert config.full_attention_layer_ids == [1]
    assert config.mamba2_cache_params.dtype.conv is torch.float16
    assert config.mamba2_cache_params.dtype.temporal is torch.float32
    assert config.mamba2_cache_params.layers == [0]


def test_prepare_olmo_config_ignores_attention_only_model():
    config = _config()
    config.layer_types = ["full_attention", "full_attention"]
    assert not _prepare_olmo_config(config)


def test_virtual_config_type_matches_and_prepares_only_olmo_kda():
    config = _config()
    assert isinstance(config, _OlmoKDAConfig)
    assert config.linear_layer_ids == [0]
    assert config.mamba2_cache_params.layers == [0]

    other_architecture = _config()
    other_architecture.architectures = ["OtherForCausalLM"]
    assert not isinstance(other_architecture, _OlmoKDAConfig)

    attention_only = _config()
    attention_only.layer_types = ["full_attention", "full_attention"]
    assert not isinstance(attention_only, _OlmoKDAConfig)


def test_fla_constexpr_shim_covers_pinned_triton_runtime():
    assert not _triton_needs_fla_patch("3.5.1")
    assert _triton_needs_fla_patch("3.6.0")
    assert _triton_needs_fla_patch("3.7.0")
