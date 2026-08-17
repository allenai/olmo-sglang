import sys
from types import ModuleType
from types import SimpleNamespace

import torch

from olmo_sglang import kda_backend
from olmo_sglang.kda_backend import (
    OlmoKDAStateShape,
    _OlmoKDAConfig,
    _prepare_olmo_config,
    _rewrite_fla_kernel_source,
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


def test_fla_constexpr_shim_replaces_helper_with_literal():
    source = """def kernel(
    BC: tl.constexpr,
    BH: tl.constexpr,
):
    BK: tl.constexpr = triton.next_power_of_2(K)
    offsets = tl.arange(0, BK)
"""

    rewritten = _rewrite_fla_kernel_source(source, block_width=64)

    assert "    BK: tl.constexpr = 64" in rewritten
    assert "triton.next_power_of_2(K)" not in rewritten
    assert _rewrite_fla_kernel_source(rewritten, block_width=64) == rewritten


def test_fla_constexpr_shim_rewrites_on_first_launcher_call(monkeypatch):
    class FakeJitKernel:
        src = "    BK: tl.constexpr = triton.next_power_of_2(K)\n"

        def _unsafe_update_src(self, source):
            self.src = source

    jit_kernel = FakeJitKernel()
    kernel = SimpleNamespace(fn=jit_kernel)
    calls = []

    def original_launcher(**kwargs):
        calls.append(kwargs)
        return kwargs["Aqk"], kwargs["Akk"]

    fla = ModuleType("fla")
    ops = ModuleType("fla.ops")
    kda = ModuleType("fla.ops.kda")
    chunk_intra = ModuleType("fla.ops.kda.chunk_intra")
    token_parallel = ModuleType("fla.ops.kda.chunk_intra_token_parallel")
    token_parallel.chunk_kda_fwd_kernel_intra_token_parallel = kernel
    token_parallel.chunk_kda_fwd_intra_token_parallel = original_launcher
    chunk_intra.chunk_kda_fwd_intra_token_parallel = original_launcher
    fla.ops = ops
    ops.kda = kda
    kda.chunk_intra = chunk_intra
    kda.chunk_intra_token_parallel = token_parallel

    triton = ModuleType("triton")
    triton.next_power_of_2 = lambda value: 1 << (value - 1).bit_length()
    for name, module in {
        "fla": fla,
        "fla.ops": ops,
        "fla.ops.kda": kda,
        "fla.ops.kda.chunk_intra": chunk_intra,
        "fla.ops.kda.chunk_intra_token_parallel": token_parallel,
        "triton": triton,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(kda_backend, "_FLA_PATCHED", False)
    monkeypatch.setattr(kda_backend.importlib.metadata, "version", lambda _: "3.6.0")

    kda_backend._patch_fla_for_triton_3_6()
    q = torch.empty(1, 2, 3, 48)
    aqk = torch.empty(1)
    akk = torch.empty(1)
    result = chunk_intra.chunk_kda_fwd_intra_token_parallel(
        q=q,
        k=q,
        gk=q,
        beta=torch.empty(1),
        Aqk=aqk,
        Akk=akk,
        scale=1.0,
    )

    assert result == (aqk, akk)
    assert calls[0]["q"] is q
    assert "    BK: tl.constexpr = 64" in jit_kernel.src
    assert (
        chunk_intra.chunk_kda_fwd_intra_token_parallel
        is token_parallel.chunk_kda_fwd_intra_token_parallel
    )
