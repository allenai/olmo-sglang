import sys
from types import ModuleType, SimpleNamespace

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
    assert calls[0]["gk"] is q
    assert "g" not in calls[0]
    assert "    BK: tl.constexpr = 64" in jit_kernel.src
    assert (
        chunk_intra.chunk_kda_fwd_intra_token_parallel
        is token_parallel.chunk_kda_fwd_intra_token_parallel
    )


def test_kda_kernel_adapts_beta_and_state_layout_to_fla_0_5_2(monkeypatch):
    calls = []
    inference_modes = []
    intermediate_state = torch.zeros(1, 1, 1, 3, 2)

    def chunk_kda(**kwargs):
        calls.append(kwargs)
        inference_modes.append(torch.is_inference_mode_enabled())
        output = torch.zeros_like(kwargs["v"])
        return output, kwargs["initial_state"] + 1, intermediate_state

    fla = ModuleType("fla")
    ops = ModuleType("fla.ops")
    kda = ModuleType("fla.ops.kda")
    kda.chunk_kda = chunk_kda
    fla.ops = ops
    ops.kda = kda
    for name, module in {
        "fla": fla,
        "fla.ops": ops,
        "fla.ops.kda": kda,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    kernel = object.__new__(kda_backend.OlmoFLAKDAKernel)
    kernel.allow_neg_eigval = True
    q = torch.zeros(1, 2, 1, 2)
    v = torch.zeros(1, 2, 1, 3)
    raw_beta = torch.tensor([[[0.0], [1.0]]])
    state_pool = torch.zeros(1, 1, 3, 2)

    result = kernel.extend(
        q,
        q,
        v,
        q,
        raw_beta,
        A_log=torch.zeros(1),
        dt_bias=torch.zeros(2),
        ssm_states=state_pool,
        cache_indices=torch.tensor([0]),
        query_start_loc=torch.tensor([0, 2]),
        return_intermediate_states=True,
    )

    expected_beta = raw_beta.float().sigmoid() * 2.0
    torch.testing.assert_close(calls[0]["beta"], expected_beta)
    assert calls[0]["transpose_state_layout"] is True
    assert "state_v_first" not in calls[0]
    assert "use_beta_sigmoid_in_kernel" not in calls[0]
    assert "allow_neg_eigval" not in calls[0]
    assert calls[0]["initial_state"].shape == (1, 1, 3, 2)
    assert inference_modes == [True]
    assert result[1] is intermediate_state
    torch.testing.assert_close(state_pool, torch.ones_like(state_pool))
