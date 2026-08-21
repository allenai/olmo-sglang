import sys
from itertools import pairwise
from types import ModuleType, SimpleNamespace

import pytest
import torch
from sglang.srt.runtime_context import get_parallel

from olmo_sglang import kda_backend
from olmo_sglang.kda_backend import (
    OlmoKDAStateShape,
    OlmoPackedKDAKernel,
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


def test_olmo_kda_state_shape_shards_heads_and_conv_groups_for_tp2():
    shape = OlmoKDAStateShape.from_config(_config(), tp_world_size=2)
    assert shape.conv == [(3, 64)]
    assert shape.temporal == (2, 16, 8)
    assert shape.num_heads == 4
    assert shape.num_k_heads == 4
    assert shape.num_k_heads_per_tp == 2
    assert shape.conv_shard_groups == [32, 32, 64]


def test_olmo_kda_state_shape_rejects_nondivisible_tp():
    with pytest.raises(ValueError, match="linear_num_key_heads"):
        OlmoKDAStateShape.from_config(_config(), tp_world_size=3)


def test_prepare_olmo_config_marks_hybrid_layers_and_cache():
    config = _config()
    assert _prepare_olmo_config(config)
    assert config.linear_layer_ids == [0]
    assert config.full_attention_layer_ids == [1]
    assert config.mamba2_cache_params.dtype.conv is torch.float16
    assert config.mamba2_cache_params.dtype.temporal is torch.float32
    assert config.mamba2_cache_params.layers == [0]


def test_prepare_olmo_config_uses_worker_attention_tp_size():
    config = _config()
    with get_parallel().override(attn_tp_size=2):
        assert _prepare_olmo_config(config)

    assert config.mamba2_cache_params.shape.conv == [(3, 64)]
    assert config.mamba2_cache_params.shape.temporal == (2, 16, 8)


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


def test_kda_target_verify_snapshots_ragged_chains_without_committing():
    torch.manual_seed(13)
    batch_size, token_count = 2, 5
    query_heads, value_heads = 1, 2
    key_dim, value_dim = 3, 4
    q = torch.randn(1, token_count, query_heads, key_dim)
    k = torch.randn_like(q)
    v = torch.randn(1, token_count, value_heads, value_dim)
    raw_gate = torch.randn(token_count, value_heads, key_dim)
    raw_beta = torch.randn(1, token_count, value_heads)
    a_log = torch.randn(value_heads)
    dt_bias = torch.randn(value_heads, key_dim)
    state_pool = torch.randn(4, value_heads, value_dim, key_dim)
    original_state_pool = state_pool.clone()
    cache_indices = torch.tensor([2, 0], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 2, 5], dtype=torch.int32)
    scratch_indices = torch.tensor([1, 3], dtype=torch.int32)
    scratch = torch.full((4, 3, value_heads, value_dim, key_dim), float("nan"))

    kernel = object.__new__(kda_backend.OlmoFLAKDAKernel)
    kernel.allow_neg_eigval = True
    actual = kernel.target_verify(
        A_log=a_log,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        a=raw_gate,
        b=raw_beta,
        ssm_states=state_pool,
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
        intermediate_states_buffer=scratch,
        intermediate_state_indices=scratch_indices,
        cache_steps=3,
        retrieve_parent_token=None,
    )

    expanded_q = q.float().repeat_interleave(value_heads // query_heads, dim=2)
    expanded_k = k.float().repeat_interleave(value_heads // query_heads, dim=2)
    expanded_q *= torch.rsqrt(
        (expanded_q * expanded_q).sum(dim=-1, keepdim=True) + 1e-6
    )
    expanded_q *= key_dim**-0.5
    expanded_k *= torch.rsqrt(
        (expanded_k * expanded_k).sum(dim=-1, keepdim=True) + 1e-6
    )
    expected = torch.empty_like(v)
    expected_snapshots = {}
    starts = query_start_loc.tolist()
    for request_index, (start, end) in enumerate(pairwise(starts)):
        state = original_state_pool[cache_indices[request_index]].clone()
        for step, token_index in enumerate(range(start, end)):
            decay = -a_log.exp()[:, None] * torch.nn.functional.softplus(
                raw_gate[token_index] + dt_bias
            )
            state *= decay.exp().unsqueeze(-2)
            delta = v[0, token_index] - torch.einsum(
                "hvk,hk->hv", state, expanded_k[0, token_index]
            )
            delta *= (2.0 * raw_beta[0, token_index].sigmoid()).unsqueeze(-1)
            state += torch.einsum("hv,hk->hvk", delta, expanded_k[0, token_index])
            expected[0, token_index] = torch.einsum(
                "hvk,hk->hv", state, expanded_q[0, token_index]
            )
            expected_snapshots[(scratch_indices[request_index].item(), step)] = (
                state.clone()
            )

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(state_pool, original_state_pool)
    for (scratch_index, step), expected_state in expected_snapshots.items():
        torch.testing.assert_close(scratch[scratch_index, step], expected_state)
    assert batch_size == len(starts) - 1


def test_kda_target_verify_follows_tree_parent_states_without_committing():
    torch.manual_seed(17)
    token_count = 4
    key_dim, value_dim = 2, 3
    q = torch.randn(1, token_count, 1, key_dim)
    k = torch.randn_like(q)
    v = torch.randn(1, token_count, 1, value_dim)
    raw_gate = torch.randn(token_count, 1, key_dim)
    raw_beta = torch.randn(1, token_count, 1)
    a_log = torch.randn(1)
    dt_bias = torch.randn(1, key_dim)
    state_pool = torch.randn(1, 1, value_dim, key_dim)
    original_state_pool = state_pool.clone()
    scratch = torch.full((1, token_count, 1, value_dim, key_dim), float("nan"))
    parents = torch.tensor([[0, 0, 0, 1]], dtype=torch.int32)

    kernel = object.__new__(kda_backend.OlmoFLAKDAKernel)
    kernel.allow_neg_eigval = True
    actual = kernel.target_verify(
        A_log=a_log,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        a=raw_gate,
        b=raw_beta,
        ssm_states=state_pool,
        cache_indices=torch.zeros(1, dtype=torch.int32),
        query_start_loc=torch.tensor([0, token_count], dtype=torch.int32),
        intermediate_states_buffer=scratch,
        intermediate_state_indices=torch.zeros(1, dtype=torch.int32),
        cache_steps=token_count,
        retrieve_parent_token=parents,
    )

    normalized_q = q.float()
    normalized_k = k.float()
    normalized_q *= torch.rsqrt(
        (normalized_q * normalized_q).sum(dim=-1, keepdim=True) + 1e-6
    )
    normalized_q *= key_dim**-0.5
    normalized_k *= torch.rsqrt(
        (normalized_k * normalized_k).sum(dim=-1, keepdim=True) + 1e-6
    )
    expected = torch.empty_like(v)
    expected_states = []
    for step in range(token_count):
        state = (
            original_state_pool[0].clone()
            if step == 0
            else expected_states[parents[0, step].item()].clone()
        )
        decay = -a_log.exp()[:, None] * torch.nn.functional.softplus(
            raw_gate[step] + dt_bias
        )
        state *= decay.exp().unsqueeze(-2)
        delta = v[0, step] - torch.einsum("hvk,hk->hv", state, normalized_k[0, step])
        delta *= (2.0 * raw_beta[0, step].sigmoid()).unsqueeze(-1)
        state += torch.einsum("hv,hk->hvk", delta, normalized_k[0, step])
        expected[0, step] = torch.einsum("hvk,hk->hv", state, normalized_q[0, step])
        expected_states.append(state)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(state_pool, original_state_pool)
    torch.testing.assert_close(scratch[0], torch.stack(expected_states))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("allow_neg_eigval", [False, True])
def test_packed_kda_decode_matches_torch_reference(allow_neg_eigval):
    torch.manual_seed(7)
    device = torch.device("cuda")
    batch_size, num_heads, key_dim, value_dim = 4, 2, 8, 16
    num_slots = 6
    dtype = torch.bfloat16
    mixed_qkv = torch.randn(
        batch_size,
        2 * num_heads * key_dim + num_heads * value_dim,
        device=device,
        dtype=dtype,
    )
    raw_gate = torch.randn(batch_size, num_heads, key_dim, device=device, dtype=dtype)
    raw_beta = torch.randn(batch_size, 1, num_heads, device=device, dtype=dtype)
    a_log = torch.randn(num_heads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(num_heads, key_dim, device=device, dtype=torch.float32)
    state_indices = torch.tensor([0, 2, 4, 5], device=device, dtype=torch.int64)
    state = torch.randn(
        num_slots,
        num_heads,
        value_dim,
        key_dim,
        device=device,
        dtype=torch.float32,
    )
    expected_state = state.clone()
    scale = key_dim**-0.5

    q_end = num_heads * key_dim
    k_end = 2 * q_end
    query = mixed_qkv[:, :q_end].reshape(batch_size, num_heads, key_dim).float()
    key = mixed_qkv[:, q_end:k_end].reshape(batch_size, num_heads, key_dim).float()
    value = mixed_qkv[:, k_end:].reshape(batch_size, num_heads, value_dim).float()
    query = torch.nn.functional.normalize(query, dim=-1, eps=1e-6) * scale
    key = torch.nn.functional.normalize(key, dim=-1, eps=1e-6)
    decay = -a_log.exp()[None, :, None] * torch.nn.functional.softplus(
        raw_gate.float() + dt_bias[None]
    )
    beta = raw_beta.reshape(batch_size, num_heads).float().sigmoid()
    if allow_neg_eigval:
        beta *= 2.0
    expected_output = torch.empty(
        batch_size, num_heads, value_dim, device=device, dtype=torch.float32
    )
    for batch_index, slot in enumerate(state_indices.tolist()):
        recurrent = expected_state[slot]
        recurrent *= decay[batch_index].exp().unsqueeze(-2)
        delta = value[batch_index] - torch.einsum(
            "hvk,hk->hv", recurrent, key[batch_index]
        )
        delta *= beta[batch_index].unsqueeze(-1)
        recurrent += torch.einsum("hv,hk->hvk", delta, key[batch_index])
        expected_output[batch_index] = torch.einsum(
            "hvk,hk->hv", recurrent, query[batch_index]
        )

    kernel = object.__new__(OlmoPackedKDAKernel)
    kernel.allow_neg_eigval = allow_neg_eigval
    actual_output = kernel.packed_decode(
        mixed_qkv,
        raw_gate,
        raw_beta,
        A_log=a_log,
        dt_bias=dt_bias,
        scale=scale,
        ssm_states=state,
        cache_indices=state_indices,
        num_v_heads=num_heads,
        head_v_dim=value_dim,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual_output[0].float(),
        expected_output,
        atol=2e-2,
        rtol=2e-2,
    )
    torch.testing.assert_close(state, expected_state, atol=2e-4, rtol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_packed_kda_decode_cuda_graph_replays_new_inputs():
    torch.manual_seed(11)
    device = torch.device("cuda")
    batch_size, num_heads, key_dim, value_dim = 4, 8, 128, 128
    num_slots = 6
    dtype = torch.bfloat16
    scale = key_dim**-0.5
    kernel = object.__new__(OlmoPackedKDAKernel)
    kernel.allow_neg_eigval = True

    static_mixed_qkv = torch.randn(
        batch_size,
        2 * num_heads * key_dim + num_heads * value_dim,
        device=device,
        dtype=dtype,
    )
    static_gate = torch.randn(
        batch_size, num_heads, key_dim, device=device, dtype=dtype
    )
    static_beta = torch.randn(batch_size, 1, num_heads, device=device, dtype=dtype)
    a_log = torch.randn(num_heads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(num_heads, key_dim, device=device, dtype=torch.float32)
    static_indices = torch.tensor([0, 2, 4, 5], device=device, dtype=torch.int64)
    static_state = torch.randn(
        num_slots,
        num_heads,
        value_dim,
        key_dim,
        device=device,
        dtype=torch.float32,
    )

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        kernel.packed_decode(
            static_mixed_qkv,
            static_gate,
            static_beta,
            A_log=a_log,
            dt_bias=dt_bias,
            scale=scale,
            ssm_states=static_state,
            cache_indices=static_indices,
            num_v_heads=num_heads,
            head_v_dim=value_dim,
        )
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = kernel.packed_decode(
            static_mixed_qkv,
            static_gate,
            static_beta,
            A_log=a_log,
            dt_bias=dt_bias,
            scale=scale,
            ssm_states=static_state,
            cache_indices=static_indices,
            num_v_heads=num_heads,
            head_v_dim=value_dim,
        )

    replay_mixed_qkv = torch.randn_like(static_mixed_qkv)
    replay_gate = torch.randn_like(static_gate)
    replay_beta = torch.randn_like(static_beta)
    replay_state = torch.randn_like(static_state)
    expected_state = replay_state.clone()

    q_end = num_heads * key_dim
    k_end = 2 * q_end
    query = replay_mixed_qkv[:, :q_end].reshape(batch_size, num_heads, key_dim).float()
    key = (
        replay_mixed_qkv[:, q_end:k_end].reshape(batch_size, num_heads, key_dim).float()
    )
    value = (
        replay_mixed_qkv[:, k_end:].reshape(batch_size, num_heads, value_dim).float()
    )
    query = torch.nn.functional.normalize(query, dim=-1, eps=1e-6) * scale
    key = torch.nn.functional.normalize(key, dim=-1, eps=1e-6)
    decay = -a_log.exp()[None, :, None] * torch.nn.functional.softplus(
        replay_gate.float() + dt_bias[None]
    )
    beta = 2.0 * replay_beta.reshape(batch_size, num_heads).float().sigmoid()
    expected_output = torch.empty(
        batch_size, num_heads, value_dim, device=device, dtype=torch.float32
    )
    for batch_index, slot in enumerate(static_indices.tolist()):
        recurrent = expected_state[slot]
        recurrent *= decay[batch_index].exp().unsqueeze(-2)
        delta = value[batch_index] - torch.einsum(
            "hvk,hk->hv", recurrent, key[batch_index]
        )
        delta *= beta[batch_index].unsqueeze(-1)
        recurrent += torch.einsum("hv,hk->hvk", delta, key[batch_index])
        expected_output[batch_index] = torch.einsum(
            "hvk,hk->hv", recurrent, query[batch_index]
        )

    static_mixed_qkv.copy_(replay_mixed_qkv)
    static_gate.copy_(replay_gate)
    static_beta.copy_(replay_beta)
    static_state.copy_(replay_state)
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(
        captured_output[0].float(), expected_output, atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(static_state, expected_state, atol=2e-4, rtol=2e-4)
