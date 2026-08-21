from array import array
from types import SimpleNamespace

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.sampling.sampling_params import SamplingParams

from olmo_sglang import kda_backend
from olmo_sglang.kda_backend import (
    OlmoKDACacheParams,
    OlmoKDAStateShape,
    register_olmo_kda_backend,
)


def _config():
    return SimpleNamespace(
        dtype="float16",
        architectures=["Olmo3MoeForCausalLM"],
        layer_types=["linear_attention", "full_attention", "linear_attention"],
        linear_conv_kernel_dim=4,
        linear_key_head_dim=1,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_value_head_dim=3,
    )


def _request(rid: str) -> Req:
    return Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=array("q"),
        sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
    )


def _make_cache(monkeypatch):
    import sglang.srt.mem_cache.mamba_radix_cache as radix_module

    monkeypatch.setattr(radix_module, "mamba_cache_chunk_size", lambda: 64)
    cache_params = OlmoKDACacheParams(
        shape=OlmoKDAStateShape.from_config(_config()),
        layers=[0, 2],
        dtype=SimpleNamespace(conv=torch.float16, temporal=torch.float32),
    )
    req_pool = HybridReqToTokenPool(
        size=8,
        mamba_size=8,
        mamba_spec_state_size=8,
        max_context_len=32,
        device="cpu",
        enable_memory_saver=False,
        cache_params=cache_params,
        mamba_layer_ids=cache_params.layers,
        enable_mamba_extra_buffer=False,
    )
    kv_pool = HybridLinearKVPool(
        size=32,
        dtype=torch.float16,
        page_size=1,
        head_num=1,
        head_dim=4,
        full_attention_layer_ids=[1],
        device="cpu",
        enable_memory_saver=False,
        mamba_pool=req_pool.mamba_pool,
    )
    allocator = TokenToKVPoolAllocator(
        size=32,
        dtype=torch.float16,
        device="cpu",
        kvcache=kv_pool,
        need_sort=False,
    )
    tree = MambaRadixCache(
        CacheInitParams(
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            disable=False,
        )
    )
    return tree, allocator, req_pool


def _allocate_request(req_pool: HybridReqToTokenPool, rid: str) -> Req:
    req = _request(rid)
    assert req_pool.alloc([req]) is not None
    req_pool.mamba_pool.clear_slots(req.mamba_pool_idx.unsqueeze(0))
    req.mamba_needs_clear = False
    return req


def _state_at(req_pool: HybridReqToTokenPool, index: torch.Tensor):
    pool = req_pool.mamba_pool.mamba_cache
    return [tensor[:, index].clone() for tensor in pool.conv], pool.temporal[
        :, index
    ].clone()


def _assert_state_equal(
    req_pool: HybridReqToTokenPool, index: torch.Tensor, expected
) -> None:
    expected_conv, expected_temporal = expected
    pool = req_pool.mamba_pool.mamba_cache
    for actual, wanted in zip(pool.conv, expected_conv, strict=True):
        torch.testing.assert_close(actual[:, index], wanted)
    torch.testing.assert_close(pool.temporal[:, index], expected_temporal)


def test_registration_enables_no_buffer_mamba_radix_cache(monkeypatch):
    captured = []
    import sglang.srt.configs.linear_attn_model_registry as registry

    monkeypatch.setattr(registry, "register_linear_attn_model", captured.append)
    monkeypatch.setattr(kda_backend, "_REGISTERED", False)

    register_olmo_kda_backend()

    assert len(captured) == 1
    assert captured[0].uses_mamba_radix_cache is True
    assert captured[0].support_mamba_cache is True
    assert captured[0].support_mamba_cache_extra_buffer is False


def test_radix_prefix_copy_branches_and_evicts_complete_olmo_kda_state(monkeypatch):
    tree, allocator, req_pool = _make_cache(monkeypatch)
    state_pool = req_pool.mamba_pool.mamba_cache
    prefix_req = _allocate_request(req_pool, "prefix")
    prefix_index = prefix_req.mamba_pool_idx

    for offset, conv in enumerate(state_pool.conv, start=1):
        values = torch.arange(conv[:, prefix_index].numel(), dtype=conv.dtype)
        conv[:, prefix_index] = values.reshape_as(conv[:, prefix_index]) + offset
    temporal_values = torch.arange(
        state_pool.temporal[:, prefix_index].numel(), dtype=torch.float32
    )
    state_pool.temporal[:, prefix_index] = (
        temporal_values.reshape_as(state_pool.temporal[:, prefix_index]) + 10
    )
    prefix_state = _state_at(req_pool, prefix_index)

    token_ids = [11, 12, 13, 14]
    tree.insert(
        InsertParams(
            key=RadixKey(array("q", token_ids)),
            value=allocator.alloc(len(token_ids)),
            mamba_value=prefix_index.unsqueeze(0),
        )
    )

    branch_a = _allocate_request(req_pool, "branch-a")
    match_a = tree.match_prefix(
        MatchPrefixParams(
            key=RadixKey(array("q", token_ids + [21])),
            req=branch_a,
            cow_mamba=True,
        )
    )
    assert len(match_a.device_indices) == len(token_ids)
    assert branch_a.mamba_cow_src_index is not None
    req_pool.mamba_pool.copy_from(
        branch_a.mamba_cow_src_index,
        branch_a.mamba_pool_idx.unsqueeze(0),
    )
    _assert_state_equal(req_pool, branch_a.mamba_pool_idx, prefix_state)

    for conv in state_pool.conv:
        conv[:, branch_a.mamba_pool_idx] += 100
    state_pool.temporal[:, branch_a.mamba_pool_idx] += 100
    _assert_state_equal(req_pool, prefix_index, prefix_state)

    branch_b = _allocate_request(req_pool, "branch-b")
    match_b = tree.match_prefix(
        MatchPrefixParams(
            key=RadixKey(array("q", token_ids + [22])),
            req=branch_b,
            cow_mamba=True,
        )
    )
    assert len(match_b.device_indices) == len(token_ids)
    req_pool.mamba_pool.copy_from(
        branch_b.mamba_cow_src_index,
        branch_b.mamba_pool_idx.unsqueeze(0),
    )
    _assert_state_equal(req_pool, branch_b.mamba_pool_idx, prefix_state)

    available_before_evict = req_pool.mamba_allocator.available_size()
    result = tree.evict(EvictParams(mamba_num=1))
    assert result.mamba_num_evicted == 1
    assert req_pool.mamba_allocator.available_size() == available_before_evict + 1
    miss = tree.match_prefix(
        MatchPrefixParams(key=RadixKey(array("q", token_ids + [23])))
    )
    assert len(miss.device_indices) == 0

    cancelled_index = branch_a.mamba_pool_idx.clone()
    req_pool.free_mamba_cache(branch_a)
    replacement = _allocate_request(req_pool, "replacement")
    if torch.equal(replacement.mamba_pool_idx, cancelled_index):
        for conv in state_pool.conv:
            assert torch.count_nonzero(conv[:, replacement.mamba_pool_idx]) == 0
        assert (
            torch.count_nonzero(state_pool.temporal[:, replacement.mamba_pool_idx]) == 0
        )

    tree.sanity_check()
