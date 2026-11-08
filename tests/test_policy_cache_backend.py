"""Exercise our plugin with the pinned engine's real CPU KV and KDA pools."""

from array import array
from types import SimpleNamespace

import pytest
import torch
from sglang.srt import runtime_context
from sglang.srt.managers.schedule_policy import CacheAwarePolicy, SchedulePolicy
from sglang.srt.mem_cache import registry
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.radix_cache import RadixKey
from test_kda_radix_cache import _make_cache, _request

from olmo_sglang import policy_cache
from olmo_sglang.policy_cache.backend import PolicyRadixCache, create


def version(value):
    runtime_context.get_context().override("test.weight_version", weight_version=value)


def matching(tree, rid, tokens=range(8)):
    req = _request(rid)
    req.origin_input_ids = array("q", tokens)
    req.init_next_round_input(tree, cow_mamba=True)
    return req


def publish_prefix(tree, allocator, pool):
    producer = matching(tree, "producer")
    assert pool.alloc([producer]) is not None
    tree.insert(
        InsertParams(
            key=RadixKey(array("q", range(4)), cache_salt=producer.cache_salt),
            value=allocator.alloc(4),
            mamba_value=producer.kv.mamba_pool_idx.unsqueeze(0),
        )
    )
    return producer


def test_real_prefix_hit_then_new_policy_miss(monkeypatch):
    tree, allocator, pool = _make_cache(monkeypatch, PolicyRadixCache)
    version("0")
    publish_prefix(tree, allocator, pool)
    before = matching(tree, "same-policy")
    assert len(before.prefix_indices) == 4
    pool.free_mamba_cache(before)
    version("1")
    after = matching(tree, "new-policy")
    assert len(after.prefix_indices) == 0
    assert before.cache_salt != after.cache_salt
    tree.sanity_check()


@pytest.mark.parametrize("is_insert", [True, False])
def test_retained_chunk_preserves_progress_and_releases_private_state(
    monkeypatch, is_insert
):
    tree, allocator, pool = _make_cache(monkeypatch, PolicyRadixCache)
    version("0")
    publish_prefix(tree, allocator, pool)
    tokens_available = allocator.available_size()
    states_available = pool.mamba_allocator.available_size()
    active = matching(tree, "active")
    assert len(active.prefix_indices) == 4
    active.lock_receipt = tree.inc_lock_ref(active.last_node).to_dec_params()
    assert pool.alloc([active]) is not None
    tail = allocator.alloc(4)
    indices = torch.cat([active.prefix_indices, tail])
    pool.write((active.kv.req_pool_idx, slice(0, 8)), indices)
    active.kv.kv_allocated_len = active.kv.kv_committed_len = 8
    active.set_extend_range(0, 8)
    version("1")
    tree.cache_unfinished_req(active, chunked=True)
    torch.testing.assert_close(active.prefix_indices, indices)
    assert active.kv.cache_protected_len == 4
    active.init_next_round_input(tree, cow_mamba=True)
    torch.testing.assert_close(active.prefix_indices, indices)
    assert len(matching(tree, "fresh").prefix_indices) == 0
    release_kv_cache(active, tree, is_insert=is_insert)
    assert allocator.available_size() == tokens_available
    assert pool.mamba_allocator.available_size() == states_available
    assert not active.kv.holds_kv and not active.kv.holds_mamba
    assert len(matching(tree, "after-finish").prefix_indices) == 0
    tree.sanity_check()


def test_registration_uses_existing_engine_extension_point(monkeypatch):
    monkeypatch.setattr(registry, "_RADIX_CACHE_REGISTRY", {})
    policy_cache.register()
    policy_cache.register()
    assert (
        registry.get_radix_cache_factory(policy_cache.BACKEND) is policy_cache._create
    )


def test_factory_and_builtin_cold_sibling_scheduling(monkeypatch):
    def factory(params):
        runtime_context.get_parallel().override_permanently(
            tp_size=1, pp_size=1, dp_size=1, dcp_size=1
        )
        return create(
            SimpleNamespace(
                params=params,
                server_args=None,
                disable_radix_cache=False,
                enable_hierarchical_cache=False,
                is_hybrid_swa=False,
                is_hybrid_ssm=True,
                is_dsa=False,
                model_config=SimpleNamespace(
                    is_encoder_decoder=False, is_generation=True
                ),
            )
        )

    tree, _, _ = _make_cache(monkeypatch, factory)
    assert isinstance(tree, PolicyRadixCache)
    version("0")
    policy = SchedulePolicy("lpm", tree, False, False, False)
    siblings = [_request(str(i)) for i in range(8)]
    for req in siblings:
        req.origin_input_ids = array("q", range(64))
    deprioritized = policy._compute_prefix_matches(siblings, CacheAwarePolicy.LPM)
    assert deprioritized == {str(i) for i in range(1, 8)}
