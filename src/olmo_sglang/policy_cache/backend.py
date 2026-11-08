"""Policy-aware wrapper around the pinned engine's ordinary unified radix cache."""

from dataclasses import replace

import torch
from sglang.srt.mem_cache import registry
from sglang.srt.mem_cache.base_prefix_cache import MatchResult
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import (
    get_disagg,
    get_lora,
    get_memory,
    get_parallel,
    get_serving,
    get_spec,
)

from olmo_sglang.policy_cache.namespaces import Namespaces


class PolicyRadixCache(UnifiedRadixCache):
    def __init__(self, params):
        super().__init__(params)
        self.namespaces = Namespaces(get_serving().weight_version)

    def _current(self, req):
        self.namespaces.observe(get_serving().weight_version)
        return self.namespaces.bind(req)

    def match_prefix(self, params):
        req = params.req
        if req is None:
            raise ValueError("Policy caching requires request-aware prefix lookup")
        if not self._current(req):
            # A retained chunk's row already contains its private progress. Do
            # not rewind it to a shared prefix or replace its recurrent state.
            return MatchResult(
                device_indices=req.prefix_indices,
                last_device_node=req.last_node,
                last_host_node=req.last_host_node,
                best_match_node=req.best_match_node,
                cache_protected_len=req.kv.cache_protected_len,
                mamba_branching_seqlen=req.mamba_branching_seqlen,
            )
        key = params.key
        scoped = RadixKey(
            key.token_ids, key.extra_key, key.is_bigram, key.limit, req.cache_salt
        )
        return super().match_prefix(replace(params, key=scoped))

    def cache_unfinished_req(self, req, chunked=False, **kwargs):
        if self._current(req):
            return super().cache_unfinished_req(req, chunked=chunked, **kwargs)
        # This is the pinned cache's disabled-sharing continuation path: keep
        # all computed row positions, without inserting or changing protection.
        indices = self.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, : len(req.get_fill_ids())
        ]
        req.prefix_indices = indices.to(dtype=torch.int64, copy=True)

    def insert_req(self, req, *, up_to, **kwargs):
        if self._current(req):
            return super().insert_req(req, up_to=up_to, **kwargs)
        # release_kv_cache still frees the private tail and unpins the shared
        # prefix. Reproduce the disabled-sharing component cleanup here because
        # its subsequent on_release(inserted=True) deliberately does nothing.
        for component in self._components_tuple:
            component.cleanup_after_caching_req(req, is_finished=True)


def create(context):
    memory, parallel = get_memory(), get_parallel()
    unsupported = (
        context.disable_radix_cache
        or context.enable_hierarchical_cache
        or context.is_hybrid_swa
        or context.is_dsa
        or context.model_config.is_encoder_decoder
        or not context.model_config.is_generation
        or parallel.tp_size != 1
        or parallel.pp_size != 1
        or parallel.dp_size != 1
        or parallel.dcp_size != 1
        or get_serving().enable_streaming_session
        or get_disagg().disaggregation_mode != "null"
        or get_disagg().disaggregation_decode_retraction_backup not in (None, "none")
        or get_spec().speculative_algorithm is not None
        or get_lora().enable_lora
        or memory.enable_unified_memory
        or memory.enable_session_radix_cache
        or memory.enable_unified_cache_external_linker
        or memory.enable_lmcache
        or memory.enable_flexkv
        or memory.enable_hisparse
    )
    if unsupported:
        raise ValueError(
            "Policy caching requires device-only TP1 full-attention/Mamba generation without sessions, offload, speculation or adapters"
        )
    return registry.create_unified_radix_cache(context, cache_class=PolicyRadixCache)
