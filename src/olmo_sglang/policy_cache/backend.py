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


def _validate_configuration(context):
    memory, parallel = get_memory(), get_parallel()
    disagg, serving = get_disagg(), get_serving()
    errors = []

    def require(field, value, condition, reason):
        if not condition:
            errors.append(f"{field}={value!r}: {reason}")

    require(
        "disable_radix_cache",
        context.disable_radix_cache,
        not context.disable_radix_cache,
        "prefix sharing must be enabled",
    )
    for field, value in (
        ("is_hybrid_swa", context.is_hybrid_swa),
        ("is_dsa", context.is_dsa),
        ("model_config.is_encoder_decoder", context.model_config.is_encoder_decoder),
    ):
        require(
            field,
            value,
            not value,
            "retained-state handling is validated only for full-attention/Mamba generation",
        )
    require(
        "model_config.is_generation",
        context.model_config.is_generation,
        context.model_config.is_generation,
        "this backend requires generative requests",
    )
    for field in ("tp_size", "pp_size", "dp_size", "dcp_size"):
        value = getattr(parallel, field)
        require(
            field,
            value,
            value == 1,
            "multi-rank cache ownership and policy publication are not yet validated",
        )
    for field, value in (
        ("enable_streaming_session", serving.enable_streaming_session),
        ("enable_session_radix_cache", memory.enable_session_radix_cache),
    ):
        require(
            field,
            value,
            not value,
            "cross-turn session state is not covered by request-scoped policy bindings",
        )
    require(
        "disaggregation_mode",
        disagg.disaggregation_mode,
        disagg.disaggregation_mode == "null",
        "cross-engine KV transfer does not yet carry this backend's policy binding",
    )
    # The pinned builder resolves None to cpu_tensor even for mode='null'.
    # That inactive decode-side setting does not enable cache offload. A real
    # host_pool, however, allocates host cache state in the builder.
    backup = disagg.disaggregation_decode_retraction_backup
    require(
        "disaggregation_decode_retraction_backup",
        backup,
        backup in (None, "none", "cpu_tensor"),
        "host-backed retraction is outside the validated device-only cache path",
    )
    algorithm = get_spec().speculative_algorithm
    require(
        "speculative_algorithm",
        algorithm,
        algorithm is None,
        "speculative accept/rollback and draft-state ownership across weight updates "
        "are not yet validated; this is not a general prefix-cache incompatibility",
    )
    require(
        "enable_lora",
        get_lora().enable_lora,
        not get_lora().enable_lora,
        "adapter identity and updates are not part of this backend's policy binding",
    )
    for field, value in (
        ("enable_hierarchical_cache", context.enable_hierarchical_cache),
        *(
            (field, getattr(memory, field))
            for field in (
                "enable_unified_memory",
                "enable_unified_cache_external_linker",
                "enable_lmcache",
                "enable_flexkv",
                "enable_hisparse",
            )
        ),
    ):
        require(
            field,
            value,
            not value,
            "host/external cache ownership and transfer are not yet policy-scoped",
        )
    if errors:
        raise ValueError(
            "Policy cache configuration rejected:\n  - " + "\n  - ".join(errors)
        )


def create(context):
    _validate_configuration(context)
    return registry.create_unified_radix_cache(context, cache_class=PolicyRadixCache)
