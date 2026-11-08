# Policy-scoped prefix caching

Long-context RL generates several responses to the same prompt. Sharing their prefill can save work, but a weight update must not expose old or mixed-policy KV/recurrent state to new requests. This experimental backend adds policy namespaces in **our Olmo-SGLang plugin**, using the cache registration interface of the existing pinned SGLang runtime. It requires no SGLang source changes or version upgrade.

Select `--radix-cache-backend olmo_policy_versioned_v1` with radix caching enabled. Normal external model discovery registers the factory. The compatibility target remains SGLang `b84bec5761a6cdc8b1df86b9b89ead80a7a9bb02`, as recorded in `pyproject.toml`.

## Ownership and publication

The namespace combines the engine's existing `weight_version` with the caller's cache salt. A request binds before acquiring cached state. Following a committed weight change, new requests use a new namespace. Active requests retain their private progress and locks, but cannot look up or publish shared prefixes while their binding differs from the serving version. Finished and canceled requests release private KV and recurrent state using the pinned cache's cleanup contract. Complete retraction permits rebinding only after all ownership is released.

This is policy versioning, not a new transaction counter. Core must publish strictly increasing integer labels when weights change. Repeating a label is allowed only when republishing the **same unchanged policy**. Bootstrap `None`/`default` is isolated from numeric versions. Arbitrary weight writers that reuse a label for changed weights are unsupported. The MILES companion change pauses generation, finalizes the weight update, checks the committed version using the existing server-info API, and only then resumes. Failure leaves generation paused.

All namespaces share the existing bounded cache and eviction policy. There is no allocator reset or separate memory pool per version. Retaining old state remains the existing asynchronous-training approximation; this change prevents sharing that state with new-policy requests.

## Initial scope

Device-only TP1/PP1/DP1 text generation with full-attention/KDA cache components is supported. Sessions, multimodal input, sliding-window attention, speculation, adapters, and host/external offload are rejected. Existing defaults and ordinary cache selection are unchanged. Recurrent cache capacity must include active state, working buffers and reusable snapshots; the Open Instruct configuration check requires more than five slots per running request for `extra_buffer`.

MILES consistent-hash routing keeps siblings on the same engine. The engine's existing `lpm` scheduling deprioritizes cold siblings whose prefixes match within the waiting queue. This improves the opportunity to reuse prefill, but is not a strict single-flight guarantee and needs measurement under actual concurrency.

## Validation and remaining work

CPU tests against the unchanged pinned source exercise real KV/KDA pools: same-policy hits, cross-policy misses, retained chunk continuation, completion/cancellation cleanup, registration, and eight cold siblings through the built-in scheduler. Pure namespace tests cover salts, retraction and invalid/regressing versions.

A GPU serving/publication probe is still required before treating this as a qualified runtime: compare fixed-weight outputs with cache off/on; update weights with waiting and active requests; check fresh requests against a cold new-policy engine; verify recurrent-state memory stability over repeated updates; then measure prefill work and sustained throughput. No speedup is claimed from CPU tests.
