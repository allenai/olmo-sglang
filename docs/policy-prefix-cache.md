# Policy-scoped prefix caching

Long-context RL generates several responses to the same prompt. Sharing their prefill can save work, but a weight update must not expose old or mixed-policy KV/recurrent state to new requests. This experimental backend adds policy namespaces in **our Olmo-SGLang plugin**, using the cache registration interface of the existing pinned SGLang runtime. It requires no SGLang source changes or version upgrade.

Select `--radix-cache-backend olmo_policy_versioned_v1` with radix caching enabled. Normal external model discovery registers the factory. The compatibility target remains SGLang `b84bec5761a6cdc8b1df86b9b89ead80a7a9bb02`, as recorded in `pyproject.toml`.

## Ownership and publication

The namespace combines the engine's existing `weight_version` with the caller's cache salt. A request binds before acquiring cached state. Following a committed weight change, new requests use a new namespace. Active requests retain their private progress and locks, but cannot look up or publish shared prefixes while their binding differs from the serving version. Finished and canceled requests release private KV and recurrent state using the pinned cache's cleanup contract. Complete retraction permits rebinding only after all ownership is released.

This is policy versioning, not a new transaction counter. Core must publish strictly increasing integer labels when weights change. Repeating a label is allowed only when republishing the **same unchanged policy**. Bootstrap `None`/`default` is isolated from numeric versions. Arbitrary weight writers that reuse a label for changed weights are unsupported. The MILES companion change pauses generation, finalizes the weight update, checks the committed version using the existing server-info API, and only then resumes. Failure leaves generation paused.

All namespaces share the existing bounded cache and eviction policy. There is no allocator reset or separate memory pool per version. Retaining old state remains the existing asynchronous-training approximation; this change prevents sharing that state with new-policy requests.

## Initial scope

Device-only TP1/PP1/DP1 text generation with full-attention/KDA cache components is supported. Sessions, multimodal requests, sliding-window attention, speculation, adapters, and host/external offload are rejected by this experimental wrapper. These are implementation and validation boundaries, not claims that prefix caching is fundamentally incompatible with those features. Existing defaults and ordinary cache selection are unchanged. Recurrent cache capacity must include active state, working buffers and reusable snapshots; the Open Instruct configuration check requires more than five slots per running request for `extra_buffer`.

MILES consistent-hash routing keeps siblings on the same engine. The engine's existing `lpm` scheduling deprioritizes cold siblings whose prefixes match within the waiting queue. This improves the opportunity to reuse prefill, but is not a strict single-flight guarantee and needs measurement under actual concurrency.

## Compatibility checks and the first GPU probe

The two-GPU DocQA probe [01M6X52W0E4WRPA101MM6TZQEZ](https://beaker.org/ex/01M6X52W0E4WRPA101MM6TZQEZ) loaded weights and captured CUDA graphs, then failed at cache construction with zero optimizer updates. The guard incorrectly rejected `disaggregation_decode_retraction_backup='cpu_tensor'`. The pinned `kv_cache_builder.resolve_decode_retraction_backup()` resolves the unset value to `cpu_tensor` even when `disaggregation_mode='null'`. This inactive decode-side setting does not enable cache offload. The factory now accepts it, and a CPU regression test runs the real pinned resolver before constructing the real KV/KDA cache. Rejections report every offending field, its resolved value, and its reason.

| Restriction | Why it currently exists | What would allow relaxing it |
|---|---|---|
| Radix cache disabled / non-generative model | This backend implements prefix sharing for generative requests. | A different execution contract, rather than removing the check. |
| TP/PP/DP/DCP greater than one | Cross-rank ownership and publication were outside the initial single-engine validation. | Multi-rank cache, publication, retraction, and cleanup tests. There is no inherent TP1 requirement in policy namespacing. |
| SWA, DSA, encoder-decoder | Our retained-request continuation and cleanup path was exercised only with full-attention and KDA state. | Component-specific ownership, eviction, and continuation validation. |
| Sessions / multimodal requests | Cross-turn ownership and non-text prefix identity are outside the request binding contract. | Bind and preserve those identities and lifetimes explicitly. |
| Disaggregated prefill/decode | Transferred state must retain a policy binding across engines. | Verify or extend the transfer contract and publication coordination. |
| Host/external caches and host-pool retraction | These add transfers, sidecars, and ownership outside the device-only cleanup path. | Policy-scoped transfer and retained-state cleanup tests. The resolved inactive `cpu_tensor` default is allowed. |
| LoRA | Adapter identity and adapter updates are not covered by our single serving-version binding. | Explicit adapter identity/version guarantees plus tests. |
| Speculation | Accepted/rejected draft branches, recurrent-state rollback, and any draft-model state need to remain correct across live weight publication. | Test each algorithm's state lifecycle; do not treat all speculative algorithms as equivalent. |

Speculation and prefix caching can coexist. **NGRAM is the simplest follow-up**: it proposes tokens from a CPU corpus and has no separate draft model or draft-model KV cache. A corpus proposal is not reused target-model state; target verification still decides acceptance. We nevertheless need to test accepted/rejected branches and KDA rollback while a request spans a weight update. EAGLE/MTP-style paths can also own draft runners and pools; their identities, updates, and retained state need a separate audit. Existing fixed-weight speculative KDA smoke tests do not establish correctness of this live-update wrapper. For now the guard states this qualification gap explicitly rather than implying impossibility.

## Validation and remaining work

CPU tests against the unchanged pinned source exercise real KV/KDA pools: same-policy hits, cross-policy misses, retained chunk continuation, completion/cancellation cleanup, registration, and eight cold siblings through the built-in scheduler. Pure namespace tests cover salts, retraction and invalid/regressing versions.

A two-B300 DocQA integration probe completed four optimizer updates with exit code 0. The factory initialized successfully, and one observed prefix hit reused 12,416 tokens with only 15 new prefill tokens. Versions 0–4 were published with retained state; 32 logged responses crossed policy boundaries. The final rollout counters reported no failed or requeued groups.

All four consumed training batches were generated under version 0 while the first update compiled. Cross-version responses completed in serving but were not consumed by those updates.

Subsequent local RTX 4090 tests used a deterministic tiny full-attention/KDA model with the unchanged pinned runtime:

| Check | Result |
|---|---|
| Six publications, first unchanged then five changed-weight updates | With both torch-native and Triton attention, fresh requests missed old-policy prefixes and exactly matched the corresponding cache-off control's tokens and logprobs at every version. |
| Identical-weight publication during a live response | Exact agreement against a control using the same cache boundary. |
| Repeated/streamed requests versus ordinary SGLang caching | Exact agreement in the isolated native-cache comparison. |
| Concurrent old/new requests with a constrained 32-slot recurrent pool | 48 live responses crossed 12 publications. Each new-policy request had zero stale-cache hits; every warm repeat reused 768 tokens; no retractions occurred. |

A deeper cache hit (960 versus 768 prompt tokens) changed the tiny model's continuation even without publication. Ordinary SGLang caching reproduced this exactly; the first 250 generated tokens agreed, with maximum logprob error about 0.004 over that shared history. Publication comparisons therefore use matched cache boundaries, and wrapper comparisons use native caching as the reference. Cache-on/off trajectory differences are reported separately; these checks do not establish identical cached and uncached trajectories in general.

[Immutable local receipts and reproduction scripts](https://beaker.org/ds/01M6YD4TS8PC6SZJKED2TCSDWW) include source hashes and image provenance. The [one-node harness](https://github.com/allenai/open-instruct/tree/13d47a043/scripts/miles/cache_panel) adds real-checkpoint off/on/native/off serving measurements and paired 24-update cache-off/on training. It requires optimizer consumption of cross-version responses and records cache hits, generated tokens, timing phases and occupancy. That panel is submitted, with no performance result yet. Representative throughput, sustained training consumption and long-run memory/learning behavior remain unqualified.

Runs:

1. Successful four-update probe: [Beaker](https://beaker.org/ex/01M6Y5Q3BWZQVRVN1CG0M7NNHM), immutable result dataset `01M6Y5Q3C5VM98YMG402TE8WSY`. Image `01M6Y5NN3FMQGCVV2WX92AMJS6` used plugin `f0444a8`, MILES `0c2b405`, application `a2feaaef3` and Core `903e57d`. Plugin changes after that tested revision only document results.

2. One-node comparison, submitted pending capacity: [Beaker](https://beaker.org/ex/01M6YCQWJC7Q3ESTATPQZRFKGY). Image `01M6YCM32SZRRMNNYX6WG1HDGV` uses application `13d47a043`, MILES `8802440`, plugin `f0444a8`, Core `903e57d` and unchanged SGLang `b84bec5`.
