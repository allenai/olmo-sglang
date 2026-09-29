# Validation status

`olmo-sglang` supports standalone OLMo inference in the
[pinned SGLang runtime](compatibility.md#integration-baseline). The implementation
has CPU regression tests, GPU kernel tests, and whole-engine checks. The table
below records what those checks establish and where evidence is still missing.

## Coverage by feature

| Area | Established coverage | Limits |
|---|---|---|
| Full/sliding attention | Native attention, headwise Q/K normalization, optional per-head gains, scalable softmax, and elementwise output gates. Tiny BF16 full and biased sliding-attention/HF parity includes chunked prefill and decode graphs. | Triton attention was exercised in the current audit. Only the [supported configuration](compatibility.md#checkpoint-contract) is accepted. Strict full-checkpoint probability comparisons show execution-path sensitivity. |
| KDA prefill and decode | FLA 0.5.2 prefill, packed Triton decode, FP32 recurrent state, unequal K/V widths, and optional `2 * sigmoid(beta)` semantics. GPU recurrence and CUDA-graph tests pass. | GPU/FLA runtime and head-dimension constraints are documented in [compatibility](compatibility.md#checkpoint-contract). |
| Dense and sparse MoE | Dense SwiGLU, FP32 routing with restored top-k scaling, latent/shared experts, peri-LN, and HF-to-fused weight loading. Tiny hybrid-MoE serving is compared with HF. | BF16 kernel rounding can change full-model probabilities and expert choices; see [numerical findings](numerical-findings.md). |
| Weight updates | TP1 updates to all 72/77 fixture tensors match fresh changed-checkpoint engines under decode graphs and populated radix caches. Restoring original weights reproduces original outputs exactly. Every begin, bucket, and end response must succeed. | Standalone publication is validated. Multi-replica admission and policy-version ownership belong to the caller. |
| KDA radix caching | Changed weights invalidate old state: first requests miss, then repeated and mixed-length requests reuse 256-token prefixes. Full and biased sliding-attention fixtures match fresh engines exactly with 64-token prefill chunks and decode graphs. Branch copy, eviction, isolation, and cleared-slot reuse also have focused coverage. | The previous standalone changed-policy gap is closed for these fixtures. Cache hits depend on tracked recurrent-state boundaries; this is a cache design constraint. |
| Tensor parallelism | KDA sharding and per-head gain loading have rank-level tests; TP1/TP2 inference and changed-weight serving have been exercised. | TP1 remains the default. Whole-engine TP2 for newer gain/scale features needs additional coverage. Validate the intended checkpoint and topology. |
| Speculative decoding | Chain/tree target verification, scratch-state isolation, and CUDA-graph replay have correctness tests. | Keep opt-in. NGRAM speculation has been slower than ordinary decode in tested workloads; benchmark the intended draft source. |

See [the September 23 audit](validation.md#september-23-2026-validation-audit) for
current commands, runtime identity, and results. Earlier scheduler probes also
exercised cancellation, forced retraction, and organic KV-pressure retraction;
[validation](validation.md#radix-cache-lifecycle) describes the runnable checks.

## Probability sensitivity across execution paths

Full-checkpoint comparisons have found probability differences across reference
implementations and prefill chunk sizes, even when short greedy continuations
agree. Checkpoint-conversion and graph-execution checks do not establish strict
cross-runtime probability equivalence. See
[numerical behavior](numerical-findings.md) for the implications and arithmetic
controls.

Applications requiring reference-equivalent probabilities or routing must
validate the actual checkpoint and dtype. Token agreement alone is insufficient.

## Readiness assessment

| Item | Assessment | Useful next action |
|---|---|---|
| Strict probability equivalence | Full-checkpoint comparisons have found differences against ordinary and Core-matched HF. | If close agreement is required, compare KDA/MoE arithmetic and routing on matched inputs across chunk sizes. |
| Exact repeatability across tuning/batching | Independent kernel tuning and concurrent graph/cache execution can change results. | Validate repeatability separately on the intended deployment; fixture passes do not establish it. |
| Changed weights with radix and graphs | Validation gap closed for standalone TP1 fixtures; no implementation defect found. | Reuse the grouped probe when changing runtime or cache/update code. |
| TP2 with newer attention gains/scales | Additional coverage only; earlier production TP2 already works. | Optional two-GPU smoke when needed for capacity. |
| Multi-replica publication | Caller-owned integration behavior. | Check admission/version boundaries when changing the caller's topology. |
| NGRAM speculation | Measured performance regression on the screened workload. | Keep opt-in; a useful trained draft needs a new performance screen. |
| FLA/Triton shim | Maintenance debt, not a failing inference test. | Replace or upstream when updating the pinned runtime. |

## Deployment checks

For a new deployment, validate its checkpoint/tokenizer, stop conditions,
cancellation, realistic request lengths, and concurrency. When publishing changed
weights, require successful updates on every replica, verify the policy version
and changed outputs, and invalidate cached state before admitting new requests.
These are topology-specific integration checks; see
[integration boundaries](compatibility.md#integration-boundaries).

## Performance and maintenance

Measure CUDA-graph and speculative-decoding performance on the intended
checkpoint, request lengths, and concurrency. Correctness checks do not promise
throughput gains for another workload or topology.

Remaining optimization work includes a more useful draft source and profiling
KDA/MoE decode overhead on the intended workload. The guarded FLA 0.5.2/Triton
compatibility shim remains maintenance work. An independent PyTorch KDA
recurrence already exists in `validation/reference.py` and runs in CPU CI.
