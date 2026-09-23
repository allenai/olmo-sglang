# Validation status

`olmo-sglang` supports standalone OLMo inference in the
[pinned SGLang runtime](compatibility.md#integration-baseline). The implementation
has CPU regression tests, GPU kernel tests, and whole-engine checks. The table
below records what those checks establish and where evidence is still missing.

## Evidence by feature

| Area | Established coverage | Limits |
|---|---|---|
| Full/sliding attention | Native attention, headwise Q/K normalization, optional per-head gains, scalable softmax, and elementwise output gates. Tiny BF16 full and biased sliding-attention/HF parity includes chunked prefill and decode graphs. | Triton attention was exercised in the current audit. Only the [supported configuration](compatibility.md#checkpoint-contract) is accepted. Strict full-checkpoint probability comparisons show execution-path sensitivity. |
| KDA prefill and decode | FLA 0.5.2 prefill, packed Triton decode, FP32 recurrent state, unequal K/V widths, and optional `2 * sigmoid(beta)` semantics. GPU recurrence and CUDA-graph tests pass. | GPU/FLA runtime and head-dimension constraints are documented in [compatibility](compatibility.md#checkpoint-contract). |
| Dense and sparse MoE | Dense SwiGLU, FP32 routing with restored top-k scaling, latent/shared experts, peri-LN, and HF-to-fused weight loading. Tiny hybrid-MoE serving is compared with HF. | BF16 kernel rounding can change full-model probabilities and expert choices; see [numerical findings](numerical-findings.md). |
| Weight updates | TP1 updates to all 72/77 fixture tensors match fresh changed-checkpoint engines under decode graphs and populated radix caches. Restoring original weights reproduces original outputs exactly. Every begin, bucket, and end response must succeed. | Standalone publication is validated. Multi-replica admission and policy-version ownership belong to the caller. |
| KDA radix caching | Changed weights invalidate old state: first requests miss, then repeated and mixed-length requests reuse 256-token prefixes. Full and biased sliding-attention fixtures match fresh engines exactly with 64-token prefill chunks and decode graphs. Branch copy, eviction, isolation, and cleared-slot reuse also have focused coverage. | The previous standalone changed-policy gap is closed for these fixtures. Cache hits depend on tracked recurrent-state boundaries; this is a cache design constraint. |
| Tensor parallelism | A [production TP1/TP2 screen](https://github.com/allenai/olmo-miles/blob/07887b783ab254577a6656168dc0e0d21aebfe3d/docs/measurements/tensor-parallel-screen.md) matched eight greedy tokens, then completed two changed-weight rollout cycles across four TP2 engines with cache reuse and consistent policy versions. KDA sharding and per-head gain loading also have rank-level tests. | Functional capacity option; TP1 remains the default. Historical max chosen-token logprob difference was 0.04458. Whole-engine TP2 for newer gain/scale features is an optional coverage gap, not a demonstrated failure. |
| Speculative decoding | Chain/tree target verification, scratch-state isolation, and CUDA-graph replay have correctness tests. | The [trained-checkpoint NGRAM screen](https://github.com/allenai/olmo-miles/blob/07887b783ab254577a6656168dc0e0d21aebfe3d/docs/measurements/ngram-speculative-screen.md) was slower than ordinary decode. Keep it opt-in; a better draft source needs a new screen. |

See [the September 23 audit](validation.md#september-23-2026-validation-audit) for
current commands, runtime identity, and results. Earlier scheduler probes also
exercised cancellation, forced retraction, and organic KV-pressure retraction;
[validation](validation.md#radix-cache-lifecycle) describes the runnable checks.

## Probability sensitivity across execution paths

The September 23 current-source B300 audit of step-75500 matched all eight
greedy predictions against both ordinary HF and a Core-matched reference, in
all four graph/chunk modes. Strict probability comparisons still failed the
unchanged 0.1 threshold: maximum checked top-token error was 0.561879 against
ordinary HF and 0.493953 against Core-matched HF. Chosen decode-token errors
were at most 0.089736 and 0.149366 respectively. Graphs produced identical
checked probabilities; changing prefill chunk size produced differences.

Recovered controls also establish that all 23,441 checkpoint tensors convert
correctly and matched Core/HF execution gives identical full-vocabulary
probabilities on both initial prefixes. Conversion and graph execution have
positive evidence; strict cross-runtime probability equivalence needs further
numerical investigation. See [numerical findings](numerical-findings.md) for
the current and historical measurements, reference settings, and scope.

Applications requiring reference-equivalent probabilities or routing must
validate the actual checkpoint and dtype. Token agreement alone is insufficient.

## Readiness assessment

| Item | Assessment | Useful next action |
|---|---|---|
| Strict probability equivalence | Current-source measurements still fail 0.1 against ordinary and Core-matched HF; conversion and graph execution pass their controls. | If this is required, isolate chunk-dependent KDA/MoE arithmetic and routing on matched inputs. |
| Exact repeatability across tuning/batching | Historical MILES cold-retuning and concurrent graph/cache probes found divergence; fixed tuning caches matched in a controlled pair. | Treat this as a separate requirement when exact reproducibility matters; see [the controlled evidence](https://github.com/allenai/olmo-miles/blob/07887b783ab254577a6656168dc0e0d21aebfe3d/docs/measurements/kda-serving-determinism-20260906.md). Current fixture passes do not clear it. |
| Changed weights with radix and graphs | Validation gap closed for standalone TP1 fixtures; no implementation defect found. | Reuse the grouped probe when changing runtime or cache/update code. |
| TP2 with newer attention gains/scales | Additional coverage only; earlier production TP2 already works. | Optional two-GPU smoke when needed for capacity. |
| Multi-replica publication | Caller-owned integration behavior with existing MILES evidence. | Check admission/version boundaries when changing the caller's topology. |
| NGRAM speculation | Measured performance regression on the screened workload. | Keep opt-in; a useful trained draft needs a new performance screen. |
| FLA/Triton shim | Maintenance debt, not a failing inference test. | Replace or upstream when updating the pinned runtime. |

## Deployment checks

For a new deployment, validate its checkpoint/tokenizer, stop conditions,
cancellation, realistic request lengths, and concurrency. When publishing changed
weights, require successful updates on every replica, verify the policy version
and changed outputs, and invalidate cached state before admitting new requests.
These are topology-specific integration checks. Existing OLMo-MILES evidence and
its limits are recorded in [compatibility](compatibility.md#olmo-miles-qualification-limits).

## Performance and maintenance

Decode CUDA graphs already have a [matched production workload screen](https://github.com/allenai/olmo-miles/blob/07887b783ab254577a6656168dc0e0d21aebfe3d/docs/measurements/decode-cuda-graph-screen.md),
including a 50-update comparison. NGRAM speculation was also screened: mean
response tokens/GPU/s fell by 26.6% with about 15.5% draft acceptance. These
historical results inform defaults for that workload; they are not throughput
promises for another checkpoint or topology.

Remaining optimization work includes a more useful draft source and profiling
KDA/MoE decode overhead on the intended workload. The guarded FLA 0.5.2/Triton
compatibility shim remains maintenance work. An independent PyTorch KDA
recurrence already exists in `validation/reference.py` and runs in CPU CI.
