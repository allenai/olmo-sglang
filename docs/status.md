# Validation status

`olmo-sglang` supports standalone OLMo inference in the
[pinned SGLang runtime](compatibility.md#integration-baseline). The implementation
has CPU regression tests, GPU kernel tests, and whole-engine checks. The table
below records what those checks establish and where evidence is still missing.

## Evidence by feature

| Area | Established coverage | Limits |
|---|---|---|
| Full/sliding attention | Native attention, headwise Q/K normalization, optional per-head gains, scalable softmax, and elementwise output gates. Tiny BF16 full and biased sliding-attention/HF parity includes chunked prefill and decode graphs. | Triton attention was exercised in the current audit. Only the [supported configuration](compatibility.md#checkpoint-contract) is accepted. Full-checkpoint probability parity remains unresolved. |
| KDA prefill and decode | FLA 0.5.2 prefill, packed Triton decode, FP32 recurrent state, unequal K/V widths, and optional `2 * sigmoid(beta)` semantics. GPU recurrence and CUDA-graph tests pass. | These checks establish operator and fixture correctness, not numerical equivalence for every checkpoint. |
| Dense and sparse MoE | Dense SwiGLU, FP32 routing with restored top-k scaling, latent/shared experts, peri-LN, and HF-to-fused weight loading. Tiny hybrid-MoE serving is compared with HF. | BF16 kernel rounding can change full-model probabilities and expert choices; see [numerical findings](numerical-findings.md). |
| Weight updates | Tiny TP1 live Q/K-gain and scale updates match a fresh changed-checkpoint engine under decode graphs. Every begin, bucket, and end response must succeed. Regression coverage includes expert-name resolution and gate-bias loading. | This is a standalone engine check. Multi-replica publication and policy-version ownership belong to the caller. |
| KDA radix caching | Branch copy, eviction, isolation, cleared-slot reuse, mixed-length chunked prefill, and idle flush/reload have focused coverage. | The reload probe uses the same weights to isolate invalidation. Changed-policy publication with cache enabled still needs a matched fresh-engine check in the intended topology. Cache hits depend on tracked recurrent-state boundaries. |
| Tensor parallelism | KDA projection/state sharding and per-head gain loading have rank-level tests. A [production TP1/TP2 screen](https://github.com/allenai/olmo-miles/blob/07887b783ab254577a6656168dc0e0d21aebfe3d/docs/measurements/tensor-parallel-screen.md) matched eight greedy tokens. | Historical TP2 max chosen-token logprob difference was 0.04458. The September 23 local checks use one GPU; the newer gain/scale combination has no whole-engine TP2 result. |
| Speculative decoding | Chain/tree target verification, scratch-state isolation, and CUDA-graph replay have correctness tests. | The [trained-checkpoint NGRAM screen](https://github.com/allenai/olmo-miles/blob/07887b783ab254577a6656168dc0e0d21aebfe3d/docs/measurements/ngram-speculative-screen.md) was slower than ordinary decode. Keep it opt-in; a better draft source needs a new screen. |

See [the September 23 audit](validation.md#september-23-2026-validation-audit) for
current commands, runtime identity, and results. Earlier scheduler probes also
exercised cancellation, forced retraction, and organic KV-pressure retraction;
[validation](validation.md#radix-cache-lifecycle) describes the runnable checks.

## Known numerical limitation

The step-75500 full-checkpoint comparison matched eight greedy predictions but
failed its 0.1 absolute log-probability threshold: maximum checked error was
0.3118 with unchunked prefill and 0.5619 with chunking plus decode graphs.
Tiny-model passes do not clear this result. The recorded Core/HF comparison also
found discrepancies, so the error cannot yet be attributed entirely to the
SGLang implementation. See [numerical findings](numerical-findings.md).

Applications requiring reference-equivalent probabilities or routing must
validate the actual checkpoint and dtype. Token agreement alone is insufficient.

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
