# Numerical behavior and limits

## Probability sensitivity across execution paths

Matching greedy tokens does not establish matching probabilities or expert
routing. Full-checkpoint comparisons have found probability differences between
SGLang, ordinary Hugging Face execution, and OLMo-core-compatible references,
even when short greedy continuations agree.

Prefill chunk size, batch composition, kernel tuning, and BF16 rounding can
change results. Passing checkpoint-conversion checks or CUDA-graph comparisons
does not establish probability equivalence across other execution paths.
Tiny-model tests also do not establish full-checkpoint or long-context parity.

Applications requiring close trainer/serving probabilities should compare the
same token prefixes with matched weights, dtype, attention settings, and
reference arithmetic. Check logits or log probabilities and expert routing,
not just generated token IDs. Use the bounded serving checks in
[validation](validation.md) and validate the intended request lengths and
concurrency.

## KDA query/key normalization rounding

OLMo-core and the overlay's prefill path both call the FLA chunk kernel with
in-kernel L2 normalization, which stores the normalized query and key as BF16
tensors before the recurrence. The packed decode kernel normalizes in FP32 and
rounds the normalized query and key to BF16 at the same point, so decode and
prefill share that rounding boundary. The division-versus-reciprocal form of the
normalization made no measurable difference at the kernel level.

Measured on the frozen H100 reference (1,185,618 identical output tokens, same
tokens scored by the pinned OLMo-core model, three arms in one job):

| Serving arm vs OLMo-core | Mean abs error (nats/token) | Signed mean | Ratio outliers outside [0.8, 1.2] |
|---|---:|---:|---:|
| Decode, FP32 normalized query/key (previous) | 0.03699 | +0.00206 | 1.744% |
| Decode, BF16 normalized query/key (current) | 0.03704 | +0.00019 | 1.788% |
| Prefill-mode scoring of the same tokens | 0.03699 | +0.00012 | 1.770% |

The BF16 rounding removes the systematic decode bias relative to both the trainer
and prefill; the mean absolute error is unchanged (prompt-clustered 95% interval
on the change: -4.8e-5 to +1.3e-4 nats). The outlier rate rises by a small but
measurable amount: +0.044 percentage points with a paired prompt-clustered 95%
interval of +0.019 to +0.071, about four extra tokens per 10,000 outside the
[0.8, 1.2] ratio band. The change is therefore a documented trade-off (bias
removed, slightly more clipped tokens), not a uniform improvement.

Per-token errors are uncorrelated along a sequence (lag-1 autocorrelation within
0.001 of zero), so they add like independent noise: at 1,024 tokens the standard
deviation of the summed per-sequence error is 2.1 nats against an independence
prediction of 2.06. The bias does not average out: with FP32 query/key the mean
summed error over a 1,024-token response was +2.27 nats, as large as the random
spread; with BF16 query/key it is +0.41 nats.

Decode and prefill-mode scoring differ from each other by 0.0358 nats/token on
the same tokens, as much as either differs from OLMo-core. Three implementations
being pairwise equidistant is what independent accumulation-order noise produces,
and it is the leading explanation for the remaining 0.037, but it does not rule
out a smaller shared component underneath. Distinguishing the two would need a
lower-noise reference (for example FP32 rescoring), not more BF16 arms.

## SwiGLU rounding

A fused SwiGLU kernel can evaluate the activation and multiplication in FP32
before casting once to BF16. Separate operations can instead round `silu(gate)`
to BF16 before multiplying by `up`. These paths can produce different outputs
from the same inputs; small differences can affect downstream expert routing.
Reduction order and grouped versus separate matrix multiplications can introduce
additional differences.

The dense MLP, the shared expert, and the fallback for unaligned routed widths
apply SwiGLU through `rounding_kernels.silu_mul` on BF16 CUDA tensors. That
kernel rounds `silu(gate)` to BF16 before the product, so it is bitwise equal to
the separate operations (exhaustively tested over finite BF16 gates) while
launching one kernel instead of three.

[Core-compatible execution modes](core-compat.md) provide explicit arithmetic
controls for investigating these differences. They do not guarantee bitwise
agreement for arbitrary batch shapes, cached decoding, or hardware. Measure
numerical agreement separately from throughput and training quality.
