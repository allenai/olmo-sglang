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
on the change: -4.8e-5 to +1.3e-4 nats). Decode and prefill-mode scoring differ
from each other by 0.0358 nats/token on the same tokens, as much as either differs
from OLMo-core, so the remaining divergence is accumulation-order noise shared by
any two implementations of this model rather than a KDA-specific mismatch.
Outlier-rate differences of a few hundredths of a percentage point are within
that noise.

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
