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

## SwiGLU rounding

A fused SwiGLU kernel can evaluate the activation and multiplication in FP32
before casting once to BF16. Separate operations can instead round `silu(gate)`
to BF16 before multiplying by `up`. These paths can produce different outputs
from the same inputs; small differences can affect downstream expert routing.
Reduction order and grouped versus separate matrix multiplications can introduce
additional differences.

[Core-compatible execution modes](core-compat.md) provide explicit arithmetic
controls for investigating these differences. They do not guarantee bitwise
agreement for arbitrary batch shapes, cached decoding, or hardware. Measure
numerical agreement separately from throughput and training quality.
