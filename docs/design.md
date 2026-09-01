# Design

## Integration boundary

`olmo_sglang.register()` is the public Python API. It registers
`olmo_sglang.models` in the current process and configures spawned workers to
discover the same package. The `models` package performs the small import-time
compatibility registrations required before SGLang constructs the model.

The runtime implementation is intentionally split by responsibility:

```text
src/olmo_sglang/
  registration.py       public SGLang discovery API
  config.py             supported-checkpoint validation
  activations.py        narrow fused-MoE activation fallback
  routing.py            OLMo router numerics
  models/               SGLang causal-LM implementation and weight loading
  kda/                  KDA layer, cache backend, and Triton kernels
  validation/           parity, smoke, and reference tooling
```

`olmo_sglang.kda_backend` is a compatibility import retained because the
OLMo-MILES preflight loads that historical module path. New code should import
`olmo_sglang.kda.backend`.

## Model execution

`Olmo3MoeForCausalLM` composes SGLang tensor-parallel embeddings, full or
sliding attention, OLMo KDA layers, peri-LN residual ordering, dense SwiGLU or
routed MoE blocks, and the SGLang logits processor. The loader converts
HF-layout projections and experts into fused SGLang tensors while preserving
OLMo-specific router scaling and optional latent/shared-expert structure.

FLA is not imported for attention-only checkpoints. The external model package
still registers the KDA configuration hook so SGLang can prepare hybrid cache
metadata when a KDA checkpoint is selected.

## KDA state

OLMo KDA has a convolution window and a recurrent matrix. The recurrent state
supports unequal key and value widths and is stored in FP32 while convolution
state follows the model activation dtype. Tensor parallelism shards projection
outputs, convolution channels, recurrent heads, and associated parameters.

Prefill uses FLA 0.5.2 so OLMo's optional negative-eigenvalue behavior remains
exact: raw beta logits are interpreted as `2 * sigmoid(beta)`. One-token decode
uses an overlay-owned packed Triton recurrence with the same semantics.

## Radix caching

An attention KV cache can split state per token; recurrent KDA state cannot.
The KDA backend therefore snapshots the convolution window and recurrent matrix
at tracked prefix-tree boundaries. SGLang's `extra_buffer` strategy stores FLA
intermediate states and applies copy-on-write at branches. Cache invalidation is
required whenever policy weights change.

## Speculative verification

Target verification writes proposal states to SGLang's speculative scratch
pool and leaves committed state untouched. The verifier supports linear chains
and explicit tree parents. SGLang's normal commit path selects the accepted
state, preserving rejection and rollback semantics. CUDA uses a fused Triton
recurrence; CPU execution remains the independent eager reference.

## Validation boundary

The `validation` package and `tools/` directory are not part of request-time
model execution. They exist to compare implementations, generate deterministic
fixtures, and exercise scheduler behavior. Keeping them separate makes it
clear which code is serving-critical.
