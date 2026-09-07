# Status

The implementation is suitable for controlled inference experiments and is
integrated into the OLMo-MILES runtime. Ordinary autoregressive decoding is the
production default. Experimental cache and speculative features require the
feature-specific gates below before enabling them in a new topology.

## Implemented

- Native full and sliding-window attention.
- Native OLMo KDA prefill and cached decode through FLA 0.5.2.
- Exact optional negative-eigenvalue semantics using `2 * sigmoid(beta)`.
- Unequal K/V widths in the recurrent-state cache.
- Headwise Q/K normalization and optional elementwise attention output gates.
- Peri-LN residual ordering.
- Dense SwiGLU and routed MoE, including OLMo routing normalization and restored
  top-k scaling.
- Optional latent down/up projections and a full-width shared expert.
- HF-layout loading into SGLang's fused QKV, SwiGLU, and MoE tensors.
- Standard tensor-parallel sharding for KDA projections, convolution state, and
  recurrent state.
- Branch-capable KDA radix snapshots through SGLang's `extra_buffer` strategy.
- Fused speculative target verification for chains and explicit trees, with
  CUDA-graph replay coverage.
- A package-local FLA 0.5.2 compatibility shim for newer Triton releases.

## Readiness by area

| Area | Current position |
|---|---|
| Full/sliding attention inference | Implemented; validate each production checkpoint and runtime |
| Ordinary KDA inference | Correctness path implemented; GPU and FLA required |
| KDA radix caching | Branch lifecycle validated locally; production refresh/load topology remains a gate |
| Tensor parallelism | TP=1 versus TP=2 greedy parity established under the tested constraints |
| Speculative decoding | Correctness path implemented; trained-checkpoint performance is unscreened |
| MILES policy refresh | Changed-weight async/replay exercised on EP2 and EP8 with eager, cache-disabled serving; other combinations retain the gates below. See [integration limits](compatibility.md#olmo-miles-qualification-limits). |

Current OLMo-MILES experiments and their precision, determinism, replay and
recovery limits are recorded in [compatibility](compatibility.md#olmo-miles-qualification-limits).
Standalone feature implementation does not imply that every RL combination is qualified.

## Required production correctness gates

- Compare prompt logits, greedy tokens, and MoE router selections against the
  training/reference implementation at the intended serving dtype.
- Validate changed actor weights on every rollout replica and prove generation
  changes at the new policy version.
- Exercise production checkpoint, tokenizer, stop conditions, cancellation,
  request distribution, and concurrency in the intended serving topology.
- If radix caching is enabled, prove first-request misses and subsequent reuse
  after each weight refresh across every replica.

## KDA and scheduler gates

Completed focused coverage includes prefix hits, divergent branches, eviction,
request isolation, cleared-slot reuse, mixed-length chunked prefill,
cancellation, forced scheduler retraction, organic KV-pressure retraction, and
idle cache flush/reload behavior.

Before enabling those paths for a new workload, repeat them with the production
checkpoint and realistic request lengths. Recurrent state cannot be split from
an arbitrary compressed radix edge like per-token attention KV, so cache-hit
behavior depends on tracked state boundaries.

## Performance and hardening work

- Compare decode CUDA-graph replay with the matched packed eager path.
- Replace or upstream the FLA 0.5.2/Triton compatibility shim.
- Expand BF16 and production-dimension coverage across real sparse-MoE layers.
- Measure speculative proposed tokens, acceptance, verifier latency, and net
  throughput with a useful trained draft source.
- Profile and reduce remaining unfused KDA/MoE decode glue.
- Optionally add a slow, dependency-light CPU reference recurrence for broader
  portable coverage.

The packed one-token Triton recurrence has shown a large microbenchmark
improvement over invoking an FLA chunk at batch-one model-shaped dimensions,
but microbenchmarks are not an end-to-end serving claim. Use the validation
ladder in [validation](validation.md) for any runtime or checkpoint change.
