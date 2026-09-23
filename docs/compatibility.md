# Compatibility

`olmo-sglang` imports SGLang runtime internals, so compatibility is a tested
stack rather than a broad semantic-version promise. Installing this Python
package alone is expected to succeed, but calling `register()` requires a
compatible SGLang installation.

## Integration baseline

The pinned runtime uses the SGLang source revision recorded in `pyproject.toml`:

| Component | Tested version or revision |
|---|---|
| SGLang | `0.5.19.dev49+g3145136` / `3145136dcd1238754e0ea2b2ffd546532119c71c` |
| flash-linear-attention | `0.5.2` |
| PyTorch | `2.13.0+cu130` |
| Transformers | `5.12.1` |
| Triton | `3.7.1` |
| Python | `3.12` |

The source revision adds a bounded HTTP-readiness wait before optional GC
freezing to the audited `sglang-miles` base
`3bbb2812e2ca1defdee76f6ec09dbb2456c21c69`. It does not change model kernels.

This table is a known-good integration point, not a claim that adjacent SGLang
commits are incompatible. Run the validation suite whenever changing the
runtime because model registry, fused-MoE, linear-attention cache, and scheduler
interfaces are not stable extension APIs.

## Checkpoint contract

The Hugging Face configuration must declare
`architectures: ["Olmo3MoeForCausalLM"]` and one `layer_types` entry per model
layer. Supported entries are:

- `full_attention`;
- `sliding_attention`;
- `linear_attention` for OLMo KDA.

Full and sliding attention require `use_head_qk_norm=true`. Shared gains and
`qk_norm_per_head_gains=true` are supported; normalization across the entire Q/K
projection is not. Attention output gates can be absent or `elementwise`, with
projection biases following `attention_bias`. Headwise output gates are not
supported. Sliding attention requires an integer `sliding_window >= 2`.
Unsupported normalization and gate settings are rejected during model setup.

KDA configurations must provide the `linear_*` head, width, convolution, norm,
and negative-eigenvalue fields validated in `olmo_sglang.config`. KDA key and
value head counts must currently match, key head dimensions must be at most
256, and both head counts must divide the attention tensor-parallel size.

Routing currently supports softmax gating with L1-normalized expert weights.
The loader accepts HF-layout attention, KDA, dense SwiGLU, routed-expert,
latent-projection, and shared-expert tensors and maps them into SGLang's fused
representations.

## Runtime constraints

- KDA serving is GPU-oriented and requires FLA 0.5.2.
- The package carries a narrow FLA 0.5.2 source compatibility shim for newer
  Triton releases. Replacing or upstreaming that shim remains open work.
- Model and attention TP groups must match for KDA.
- Tensor-parallel head counts must divide evenly by the selected TP size.
- Radix caching for KDA uses SGLang's `extra_buffer` strategy to preserve branch
  states. Production rollout refreshes still require topology-specific testing.
- Speculative verification has a fused tree-capable correctness path, but
  performance depends on a useful trained draft source and has not been
  established as a serving default.

## OLMo-MILES qualification limits

OLMo-MILES owns its runtime pins, checkpoint precision and supported trainer
combinations. It stores router weights in BF16 and performs routing math in
FP32. Its trainer guards require TP=PP=CP=1; standalone inference TP evidence
does not qualify trainer TP or other RL topologies.

Live rollout routing replay requires OLMo-MILES' managed MILES router: the pinned
compiled SGLang router drops the expert-capture request field. The managed router
preserves expert metadata and request seeds, and uses independent health
connections. Replay passed all 19 routed layers through forward/recomputation
and weight refresh on the basic EP2 and production EP8 trainers. Measurements
use fixed-length batches, eager serving, disabled prefix caching, context 2,560
and response cap 512. Graph/cached replay and longer replay contexts require
separate qualification. See the [integration topology guide](https://github.com/allenai/olmo-miles/blob/main/docs/topology-and-length-guide.md#async-replay-coverage).

Basic EP2 async/replay completed 45 ordinary updates, default-cadence saving,
evaluation and verified export. Fresh-process restart and stalled-generation
replacement passed separately on EP2, with current actor weights published before
replacement samples were admitted. Production EP8 passed six ordinary updates
and checkpoint/export inspection; EP8 restart and replacement remain unexercised.
Its worker hung querying Ray after the head exited. OLMo-MILES corrected that
shell cleanup and verified it with real Ray in two CPU containers; the full GPU
run was not repeated. Recovery is opt-in; failed in-flight publication remains
terminal rather than providing transactional retry. See the
[async/replay and health guide](https://github.com/allenai/olmo-miles/blob/main/docs/disaggregated-rollout.md).

The observed post-refresh serving stall was active NVCC/ptxas compilation of
FlashInfer's lazy sampling module. OLMo-MILES warms filtered sampling before
managed admission and cache flush, keeping cold compilation inside startup
budgets. Operational health timeouts are unchanged. These integration adapters
do not change this package's runtime defaults or qualify other downstream users.

Deterministic serving has narrow guarantees. Serial/eager/cache-disabled requests
repeated within tested processes and controlled fresh processes on one B300.
Independent FLA retuning on paired B300s changed kernel choices and produced raw
logit divergence despite identical weights; matched tuning caches repeated
exactly in the controlled pair. A single-GPU L2-normalization fixture isolated a
BF16 output difference from alternative FP32 reduction grouping, but that
primitive is not proven to explain all full-model divergence. Concurrent
graph/cache serving failed repeatability. Batch invariance and exact trainer
log-probability agreement remain unqualified. See the
[reproducibility guide](https://github.com/allenai/olmo-miles/blob/main/docs/comparison-runs.md#serving-reproducibility)
and its controlled KDA evidence.

## Dependency policy

`pyproject.toml` deliberately does not resolve SGLang, PyTorch, CUDA, or Triton.
Those packages form one runtime unit and are supplied by the environment. FLA
is also installed explicitly only for KDA checkpoints. This avoids a small
extension package replacing the runtime's validated GPU stack during install.
