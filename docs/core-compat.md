# Optional Core-compatible execution

Set `OLMO_SGLANG_CORE_COMPAT=1` before constructing the SGLang engine to select
Core-compatible Olmo execution. The default is **off**. This is separate from
`OLMO_HF_MOE_CORE_REFERENCE`, which only affects the exported Transformers model.

```python
import os

import sglang
from olmo_sglang import register

os.environ["OLMO_SGLANG_CORE_COMPAT"] = "1"
register()
engine = sglang.Engine(
    model_path="/path/to/hero/hf",
    trust_remote_code=True,
    dtype="bfloat16",
    tp_size=1,
    cuda_graph_backend_decode="disabled",
    cuda_graph_backend_prefill="disabled",
)
```

For MILES, pass the environment variable to serving workers through the run's
`[launch.env]` table and explicitly disable serving prefill/decode graphs. Use an
image containing this serving revision; an environment variable cannot upgrade
an older image. The trainer remains OLMo-core.

## Execution changes

- Persistent Core-compatible expert weights, Torch grouped GEMMs, Core token
  permutation and FP32 weighted unpermutation. SiLU and down-projection results
  retain Core's BF16 rounding boundaries.
- Packed Core layouts for shared experts and dense MLP blocks.
- FP32 eager RMSNorm arithmetic for embedding, block, final, and ordinary Q/K
  normalization. Existing per-head normalization already follows this expression.
- Separate full-attention Q/K/V projections and Torch SDPA with explicit KV-head
  repetition. The compatible attention layer uses SGLang's request table and KV
  cache for packed requests, prefix chunks and decode, independent of the engine's
  normal full-attention kernel selection.
- KDA prefill uses Core's per-sequence FLA dispatch and `[K,V]` state orientation.
  Final and intermediate states are converted back to SGLang's `[V,K]` cache
  format. Populated prefix states are preserved; packed recurrent decode remains
  available. Both dispatch and orientation matter for long-prefix fidelity.

The mode requires the compatible OLMo-core runtime. It currently rejects TP/EP
larger than one, quantization, non-BF16 precision, CUDA graphs, speculative
decoding, RoPE, attention projection biases and sliding attention. These are
explicit initial scope limits, not a claim that compatible execution cannot
support those features later.

## Weight publication

There are no derived weight caches to invalidate. Standard HF per-expert names
and MILES fused expert publications load into the actual tensors used for
computation, preserving storage addresses across updates. External fused payloads
retain SGLang's gate-then-up order; the loader converts to internal up-then-gate
order. Down-projection and dense weights have deliberate strides that expose
Core's packed layouts without forward-time copies. Raw internal SGLang state
exports are consequently mode-specific; use HF names/fused publication contracts
when changing modes.

## Validation and performance

The mode is a numerical compatibility option, not a promise of bitwise equality
for every batch shape or cached decode. Different GEMM shapes and chunk versus
recurrent KDA execution can still differ. Compare actual Core scores with serving
behavior probabilities on the same tokens. Measure throughput after warm-up and
record graph, batching, chunking, context and sampling settings.

The accompanying Open Instruct benchmark freezes real RL prompt identities and
uses identical token IDs across checkpoints. It reports generated-token
probability differences and Core/serving likelihood-ratio distributions, together
with timed generation throughput. It does not establish learning quality or
complete responses when a fixed token budget is used.

## Graph-compatible rounding mode

`OLMO_SGLANG_CORE_COMPAT=rounding` is a separate, experimental middle option.
It preserves BF16 SiLU/multiply/down-projection rounding, uses FP32 expert
weighting/reduction, and selects Core-style FP32 RMS norms. It retains ordinary
SGLang weight layouts, full attention and KDA dispatch, and allows decode graphs.
Dense/shared MLPs already use separate BF16 activation operations. This mode
does not use Core grouped GEMMs or the full reference's attention/KDA changes.

Use TP1/EP1, unquantized BF16 and the Triton MoE backend. Alternative quantization,
interleaved gate/up storage, speculation and the full mode's unsupported model
geometries are rejected. The implementation calls the pinned SGLang alignment
and GEMM interfaces directly; it does not install activation hooks or change
global kernels. Standard HF and fused weight publication retain existing storage.
It is off by default and requires its own workload/graph qualification.

Within rounding mode, small Triton kernels fuse activation, FP32 weighted
reduction and RMS normalization while retaining explicit BF16 boundaries.
`OLMO_SGLANG_ROUNDING_KERNELS=torch` selects the original separate tensor
operations as a diagnostic control; the default is `fused`. This setting does
not change ordinary serving or the full reference mode. Fusion reduces launches
and temporary buffers, but FP32 reduction association can still differ. Keep
probability comparisons separate from performance measurements.
The diagnostic values `moe` and `norms` fuse only that component, allowing its
effect on performance and full-model probabilities to be isolated.
