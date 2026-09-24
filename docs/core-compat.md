# Core-compatible execution modes

With `OLMO_SGLANG_CORE_COMPAT` unset (or `auto`), the adapter selects fused
rounding for the qualified hero family and ordinary serving otherwise. Explicit
`0` / `off` opts out; `rounding` forces the rounding path; `1` / `full` selects
the slow eager numerical reference. The resolved mode is logged at model
construction and retained as `model.core_compat_mode`.

Automatic selection covers the measured 12.5B hero profile: hidden size 1024,
16 layers (14 KDA, full attention at layers 7 and 15), 512 experts/top-16,
latent width 512, expert/shared width 1024, dense width 8192 and its norm/gating
configuration. `_ROUNDING_PROFILE` in `core_compat.py` is the exact contract.
It requires unquantized BF16 (the loader-resolved dtype, including `--dtype auto`
when the checkpoint resolves to BF16), TP1/EP1, `auto`/`triton` MoE backend, full or disabled
decode graphs, disabled prefill graphs, no speculation and no `torch.compile`.
Checkpoint paths and EMO ancestry do not determine selection. Other configurations
keep ordinary arithmetic. Selection is scoped to construction so later models do
not inherit it. Graph settings and attention backends are not changed.

Fused rounding retained essentially ordinary throughput on H100: 993 → 1,010
(base, batch 4), 988 → 1,000 (EMO SFT), 983 → 1,004 (non-EMO SFT), and
2,486 → 2,513 tokens/s (EMO, batch 16). Treat these as similar speed.
Fused/tensor rounding matched exactly on 73,728 generated tokens and 8,192
fixed-prefix scores. Agreement against actual Core remains mixed; this is not
proof of exact Core parity or improved RL learning. The default preserves the
intended BF16 boundaries without the earlier tensor control's 24–25% slowdown.
See the [Open Instruct report](https://github.com/allenai/open-instruct/blob/3ac5615fb/docs/miles/measurements/fused-rounding-20260923.md)
for reproducible workload, source/image pins, ablations and probability tables.
That report's image predates automatic selection and requires explicit `rounding`.
Requalify other hardware and updated runtimes; the measured GPU was H100.

## Full eager reference

Set `OLMO_SGLANG_CORE_COMPAT=1` before constructing the SGLang engine to select
the full Core reference. This slower mode remains opt-in. This is separate from
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

`OLMO_SGLANG_CORE_COMPAT=rounding` explicitly selects the rounding option,
which is also the automatic default for the qualified profile above.
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
Explicit selection outside the automatic profile requires its own workload/graph
qualification.

Within rounding mode, small Triton kernels fuse activation, FP32 weighted
reduction and RMS normalization while retaining explicit BF16 boundaries.
`OLMO_SGLANG_ROUNDING_KERNELS=torch` selects the original separate tensor
operations as a diagnostic control; the default is `fused`. This setting does
not change ordinary serving or the full reference mode. Fusion reduces launches
and temporary buffers. The fused reductions also follow the pinned PyTorch CUDA
accumulation grouping: four independent accumulators followed by their ordered
combination, with the norm's cross-warp fold before its intra-warp reduction.
This matters because rare changed BF16 outputs can alter full-model routing.
Top-k above 16 and norm widths above 4096 (or non-vectorizable widths at least
128) use the tensor fallback. The exact-equality tests and workload comparisons
must be rerun when changing the runtime; this is not a promise of bitwise Core
agreement for arbitrary cached decoding. Keep probability comparisons separate
from performance measurements.
The diagnostic values `moe` and `norms` fuse only that component, allowing its
effect on performance and full-model probabilities to be isolated.
