# Core-compatible execution modes

With `OLMO_SGLANG_CORE_COMPAT` unset (or `auto`), the adapter selects fused
rounding for a validated model configuration and ordinary serving otherwise.
Explicit `0` / `off` opts out; `rounding` forces the rounding path; `1` / `full` selects
the slow eager numerical reference. The resolved mode is logged at model
construction and retained as `model.core_compat_mode`.

Automatic selection checks model geometry, normalization, and gating against
`_ROUNDING_PROFILE` in `core_compat.py`.
It requires unquantized BF16 (the loader-resolved dtype, including `--dtype auto`
when the checkpoint resolves to BF16), TP1/EP1, `auto`/`triton` MoE backend, full or disabled
decode graphs, disabled prefill graphs, no speculation and no `torch.compile`.
Checkpoint paths do not determine selection. Other configurations keep ordinary
arithmetic. Selection is scoped to construction so later models do not inherit
it. Graph settings and attention backends are not changed.

Fused rounding preserves the intended BF16 boundaries while reducing kernel
launches. It does not guarantee exact OLMo-core parity or improved training
quality. Validate numerical behavior and throughput on the intended hardware,
checkpoint, and runtime.

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
    model_path="/path/to/olmo-hf-checkpoint",
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
The norm kernel also folds the peri-LN residual add after the norm and the
embedding scale before it, each at the same BF16 rounding point as the
separate tensor operations.
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

## Optional fused router top-k

Expert routing defaults to `OLMO_SGLANG_ROUTING_KERNELS=torch`, retaining the
existing separate PyTorch tensor operations and numerical behavior. These
operations run on the GPU for CUDA inputs and can be captured in CUDA graphs;
they are not Python arithmetic on expert values.

To opt into SGLang's fused softmax/top-k/renormalize kernel, set this in every
serving worker's launch environment **before engine startup and graph capture**:

```bash
export OLMO_SGLANG_ROUTING_KERNELS=fused
```

To turn it off, restart the engine with the variable unset or explicitly set to
`torch`. Changing an environment variable after capture does not replace kernels
in an existing CUDA graph. Accepted values are `torch` and `fused` (case and
surrounding whitespace are ignored); other values raise an error when routing
runs. CPU inputs and normalization other than L1 or no normalization retain the
tensor path even when `fused` is requested.

This switch is independent of `OLMO_SGLANG_CORE_COMPAT` and
`OLMO_SGLANG_ROUNDING_KERNELS`. Selecting a compatibility/rounding mode does not
cancel an explicit fused-routing opt-in. Leave routing at `torch` when requiring
the numerical behavior qualified in [PR #15](https://github.com/allenai/olmo-sglang/pull/15).

Fusion reduces kernel launches and intermediate work, but changes FP32 reduction
order and folds constant scale factors into one multiply. Routing weights can
differ in their last bits, expert ordering/tie behavior can differ, and downstream
BF16 outputs and token logprobs can change. It is not a bitwise-parity option.

The standalone H100 screen for [PR #10](https://github.com/allenai/olmo-sglang/pull/10)
measured this tradeoff on the frozen Olmo 3.5 KDA MoE workload at concurrency 64:

| Measurement | Tensor routing | Fused routing |
|---|---:|---:|
| Fixed-length throughput change | baseline | +3.15% |
| Mixed-length throughput change | baseline | +2.30% |
| Mean absolute trainer/inference logprob difference (nats/token) | 0.03699278 | 0.03702442 |
| Token probability-ratio outlier rate | 1.74415% | 1.77713% |

The ratio is `exp(trainer_logprob - inference_logprob)` on the same scored tokens;
an outlier lies outside `[0.8, 1.2]`. The increase is **0.03298 percentage points**,
about 330 additional flagged tokens per million. This metric counts threshold
crossings, not their severity. The 95%
prompt-bootstrap interval for the change was approximately +0.00318 to +0.06560
percentage points; the interval for the mean absolute error change crossed zero.

These are short screening results. The final repeated control was preempted.
They do not establish a learning-quality impact or a speedup on B300, nor an
incremental gain when combined with PR #15. The measured outlier increase failed
the original no-additional-divergence criterion, so fused routing remains opt-in.
The later [second-pass composition panel](decode-composition.md) completed
bracketing controls on both H100 and B300 and measured the router's incremental
effect with paired prompt-cluster intervals. That B300 result improves aggregate
agreement relative to its aligned control; the older screen is not a universal
accuracy label for this option.
Recheck throughput and same-token trainer agreement for the intended checkpoint,
runtime, and workload before choosing it. See the
[issue #31 report](https://github.com/allenai/oe-planning/issues/31#issuecomment-6386644696)
for the broader comparison and accepted default configuration.
