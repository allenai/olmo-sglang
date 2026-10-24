# Inference options and measured tradeoffs

This is a menu of inference building blocks for Olmo 3.5 KDA MoE. Measure
throughput and trainer agreement together; numerical divergence is a cost to
document and evaluate case by case. The current phase develops and benchmarks
individual options. Choosing production combinations, real-workload RL trials,
and learning-quality evaluation are a later phase.

## What is enabled by default

On main at `b75e163d6bc18605d709f7b3a0064ebe2d857070`, the accepted
[PR #15](https://github.com/allenai/olmo-sglang/pull/15) composition is automatic
for the qualified model/runtime:

| Component | Default behavior and scope |
|---|---|
| Dense/shared SwiGLU fusion (#11) | BF16 CUDA inputs without an expert filter use the fused activation, retaining the separate BF16 rounding points. |
| Norm/residual and embedding-scale fusion (#12) | Active in fused rounding mode; automatic compatibility selection chooses this mode for the validated model and serving configuration. |
| FP32 router storage (#13) | Router weights are stored in FP32, eliminating repeated casts while preserving the BF16-valued weights. |
| Tuned expert GEMMs (#14) | Packaged B300 tiles are selected for unquantized TP1/EP1, E512/N1024/input512. H100 retains upstream defaults. Explicit config-directory overrides are honored. |
| Fused router top-k (#10) | **Off by default.** Set `OLMO_SGLANG_ROUTING_KERNELS=fused` before engine startup to opt in; unset or `torch` retains the tensor path. Restart to switch captured graphs. |

See [compatibility modes](core-compat.md) for automatic-selection restrictions
and diagnostic overrides. These defaults require an image containing the merged
source; environment variables do not update an older image.

## Measured numerical tradeoff: fused routing

The historical standalone H100 C64 screen measured **+3.15% fixed-length** and
**+2.30% mixed-length** throughput. Mean absolute trainer/inference logprob error
was **0.03699278 → 0.03702442 nats/token**, and token probability-ratio outliers
were **1.74415% → 1.77713%**. An outlier has
`exp(trainer_logprob - inference_logprob)` outside `[0.8, 1.2]` on the same token.

That is an increase of **0.03298 percentage points**, about 330 additional flagged
tokens per million. The 95% prompt-bootstrap interval for the outlier increase
was positive (+0.00318 to +0.06560 percentage points); the interval for the mean
absolute error change crossed zero. The final repeated control was preempted.
This is a short screening measurement, not evidence about learning quality.
Its incremental gain and divergence on top of #15, and its B300 performance,
remain unmeasured. See [the option's full documentation](core-compat.md#optional-fused-router-top-k).

## Fresh profile of the default composition

The [2026-10-24 run](https://beaker.org/ex/01M5QCKBKR7W8Z72QYDAZCC5VV) compares
baseline `92b35cdf` with main `b75e163d` on the same B300. Both use the pinned
SGLang `3145136d` runtime, the same checkpoint and frozen C64 fixed/mixed workload,
BF16 TP1/EP1, memory fraction 0.7, prefill chunk 8192, context 12288, full decode
graphs, radix/Mamba caching, logprobs, and route capture/return. Core compatibility,
rounding-kernel, routing-kernel, and MoE-config-directory overrides were unset.
The current-main source and packaged tile table are recorded in the payload
manifest. Profiling runs separately after the three timed repeats per workload.

| Summed graph GPU work per decode step | Original baseline | Current default |
|---|---:|---:|
| Expert GEMMs | 2.923 ms | 2.710 ms |
| Other GEMMs and reductions | 0.846 ms | 0.849 ms |
| KDA recurrence | 0.353 ms | 0.353 ms |
| Router selection/normalization, partial attribution | 0.257 ms | 0.255 ms |
| Tensor copies/casts | 0.227 ms | 0.162 ms |
| RMS normalization | 0.123 ms | 0.130 ms |
| SwiGLU/SiLU | 0.103 ms | 0.078 ms |
| Remaining graph kernels | 0.539 ms | 0.416 ms |
| **Total graph work** | **5.371 ms** | **4.952 ms** |
| **Graph kernel count** | **705** | **625** |

The categories partition graph kernels by name; generic router divides, scales,
and casts are not all assigned to the router row. Norm kernels now include
additional fused work, so their individual duration need not fall. Current-main
expert work splits into **1.797 ms up/gate** and **0.913 ms down**. Expert GEMMs
remain about **55%** of graph work; that establishes the largest component,
not an achievable optimization budget.

Each trace contains 64 decode steps. The first few steps have much cheaper expert
work; over the last 32 steps, total graph work is **5.677 → 5.238 ms**. Report both
windows rather than treating the early-decode average as sustained long-response
latency. Route diversity/cache behavior and profiler effects have not been
separated as causes of this transient.

Current-main and closing-baseline output tokens/logprobs exactly match the opening
baseline and the previous qualified composition on all **384 trajectories /
1,035,090 tokens**. The 64 separate profile requests (8,192 tokens) also match.
This is unchanged agreement on the frozen workload, not zero trainer error.
Runtime Python/JSON hashes match the source manifest; kernel checks verified
the packaged tiles for both expert GEMMs. The job completed in 14.45 minutes.

Unprofiled, bracketed throughput in this repeat:

| C64 workload | Bracketed baseline | Current default | Change | Closing-control drift |
|---|---:|---:|---:|---:|
| Fixed length | 8,384 tokens/s | 8,879 tokens/s | +5.90% | +2.69% |
| Mixed lengths | 5,207 tokens/s | 5,622 tokens/s | +7.98% | +0.30% |

All three paired waves improved in both cells. The mixed result reproduces the
prior approximately 8% gain; the fixed result has more control drift than the
earlier +8.71% panel. These are short repeat measurements, not an additional
optimization or a new production forecast. The slower uncached fixed wave
recurs on the same frozen requests; its admission/cache explanation remains a
hypothesis. Comparing corresponding waves preserves that workload distinction.

### Reading the timeline correctly

GPU kernel sums, GPU-active time, and elapsed step span are different quantities.
The fresh baseline/current traces have GPU-active unions of **5.504/5.096 ms**
and spans of **6.764/5.951 ms** per step, including work outside the graph.
Profiler perturbation and asynchronous overlap prevent treating the gap as a
known recoverable CPU tax. The original baseline trace's CPU event wait averaged
0.609 ms/step, about 88% overlapping recorded GPU activity; the fresh baseline
wait is only 0.016 ms despite nearly identical graph work.

The pinned source already uses pinned, nonblocking route D2H copies in overlap
mode (`state_capturer/base.py`, `managers/utils.py`). It gathers full request
routes at completion (`scheduler_components/batch_result_processor.py`) and
encodes populated tensors in the detokenizer. It does **not** base64-encode a
full route history every decode step. Completion gather/serialization, clones,
host scatter, and capture-buffer copies remain potential costs to investigate.

The roughly 24 GB expert-weight size and peak bandwidth give a rough reference,
not proof that expert GEMMs are optimal. Actual expert coverage, transferred
bytes, and bandwidth/utilization counters are needed to establish that claim.
Likewise, hundreds of short graph kernels motivate fusion experiments but do
not each incur an independent Python launch during graph replay.

## Next building blocks to measure

These are candidates, **not implemented options or measured speedups**. Component
budgets below describe existing work; they are not savings forecasts and cannot
be added together. Each implemented arithmetic option should have an explicit
switch, a documented default, and its own throughput/agreement row.

| Candidate | Concrete change | Evidence and tradeoff to measure |
|---|---|---|
| Routing preparation/tail fusion | Emit an FP32 hidden-state copy from the pre-MLP norm; fuse the selected-weight norm/divide/scale/index-cast tail while retaining existing softmax/top-k. | Targets small copies and launches without replacing expert selection. Preserve each rounding operation and the reduction order; measure actual agreement rather than assuming parity. |
| Shared-expert add/norm fusion | Fold the rounded shared+routed addition into the post-MLP norm, retaining the intermediate BF16 sum. | A small, bounded elementwise opportunity adjacent to #12. Confirm aliasing and residual semantics. |
| KDA projection packing | Combine compatible QKV/f1/beta/g1 projections that read the same hidden input; consider grouped treatment of dependent second-stage projections separately. | The original trace attributes 0.327 ms/step to six KDA input/gate GEMMs across 14 layers, a subset of dense-GEMM time. Changed GEMM shapes may change numerics or regress performance. Preserve f1→f2 and g1→g2 BF16 boundaries, loader mappings, and refresh behavior. |
| Decode-specialized expert kernels | Replay actual route distributions, split up/gate and down measurements, inspect sparse padding and weight traffic, then test a targeted kernel/layout change. | The largest component at 2.710 ms. C64/top16 gives 1024 assignments across 512 experts while current tiles use 16 rows. Padding alone does not prove wasted wall time; prior extra tile/TMA screens did not help. |
| Expert activation epilogue | Fuse gate/up output with activation while retaining BF16 GEMM output, BF16 SiLU, and BF16 product boundaries. | Removes a small intermediate/launch cost. Tensor-core accumulation and register-pressure changes require measurement. |
| Route transport/buffering | Attribute per-step clones/scatter and completion-time index gather, base64 and IPC copies; test buffer reuse or transport changes where measured. | Async copies and completion-only route gathering already exist. Preserve ordering, cache/retraction semantics, and historical token/logprob records. |
| Host scheduling/overlap | Attribute GPU gaps to concrete CPU work using timeline correlations and an unprofiled control. | CPU wait duration alone is insufficient. The earlier admission probe had 12.6% control drift and did not establish a causal gain. |

First screen the contained routing-tail/shared-add fusions and KDA
projection packing, while using actual routes and hardware counters to decide
whether a larger expert-kernel effort is justified. Route transport remains an
attribution task before assigning it an expected gain. Prefill-only changes stay
secondary for this decode-heavy workload.

## Measurement record for each option

- Record source/runtime/model revisions, hardware, concurrency, memory/cache
  settings, graph mode, lengths, sampling, and whether routes/logprobs are returned.
- Use the frozen requests with short repeated before/option/after runs. Record
  individual waves and control drift; separate startup, compilation and profiling.
- Save token IDs and inference logprobs. Compare trainer and engine on **the same
  tokens**; if free generations differ, force-score a common reference.
- Report throughput alongside mean absolute logprob error, signed bias, error
  quantiles/tails, and probability-ratio outliers, with paired prompt-level
  uncertainty. Include baseline variability and incomplete/preempted attempts.
- Document scope and limitations: an isolated kernel win is not a whole-engine
  win; a non-bitwise result is not automatically unacceptable; changed GEMM
  shapes do not guarantee either disagreement or agreement.
- Keep individual results separate from composition effects. Real-workload RL,
  learning-quality tests, and final production selection follow this building-block
  phase.

The frozen workload, raw traces, source manifests, numerical vectors and ledgers
for the first pass are retained in `runs/issue31-20261023` in the open-instruct
workspace. The follow-up trace analysis is in `runs/issue31-profile2-20261024`.
Immutable Beaker payload: `01M5QCEZD3QJYRPP4BAP8KY8JC`;
result dataset: `01M5QCKBM1JF3NYMQC1SCPJKPD`.


### Latent/shared-expert projection packing (investigation)

`OLMO_SGLANG_MOE_PROJECTIONS=packed` opts into a shared GEMM for the latent
expert input and the shared expert's gate/up projection. The default is `torch`.
The packed path requires BF16 CUDA decode, TP1/EP1, unquantized weights, both
latent and shared branches. It supports the automatically selected rounding mode
and the ordinary optimized path. Prefill, the full eager Core-reference mode,
and other unsupported configurations retain the original calls. Restart the
engine when changing the option.

The original named Parameter objects and checkpoint loaders are retained as
contiguous row views into one persistent buffer. Device/dtype moves rebuild those
views before graph capture. The packed GEMM uses the existing unquantized dispatch
setting; changing shapes can change the selected kernel and accumulation order.

A direct split of the packed output would leave strided tensors: SGLang's fused
MoE requires contiguous expert input, and the shared SwiGLU wrapper would make
another copy. This option instead uses one kernel to copy the latent output into
contiguous storage while producing the shared SwiGLU activation in contiguous
storage. It preserves the separate BF16 SiLU rounding before multiplication.
There is no per-step weight concatenation. On this checkpoint the intended
reduction is one GEMM call in each of 15 sparse layers, with one activation
kernel retained. Actual kernel counts, throughput, and common-token trainer
divergence still need whole-engine measurement; no gain or agreement claim is
made yet. See the [investigation record](moe-projection-investigation.md).

Portable storage, partial-load, output-layout, and default-path tests pass.
CUDA checks pass for strided/empty outputs, all finite BF16 activation
inputs, actual SGLang checkpoint loading, changed inputs, and graph replay after
weight updates. A constructor check also verifies that the option remains active
after the default rounding-mode context exits. The frozen full-engine panel is
running; its performance and numerical conclusions are pending.
