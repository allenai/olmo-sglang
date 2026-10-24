# Inference options and measured tradeoffs

This is a menu of inference building blocks for Olmo 3.5 KDA MoE. Measure
throughput and trainer agreement together; numerical divergence is a cost to
document and evaluate case by case. The individual investigations are complete. The [second-pass composition](decode-composition.md)
is now being qualified on H100 and B300. Real-workload RL trials and
learning-quality evaluation remain a later phase.

The [individual measurement table](decode-option-results.md) preserves the
independent throughput and trainer-agreement results. They are not additive.

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

## Original candidate list

This list records the hypotheses before the second pass. Implemented options
and measurements appear below and in the [measurement table](decode-option-results.md).
The remaining expert-kernel ideas have not been implemented or measured. Component
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

### Investigation: shared-expert add into post-MLP norm

`OLMO_SGLANG_SHARED_ADD_NORM=fused` opts into combining the routed/shared
expert addition with the post-feedforward rounding norm. The default is `torch`.
Set it before engine startup; changing it requires an engine restart. The option
is active only for sparse layers with a shared expert and fused rounding norms;
other configurations retain the separate addition. It preserves the BF16 sum
before normalization and the BF16 norm output before the residual addition.
Weight names, loaders, and publication storage are unchanged.

The [H100 investigation](https://beaker.org/ex/01M5QFWFK04XSK73YRNPBN48X8)
compared current defaults (`70abc4ad`) with candidate `66a831d`, using the same
C64 fixed/mixed requests, BF16 TP1/EP1, memory fraction 0.7, prefill chunk 8192,
context 12288, full decode graphs, radix/Mamba cache, and route capture/return.
Each panel had three timed repeats, bracketed by controls on the same GPU.

| Whole-wave throughput | Bracketed control | Fused option | Change | Closing-control drift |
|---|---:|---:|---:|---:|
| Fixed length | 5,789.58 tokens/s | 5,838.37 tokens/s | +0.84% | +0.91% |
| Mixed lengths | 3,624.69 tokens/s | 3,630.81 tokens/s | +0.17% | +0.066% |

All six paired waves improved, but the fixed panel's control drift limits the
precision of its larger aggregate gain. Three-wave descriptive bootstrap ranges
were +0.22 to +2.02% fixed and +0.11 to +0.26% mixed; these are short screening
estimates, not production forecasts. The profile gives narrower evidence of a
small benefit: **625 → 610 graph kernels/step**, **8.172 → 8.163 ms** of summed
graph GPU work, and **8.774 → 8.763 ms** over the last 32 profiled steps. The
additional norm work offsets some of the removed additions. Profiling ran after
the timed waves and was excluded from throughput.

All **384 trajectories / 1,035,090 output tokens and logprobs** matched exactly
across the candidate and both controls. Thus any fixed trainer-reference error,
bias, tail, and ratio-outlier statistic is unchanged on these trajectories; this
does not mean absolute trainer/inference disagreement is zero. The [shared trainer scoring pass](https://beaker.org/ex/01M5QH3K0YFW17Z7J7W709V53G)
completed on the same frozen trajectories (128 prompt clusters). It uses pinned
Core `e505356`, BF16 evaluation, natural routing, torch attention, and one complete
request per document without EP. No real-RL evaluation was run.

| Trainer disagreement on the same tokens | Control and fused option | Added cost |
|---|---:|---:|
| Mean absolute error | 0.03694615 nats/token | 0 |
| Signed serving-minus-Core bias | +0.00201359 nats/token | 0 |
| p95 absolute error | 0.13185763 nats | 0 |
| p99 absolute error | 0.23219654 nats | 0 |
| Maximum absolute error | 3.29934740 nats | 0 |
| Probability ratios outside [0.8, 1.2] | 1.64720% | 0 percentage points |

All paired prompt-bootstrap changes are exactly zero because the serving vectors
are identical. This is an observed result on this panel, not a guarantee for
unmeasured hardware or workloads. The scorer retained full Core logprob vectors;
its result dataset is `01M5QH3K19XS37AFA3DTAJGW7F` and process runtime was 94 seconds.

Validation: 155 portable tests passed, 117 CUDA-dependent cases skipped locally;
45 GPU norm cases passed against separate tensor operations, including changed-
input CUDA graph replay. Model source and configuration hashes and the requested
option values were verified from the engine reports. Process runtime was
16m12s; engine startup was 54.31/31.38 seconds for opening control/candidate
and is excluded from throughput. Both earlier B300 attempts were canceled while
queued, before GPU execution. **B300 performance remains unmeasured.**

This is a small, default-off building block. It has not been selected for a
production combination. Raw finalized records, profiles, source bundle and
attempt ledger: `open-instruct/runs/issue31-decode-menu-20261024/01-shared-add`.
Immutable H100 payload: `01M5QFVDHASTZM94Q5PSN36WH4`;
result dataset: `01M5QFWFKCYYP3CKAK1D0TG4X5`.

### Investigation: routing-tail fusion — measured H100 option

`OLMO_SGLANG_ROUTING_TAIL=fused` opts into a combined selected-weight L1
normalization, scaling, and int32 index conversion. Default is `torch`; restart
the engine when changing captured graphs. Softmax and top-k remain PyTorch.
CUDA top-16 with L1 or no normalization is supported; CPU, other top-k sizes and
other norms retain the tensor path. The full `OLMO_SGLANG_ROUTING_KERNELS=fused`
option (#10) takes precedence and bypasses this tail. The pre-MLP FP32 copy is
unchanged and is a separate investigation.

The completed H100 C64 control/candidate/control panel measured:

| Frozen workload | Bracketed control tokens/s | Candidate tokens/s | Change | Closing-control drift |
|---|---:|---:|---:|---:|
| Fixed length | 5,798.14 | 5,801.57 | +0.059% | +0.168% |
| Mixed lengths | 3,627.09 | 3,662.80 | +0.985% | +0.241% |

Mixed paired gains were +1.073%, +0.957%, and +0.926%. The three-wave bootstrap
range is +0.926% to +1.073%; this short panel does not establish variability
across machines or days. Fixed paired gains were -2.889%, +1.332%, and +1.607%,
so the fixed aggregate is inconclusive. Do not report a uniform 1% speedup.

The separate 64-step trace confirms **625 → 580 graph kernels per step** and
**8.176 → 8.108 ms** summed graph GPU work (about 68.5 microseconds removed).
That sum includes the profile's ramp and is not end-to-end latency. Unprofiled
whole-wave throughput includes prefill and drain; startup/warmup/profiling are
excluded. All 384 trajectories / **1,035,090 token IDs and logprobs** exactly
match both controls. A completed common OLMo-core scoring pass verifies:

| Trainer comparison, same trajectories | Control | Candidate | Added cost |
|---|---:|---:|---:|
| Mean absolute logprob error, nats/token | 0.03694615 | 0.03694615 | 0 |
| Signed serving minus trainer mean | +0.00201359 | +0.00201359 | 0 |
| p95 absolute error | 0.13185763 | 0.13185763 | 0 |
| p99 absolute error | 0.23219654 | 0.23219654 | 0 |
| Maximum absolute error | 3.29934740 | 3.29934740 | 0 |
| Probability ratio outside [0.8, 1.2] | 1.64720% | 1.64720% | 0 percentage points |

Unchanged error is not zero absolute trainer error. The reference uses Core
`e505356`, BF16 eval, natural routing, torch attention and one full request per
document. It is not a real-RL route-replay or learning-quality result. Reference
trajectory/input/model hashes were checked before reuse. B300 and combinations
with other new options remain unmeasured.

Validation: pinned Ruff checks pass; portable suite 157 passed, 120 CUDA cases
skipped locally. The pinned GPU precheck passed 123 exact cases, covering row
counts 1–8192, normalization/scaling variants, extreme/zero values and changed
inputs under graph replay. Runtime Python/JSON hashes and option dispatch were
verified. Baseline source `70abc4ad`; tested candidate `d53c8430` (later changes
are result documentation only). BF16 TP1/EP1, memory fraction 0.7, context 12288,
prefill chunk 8192, full decode graphs, radix/Mamba cache, capture and return of
routes, original logprobs, PyTorch sampling, three fixed and three mixed waves.

Runs:

1. Completed protected H100 panel and profiles, process runtime 16m35s:
   [Beaker](https://beaker.org/ex/01M5QJZ8APPEQXPD5BDZQ1S47V).
   Payload `01M5QHJWDFE3A4TSVMEQJ6SMPT`; result `01M5QJZ8AYFGCT9WH4QXDBX04S`.
2. Earlier unallocated attempt was preempted after opening control and candidate,
   during closing control: [Beaker](https://beaker.org/ex/01M5QHTQND4K7XN8P6AZTPGD26).
   Its numerics/profile agree with the repeat, but its incomplete bracket is not
   the reported throughput result. Result `01M5QHTQNTVJABYGVFC9XV120H`.
3. Common trainer reference, 384 trajectories / 1,035,090 tokens, 94 seconds:
   [Beaker](https://beaker.org/ex/01M5QH3K0YFW17Z7J7W709V53G).
   Result `01M5QH3K19XS37AFA3DTAJGW7F`.

Raw profiles, vectors, audit, scorer comparison and attempt ledger are under
`runs/issue31-decode-menu-20261024/02-routing-tail` in the open-instruct workspace.
This is a default-off building block for review; no production default or image
was changed.

### Investigation: FP32 router input from the norm — measured H100 option

`OLMO_SGLANG_ROUTER_INPUT=fused` asks the sparse layer's pre-MLP norm to emit
both its BF16 result and an exact FP32 widening of that result. The router uses
the FP32 output; latent/shared/expert inputs stay BF16. The FP32 store occurs
**after BF16 rounding**, not from the norm's unrounded intermediate. Default is
`torch`; restart the engine when changing the option. Layers without a compatible
rounding pre-MLP norm retain their existing path. The norm API's non-BF16/CPU
fallback computes the two outputs with Torch. This does not replace routing
softmax, top-k or normalization; routing-tail fusion is a separate option.

The completed H100 C64 control/candidate/control panel measured:

| Frozen workload | Bracketed control tokens/s | Candidate tokens/s | Change | Closing-control drift |
|---|---:|---:|---:|---:|
| Fixed length | 5,784.50 | 5,834.27 | +0.860% | -0.538% |
| Mixed lengths | 3,621.76 | 3,644.32 | +0.623% | -0.0065% |

All six paired waves improved. Fixed gains ranged from +0.430% to +1.308%;
mixed gains ranged from +0.561% to +0.668%. Those are also the three-wave
bootstrap endpoints. The fixed control drift is material; this short panel is
not a cross-machine/day variability estimate. Startup, warmup and profiling
are excluded from throughput; whole-wave timing includes prefill and drain.

The separate 64-step trace confirms **625 → 610 graph kernels per step** and
**8.175 → 8.135 ms** summed graph GPU work, about 40.6 microseconds saved. These
profile sums include the ramp and are not unprofiled end-to-end latency.
All 384 trajectories / **1,035,090 token IDs and logprobs** exactly match both
controls. The completed common OLMo-core reference verifies:

| Trainer comparison, same trajectories | Control | Candidate | Added cost |
|---|---:|---:|---:|
| Mean absolute logprob error, nats/token | 0.03694615 | 0.03694615 | 0 |
| Signed serving minus trainer mean | +0.00201359 | +0.00201359 | 0 |
| p95 absolute error | 0.13185763 | 0.13185763 | 0 |
| p99 absolute error | 0.23219654 | 0.23219654 | 0 |
| Maximum absolute error | 3.29934740 | 3.29934740 | 0 |
| Probability ratio outside [0.8, 1.2] | 1.64720% | 1.64720% | 0 percentage points |

This is unchanged divergence, not zero absolute trainer error. Core `e505356`
used BF16 eval, natural routing, torch attention and one full request per document.
Reference input/model/trajectory hashes were validated before reuse. B300,
compositions with the other options, route-replay RL scoring and learning-quality
outcomes remain unmeasured.

Validation: pinned Ruff passes; portable suite 155 passed and 117 CUDA cases
skipped locally. The GPU precheck passed 45 exact BF16/FP32-output cases across
five widths and three row counts, with scaling/residual variants and changed
inputs, residuals and norm weights under graph replay. Runtime source/configuration
hashes and mode settings were verified. Tested candidate `f8f77a99`, baseline
`70abc4ad`; later changes are result documentation only. BF16 TP1/EP1, memory
fraction 0.7, context 12288, prefill chunk 8192, full decode graphs, radix/Mamba
cache, route capture/return, original logprobs and PyTorch sampling. Three fixed
and three mixed waves.

Runs:

1. Completed H100 panel, kernel checks and profiles, process runtime 16m17s:
   [Beaker](https://beaker.org/ex/01M5QM4MAT3PX0F8H75A6HYD8Y).
   Payload `01M5QKB2AQ8VMJW47FRTMWK2A4`; result `01M5QM4MB1AXWY9XVK53TC00JV`.
2. Common trainer reference, 384 trajectories / 1,035,090 tokens, 94 seconds:
   [Beaker](https://beaker.org/ex/01M5QH3K0YFW17Z7J7W709V53G).
   Result `01M5QH3K19XS37AFA3DTAJGW7F`.

Raw profiles, vectors, audit and trainer comparison are under
`runs/issue31-decode-menu-20261024/02b-router-input` in the open-instruct workspace.
This is a default-off building block for review; no production default or image
was changed.

### Investigation: fused simple sampler — measured H100 option

`OLMO_SGLANG_SAMPLER=fused` is a default-off option for pinned SGLang
`0.5.19.dev49+g3145136`, TP1 and the PyTorch sampling backend. It retains Torch's
exponential RNG and existing softmax/log-softmax, then combines probability/noise
division, argmax and validity checks. The selected token uses the first-index tie
and NaN rules checked against Torch. It uses two selection kernels plus the
existing RNG and an asynchronous validity assertion. Set before engine startup;
restart to change the process-local sampler hook. A different SGLang version
raises an error when this supported configuration is requested. Seeded/filtered
sampling, unsupported tensor layouts/dtypes, TP and other backends retain the
original path. Greedy sampling bypasses this selection hook.

H100 C64 control/candidate/control results:

| Frozen workload | Bracketed control tokens/s | Candidate tokens/s | Change | Closing-control drift |
|---|---:|---:|---:|---:|
| Fixed length | 5,781.67 | 5,762.08 | -0.339% | +0.177% |
| Mixed lengths | 3,619.28 | 3,645.82 | +0.733% | -0.139% |

Mixed paired gains were +0.641%, +0.932%, +0.653%. Fixed paired changes were
-1.040%, +0.235%, -0.326%; the fixed result is inconclusive and includes a small
aggregate regression. Three-wave bootstrap ranges are [-1.040%, +0.235%] fixed
and [+0.641%, +0.932%] mixed; they do not estimate cross-day/machine variation.
This option is not a uniform throughput improvement.

The isolated complete sampler call at vocabulary 100278 measured about
134.11 → 63.35 microseconds at batch 64 (78.73 → 54.57 at 16, 74.56 → 56.70 at1).
Those microbenchmarks are not whole-engine gains. Separate 64-step profiles show
outside-graph GPU kernel work **0.2660 → 0.2123 ms/step**. Both custom selection
kernels execute once per step outside the graph; the graph retains 625 kernels
and essentially unchanged summed work (8.179 → 8.180 ms). Profiled elapsed spans
moved the other direction, so use unprofiled waves for the throughput conclusion.
No sampler graph-capture speedup is claimed.

All 384 trajectories / **1,035,090 token IDs and logprobs** exactly match both
controls. The completed common Core reference, reused after model/input/trajectory
hash validation, gives:

| Trainer comparison, same trajectories | Control | Candidate | Added cost |
|---|---:|---:|---:|
| Mean absolute logprob error, nats/token | 0.03694615 | 0.03694615 | 0 |
| Signed serving minus trainer mean | +0.00201359 | +0.00201359 | 0 |
| p95 absolute error | 0.13185763 | 0.13185763 | 0 |
| p99 absolute error | 0.23219654 | 0.23219654 | 0 |
| Maximum absolute error | 3.29934740 | 3.29934740 | 0 |
| Probability ratio outside [0.8, 1.2] | 1.64720% | 1.64720% | 0 percentage points |

Unchanged divergence is not zero absolute trainer error. Core `e505356` used
BF16 eval, natural routing, torch attention and one full request per document.
B300, compositions and real-RL route-replay/learning-quality effects remain
unmeasured. Forced-token scoring alone would not prove sampler correctness;
this panel used ordinary free sampling, and separate RNG checks were performed.

Validation: portable suite 156 passed/115 CUDA cases skipped locally; pinned Ruff
passes. GPU precheck passed 240 repeated draws with exactly matching samples and
final CUDA RNG states across batches 1/16/64 and vocabularies 2/33/512/100278.
Crafted cross-partition ties, infinities/NaNs, invalid flags and changed-input graph
replay of the selection kernel also passed. Tested Torch 2.13.0+cu130/Triton 3.7.1.
Runtime source/configuration hashes, requested mode and actual custom-kernel
dispatch were verified. Candidate `b2bcc96`, baseline `70abc4ad`; later changes
are result documentation only. BF16 TP1/EP1, memory 0.7, context 12288, prefill 8192,
full decode graphs, radix/Mamba cache, route capture/return, original logprobs,
temperature 1 / top_p 1 / top_k -1. Three fixed and three mixed waves. Startup, warmup
and profiling excluded from throughput; prefill and drain included.

Runs:

1. Completed H100 panel, RNG checks and profiles, process runtime 16m24s:
   [Beaker](https://beaker.org/ex/01M5QN43B7H4K343V77ACTX4EE).
   Payload `01M5QKSRY6KR12QZ173ZD0KV6Z`; result `01M5QN43BENREX7PJKAHGHE9KV`.
2. Common trainer reference, 384 trajectories / 1,035,090 tokens, 94 seconds:
   [Beaker](https://beaker.org/ex/01M5QH3K0YFW17Z7J7W709V53G).
   Result `01M5QH3K19XS37AFA3DTAJGW7F`.

Raw profiles, vectors, source manifest, audit, sampler microtimings and trainer
comparison are under `runs/issue31-decode-menu-20261024/03-sampler` in the
open-instruct workspace. This is a default-off building block for review; no
production image or default was changed.

### KDA first-stage projection packing (investigation)

`OLMO_SGLANG_KDA_PROJECTIONS=packed` opts into one GEMM for the first-stage
QKV, forget-gate, beta, and output-gate projections during BF16 CUDA decode
with TP1 and unquantized weights. The default is `torch`. Prefill and unsupported
configurations use the original projection calls. Restart the engine when
changing the option.

The packed call retains SGLang's unquantized dispatch setting; shape-dependent
heuristics may select a different kernel or backend after packing.
The existing named parameters become views into one persistent packed buffer;
checkpoint loaders and in-place online copies continue updating those parameters.
Device/dtype moves rebuild the views before graph capture. The second-stage gate
projections and their BF16 inputs remain separate, including the output gate's
bias. The recurrence accepts beta's padded row stride directly; a non-unit head
stride still takes the contiguous fallback. No per-step weight concatenation is
performed. Changing GEMM shapes may
change rounding, so this option needs measured common-token trainer divergence.
This first-stage change replaces four GEMM calls with one in each of the
checkpoint's 14 KDA layers: 42 calls in principle, before checking extra copies
and the actual trace. Second-stage gate packing and latent/shared-expert
projection packing are separate candidates and are not included here.
The frozen H100 C64 panel improves fixed-length throughput by **1.786%** and
mixed-length throughput by **2.207%**. All 1,035,090 output token IDs and logprobs
match both controls exactly; added trainer error and ratio-outlier cost are zero
on these trajectories. The trace has 583 rather than 625 graph kernels per step,
matching the intended 42-call reduction. These results qualify this checkpoint,
runtime and H100 workload; B300, other shapes and composed options remain
unmeasured. See [the investigation](kda-projection-investigation.md) for the
protocol, per-wave results, loader/graph checks and limitations.


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
There is no per-step weight concatenation. The H100 trace confirms one fewer
GEMM call in each of 15 sparse layers, with one activation kernel retained:
625 → 610 graph kernels per step. The independent frozen panel measures
+0.759% fixed-length and +0.766% mixed-length throughput. All 1,035,090 output
tokens/logprobs match both controls exactly, so added Core-error metrics are
zero on these trajectories. This does not establish equality on other models
or hardware. See the [investigation record](moe-projection-investigation.md)
for controls, uncertainty, limitations and raw artifacts.

Portable storage, partial-load, output-layout, and default-path tests pass.
CUDA checks pass for strided/empty outputs, all finite BF16 activation
inputs, actual SGLang checkpoint loading, changed inputs, and graph replay after
weight updates. A constructor check also verifies that the option remains active
after the default rounding-mode context exits. Full-engine source, checkpoint,
trajectory and finalized-profile hashes were verified.

### Grouped second-stage KDA gate projections (investigation)

`OLMO_SGLANG_KDA_GATE_PROJECTIONS=grouped` selects a single Triton launch
containing the two second-stage gate GEMMs during TP1, unquantized BF16 CUDA
decode. The default is `torch`. Prefill and unsupported configurations retain
the original calls. Restart the engine when changing the option.

The forget and output gates read distinct first-stage inputs. This implementation
groups their tiles without introducing zero-filled blocks or doubling arithmetic.
It preserves both BF16 intermediate tensors and the output-gate bias, and writes
two contiguous BF16 results. It reads the original parameter tensors directly;
there is no packed copy or new checkpoint mapping. Different tensor-core reduction
and bias handling may change rounding, so agreement must be measured.

This is evaluated independently of first-stage QKV/f1/beta/g1 packing and
latent/shared-expert packing. It targets one fewer GEMM launch in each of the
14 KDA layers; the full-engine trace confirms 625 → 611 graph kernels per step.
GPU checks pass. The independent H100 panel measures +0.400% fixed-length and
+0.669% mixed-length throughput, with all 1,035,090 output tokens/logprobs exactly
equal against both controls and zero added Core-error metrics. The fixed gain is
small and varies across waves; see the
[investigation record](kda-grouped-gates-investigation.md) for uncertainty,
qualification scope and raw evidence. Other models/hardware are not qualified.
First-stage and second-stage packing touch the same KDA forward path, so their
composition requires integration and another measurement.
