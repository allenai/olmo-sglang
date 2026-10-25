# Second-pass decode composition

H100 qualification is complete, including the finalized-dataset audit and all
same-token numerical comparisons. B300 qualification is in progress. This
branch integrates the eight individually measured performance
options and the q/k rounding correction; it does not yet establish a production
recommendation or learning-quality results. The
[individual option measurements](decode-option-results.md) remain the evidence
for the components.

## Configurations under test

| Arm | Runtime source | Settings |
|---|---|---|
| Current main | `70abc4a` | First-pass defaults, original FP32 normalized q/k, tensor routing, pickle |
| Performance composition | `5dc7f4f` | All eight performance options below, original q/k arithmetic |
| Aligned composition | `2c12aa6` | Performance composition plus normalized q/k rounded to BF16 before recurrence |
| Optional fused routing | Same aligned source | Above plus `OLMO_SGLANG_ROUTING_KERNELS=fused` |

The aligned source includes the arithmetic change from PR #21 unconditionally.
The original-q/k source is a qualification control, not a proposed permanent
configuration axis. The intended final configuration adopts q/k alignment after
the composition's exact-output check and its separate numerical measurement.
Fused routing remains an explicit opt-in throughput tradeoff.

## H100 timing result

These are the audited timing arms from the finalized H100 result dataset below.
Throughput uses the mean opening/closing control time for each paired wave.

| Configuration | Fixed tokens/s | Change | Mixed tokens/s | Change |
|---|---:|---:|---:|---:|
| Current main, bracketed control | 5,724 | — | 3,605 | — |
| Performance composition, original q/k | 6,127 | +7.03% | 3,859 | +7.07% |
| Composition with q/k alignment | 6,126 | +7.02% | 3,854 | +6.91% |
| Above plus optional fused routing | 6,228 | +8.80% | 3,915 | +8.59% |

Opening-to-closing control drift was +0.87% fixed and +0.15% mixed. The
three-wave paired bootstrap intervals for the aligned configuration's gains
are [5.84%, 8.66%] fixed and [6.13%, 7.68%] mixed. These describe this panel;
they do not establish machine/day variability. The 3–5% planning estimate was
not a bound on the composed result.

The optional fused router adds **1.66% fixed / 1.57% mixed** over the aligned
composition, with paired-wave intervals [1.14%, 2.10%] and [1.52%, 1.64%]. This
direct treatment comparison is not separately bracketed. Q/k alignment changed
timing by −0.007% fixed and −0.14% mixed versus the original-q/k composition;
neither interval excludes zero.

The original-q/k performance composition matches both main controls exactly
across all **384 requests / 1,035,090 output tokens**, including logprobs and
the hashes of **1,155,807,232 bytes** of route data. Aligned and fused-router
generation trajectories change; their numerical tradeoffs come from the separate
same-token replay below.

The H100 traces confirm **625 → 479** graph kernels for the performance
composition and **464** with the optional fused router. The latter replaces four
launches per sparse layer with three: a zero-bias fill, the fused router, and a
scale multiply. Thus it saves only 15 additional launches after the routing-tail
and router-input fusions, explaining why its standalone gain cannot be added.
Profiled graph-kernel sums were 8.178 ms for main, 7.804 ms for performance,
7.827 ms with q/k alignment, and 7.665 ms with optional fused routing. These are
profile attribution measurements; the throughput table uses unprofiled waves.

## H100 trainer agreement

All rows score the same 384 requests / **1,035,090 tokens / 128 prompt clusters**
against the fixed Core reference. Signed error is serving minus Core, in nats;
outliers are `exp(Core - serving)` outside [0.8, 1.2]. The
[machine-readable audit](data/decode-composition-h100.json) includes source,
settings and artifact identities, paired intervals, tail metrics and profiles.

| Configuration | Mean absolute error | Signed mean | Ratio outliers |
|---|---:|---:|---:|
| Main / original-q/k performance composition | 0.036951 | +0.001789 | 1.6640% |
| Composition with q/k alignment | 0.036966 | +0.000095 | 1.7020% |
| Above plus optional fused routing | 0.036988 | +0.000118 | 1.7308% |

The original-q/k composition also matches main exactly in the fresh forced
replay: every logprob and every trainer-error metric is unchanged. Q/k alignment
reproduces the earlier q/k-only candidate's full logprob vector exactly. Thus
the eight performance options add no observed numerical cost in either of these
two H100 arithmetic configurations on this panel.

| Addition | MAE change, nats [95% interval] | Outlier change, percentage points [95% interval] |
|---|---:|---:|
| Q/k alignment | +0.000015 [−0.000088, +0.000120] | +0.03797 [+0.00723, +0.06890] |
| Fused routing on aligned composition | +0.000021 [−0.000070, +0.000116] | +0.02879 [−0.00143, +0.06088] |
| Aligned + fused routing versus main | +0.000036 [−0.000074, +0.000148] | +0.06676 [+0.03681, +0.09668] |

Q/k alignment reduces the sampled-token signed mismatch by 94.7%, with
negligible timing effect. Its outlier increase is small but resolved by this
paired interval: about four extra outliers per 10,000 tokens. It remains the
intended arithmetic default. Fused routing offers another 1.6% throughput, with
a point estimate of about three extra outliers per 10,000; that incremental
interval includes zero. It remains opt-in. Do not equate either observation
with a demonstrated training-quality effect or dismiss all small outlier shifts
as noise.

The maximum absolute trainer error was 3.299 nats for main, 1.252 for aligned,
and 1.627 with fused routing; these are descriptive maxima. No aggregate MAE
change interval excludes zero. Similar error magnitudes across implementations
do not prove an irreducible numerical floor or exclude a shared hidden cause.

## Select the candidate

The performance configuration sets these variables before engine startup and
graph capture. `source configs/decode-optimized.env` selects the complete
candidate below; its qualification status is the status of this document.
After sourcing, set `OLMO_SGLANG_ROUTING_KERNELS=fused` only to select the
additional fused-routing tradeoff. BF16 q/k alignment is built into this
revision and has no separate preset switch.

```bash
export OLMO_SGLANG_SHARED_ADD_NORM=fused
export OLMO_SGLANG_ROUTING_TAIL=fused
export OLMO_SGLANG_ROUTER_INPUT=fused
export OLMO_SGLANG_SAMPLER=fused
export OLMO_SGLANG_KDA_PROJECTIONS=packed
export OLMO_SGLANG_MOE_PROJECTIONS=packed
export OLMO_SGLANG_KDA_GATE_PROJECTIONS=grouped
export SGLANG_USE_PICKLE_IPC=false
export OLMO_SGLANG_ROUTING_KERNELS=torch
```

Restart to change captured behavior. The individual optimization switches still
default to their existing tensor paths in this investigation branch; the block
above selects the complete candidate. Their documented guards and fallbacks
remain active. The fused-router switch supersedes the routing-tail path; their
individual savings cannot be added.

## Integration contracts

- The pre-MLP norm can emit the FP32 widening of its rounded BF16 output while
  the post-MLP norm separately accepts the rounded shared+routed expert sum.
  Residual and scale rounding boundaries are preserved.
- Latent/shared packing supplies the shared expert's activation to its down
  projection and can return that result separately for the post-MLP norm fusion.
  Dense MLP calls retain their original interface.
- Packed KDA first-stage projections feed the two grouped second-stage gate
  GEMMs directly. The first-stage slices may have a larger row stride, and each
  second-stage input retains its BF16 boundary. The two switches can operate
  independently or together.
- Original parameter objects, checkpoint names and loader mappings remain intact.
  Packed storage is nonpersistent; in-place weight updates refresh graph-visible
  weights. Explicit integration tests cover both projection stages under one
  captured graph and combined norm boundaries with changed inputs and weights.
- Route capture and return remain enabled. MessagePack changes IPC serialization;
  all timed routes are length-checked and hashed in order alongside retained
  output token IDs and full logprob vectors.

## Measurement contract

Each hardware panel uses one GPU, BF16 TP1/EP1, admission C64, context 12,288,
memory fraction 0.7, prefill chunk 8,192, full decode graphs, radix/Mamba caching,
original logprobs, and route capture/return. It preserves the step5402 checkpoint
and pinned SGLang `3145136d` / Torch 2.13.0+cu130 / Triton 3.7.1 runtime.
H100 retains its default expert tiles; B300 selects the shipped tuned tiles.

The five timing arms are opening main, performance, aligned, fused routing,
and closing main. Each has three fixed 1,024-token waves and three frozen mixed
RL-length waves up to 10,240 tokens. Whole-wave wall time includes admission,
drain and result handling; startup, warmup, profiling and scoring are separate.
Report control drift, paired-wave changes and descriptive three-wave intervals.
Direct comparisons between adjacent treatment arms are not independently
bracketed and must be read alongside the repeated-main drift.

All four arms also force-score the same 384 requests / 1,035,090 frozen H100
output tokens, with identical fresh-engine cache history on each GPU type.
That preserves reuse of the pinned Core `e505356` reference: BF16 evaluation,
natural routing, torch attention, one complete request/document, no EP.
Those Core scores were computed on H100; both serving-hardware panels compare
against that same fixed trainer reference.
Forced selection synchronizes with the CPU; replay timing is diagnostic only.

Numerical reports include MAE, signed serving-minus-Core error, p95/p99/maximum
absolute error and ratios `exp(Core - serving)` outside [0.8, 1.2]. Paired
prompt-cluster bootstrap intervals retain repeated trajectories from each
prompt. Report both total changes against main and incremental changes for
q/k alignment and fused routing. Tail maxima are descriptive, not a standalone
adoption criterion. Cross-hardware equality and real-RL learning effects are
not assumed.

## Runs and artifacts

1. H100 panel, one Jupiter GPU, high priority, one hour protected runtime,
   80-minute timeout, no automatic restart:
   [Beaker](https://beaker.org/ex/01M5R5BKE1MX67JVVBKV9SMXK8).
2. Earlier H100 attempt stopped in a harness-only transport precheck before model
   loading or timing; the replacement retains identical runtime source hashes:
   [Beaker](https://beaker.org/ex/01M5R4ZSPDDXR4GNRJJDKRQVQD).
3. B300 panel, one Holmes GPU, high priority, 45 minutes protected runtime,
   65-minute timeout, no automatic restart:
   [Beaker](https://beaker.org/ex/01M5R8G8DD0JVH434ZQTHDGNW7).

Payload: `01M5R597W29Z0M528YKNQ0VYEB`.
H100 result: `01M5R5BKEFMG1YZEJKV0HJ6E5Y` (finalized and audited).
B300 result: `01M5R8G8DNG01BC42EP1NTGRJ3` (active).
Common Core reference: `01M5QH3K19XS37AFA3DTAJGW7F`.
The payload retains complete source snapshots, hashes, GPU prechecks, the frozen
workload and replay inputs. `tools/run_decode_composition.py` runs from that
payload; its auxiliary benchmark/replay scripts are included in the dataset.
Local preparation, analyses and the attempt ledger live under
`open-instruct/runs/issue31-composition-20261024`.

The successful H100 job queued for 8.35 seconds, initialized for 25.26 seconds,
and executed for **52m44s**. That execution includes 1m54s of GPU prechecks,
26m07s for five timing-engine processes (including their separate startup,
warmup and profiles), and 24m39s for four diagnostic replay processes. Per-arm
startup/warmup and measured wave times are retained in the audit. The earlier
harness-only attempt executed for 47.41 seconds after 3m08s of initialization;
it produced no throughput or divergence measurement.

### Reproduce the analysis

After the result dataset is finalized, download it alongside the frozen payload
and Core scores. From the repository root, in an environment with NumPy:

```bash
beaker dataset fetch 01M5R597W29Z0M528YKNQ0VYEB --output artifacts/composition/payload
beaker dataset fetch 01M5R5BKEFMG1YZEJKV0HJ6E5Y --output artifacts/composition/h100
beaker dataset fetch 01M5QH3K19XS37AFA3DTAJGW7F --output artifacts/composition/core-reference
python -m tools.decode_composition.analyze \
  --root artifacts/composition/h100 \
  --payload artifacts/composition/payload \
  --core artifacts/composition/core-reference/core \
  --output artifacts/composition/h100-analysis
```

The analyzer rejects incomplete runs, changed source/configuration hashes,
changed workloads, codec dispatch mismatches and mismatched scoring trajectories.
It emits a compact audit and full per-prompt numerical comparisons, including
incremental aligned-to-fused-router intervals, exact forced-replay comparisons,
and summaries with SHA256 hashes for all four decode traces. Each trace must
contain the same 64 C64 decode steps. The paired-wave intervals use
only three observations per workload and do not measure machine/day variance.
The numerical intervals resample prompts, keeping repeated trajectories and all
their tokens together; a million tokens are not a million independent samples.

For a downloaded decode trace, run
`python -m tools.decode_composition.analyze_trace TRACE --output PROFILE_SUMMARY`.
Kernel sums, GPU activity union and elapsed span are reported separately.
Use the unprofiled timed waves for throughput; profiled durations are attribution
evidence, and a gap alone does not establish a CPU bottleneck.

Local validation: Ruff/format/diff checks passed; 174 portable tests passed,
173 CUDA/runtime-dependent cases skipped. All ten H100 GPU prechecks passed,
including the combined norm and KDA projection graph tests. The complete H100
audit passed, including both free-generation controls, forced vectors, all
four traces and Core provenance. B300 qualification remains pending.
