# Second-pass decode composition

Qualification is in progress on H100 and B300. This branch integrates the eight
individually measured performance options and the q/k rounding correction. It
does not yet establish a composed speedup, a production recommendation, or
learning-quality results. The [individual option measurements](decode-option-results.md)
remain the evidence for the components.

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

The performance configuration sets these variables before engine startup and
graph capture:

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
3. B300 panel: not launched yet; runs after the H100 investigation.

Payload: `01M5R597W29Z0M528YKNQ0VYEB`.
H100 result: `01M5R5BKEFMG1YZEJKV0HJ6E5Y` (not finalized).
Common Core reference: `01M5QH3K19XS37AFA3DTAJGW7F`.
The payload retains complete source snapshots, hashes, GPU prechecks, the frozen
workload and replay inputs. `tools/run_decode_composition.py` runs from that
payload; its auxiliary benchmark/replay scripts are included in the dataset.
Local preparation, analyses and the attempt ledger live under
`open-instruct/runs/issue31-composition-20261024`.

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
including the combined norm and KDA projection graph tests. Whole-engine
throughput/agreement results and B300 qualification are pending. A 3–5% throughput gain over current
main is a planning assumption only; no composed gain is claimed yet.
