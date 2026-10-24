# Latent/shared-expert projection investigation

The independent H100 panel measures +0.759% fixed-length throughput and +0.766%
mixed-length throughput. All 384 requests / 1,035,090 output tokens and logprobs
match both controls exactly, so every measured incremental Core-error metric is
zero on these trajectories. The option remains default-off; this is not a
composition, B300, or real-RL quality result.

## Option and implementation

Set `OLMO_SGLANG_MOE_PROJECTIONS=packed` before starting the engine. The default
is `torch`; restart the engine to change it. The path requires unquantized BF16
CUDA decode, TP1/EP1, a latent MoE projection and a shared expert. It supports
the model's default automatic rounding mode. Prefill, full eager Core-reference
mode and unsupported layouts retain their original calls.

The latent down projection and shared gate/up projection read the same hidden
state. Their original named Parameter objects become contiguous row views of
one persistent packed buffer. Original checkpoint names and loader functions
remain attached, including repeated partial updates. Device/dtype moves rebuild
the views before graph capture; weights are not concatenated per step.

One GEMM produces both branches. A combined output kernel materializes
contiguous latent expert input and shared SwiGLU output, preserving the separate
BF16 SiLU rounding boundary before multiplication. This avoids a strided latent
input and an extra copy in the shared activation wrapper. The original hidden
state remains available to routing and the residual path.

On the qualified model, a 2,560-by-1,024 packed weight replaces the separate
512-row latent and 2,048-row shared projections in each of 15 sparse layers.
The trace confirms 15 fewer GEMM launches per decode step, retaining one output
activation kernel per sparse layer. The packed call retains SGLang's configured
unquantized GEMM dispatch, but the changed shape may choose a different kernel
or accumulation order. Bitwise full-model equality is not assumed.

## Validation so far

Tested runtime source: `7d039bbeef630b4481f806ce35ea9f12bc30f5a2`.
Pinned SGLang: `3145136dcd1238754e0ea2b2ffd546532119c71c`.
The H100 GPU precheck passed in 16.84 seconds on 2026-10-24:

- Exact split/activation outputs for empty, strided and representative decode
  shapes, including all finite BF16 gate inputs at three multiplier values.
- The real sparse-layer constructor under the automatic rounding selector,
  with only expert allocation/logging stubbed. It verifies that packing remains
  active after the temporary mode-selection context exits.
- Actual SGLang checkpoint loaders, two successive partial weight updates,
  stable parameter names/pointers, changed input values and CUDA graph replay.

The local portable suite passes (154 passed, 106 skipped), as do Ruff and
`git diff --check`. GPU tests supplement the local skips; they do not substitute
for the full-model numerical panel.

An earlier precheck failed because its isolated constructor fixture supplied
TP1 but omitted EP1, causing an uninitialized distributed-group lookup. Adding
the EP1 override corrected the fixture; all runtime source hashes remained
unchanged. That attempt used 20.70 seconds and never loaded the model.

## Whole-engine measurements

The completed run compares baseline `70abc4a`, candidate, then baseline on one H100 using
the frozen six-wave C64 panel: three fixed 1,024-token waves and three mixed RL
lengths up to 10,240 tokens. Memory fraction is 0.7, prefill chunk 8,192,
context 12,288, BF16 TP1/EP1, full decode graphs, with caching, route capture,
route return and original logprobs enabled. Startup, warmup, profiles and scoring
are excluded from throughput.

| Panel | Before tok/s | After tok/s | Bracketed control tok/s | Packed tok/s | Change |
|---|---:|---:|---:|---:|---:|
| Fixed | 5803.13 | 5781.76 | 5792.43 | 5836.37 | +0.759% |
| Mixed | 3626.81 | 3629.14 | 3627.98 | 3655.75 | +0.766% |

Fixed paired-wave changes are +0.630%, +1.096% and +0.476%; closing-control
variation is -0.368%. Mixed changes are +0.849%, +0.711% and +0.735%; control
variation is +0.064%. Three-wave percentile bootstrap intervals are
[+0.476%, +1.096%] and [+0.711%, +0.849%]. All six paired changes are positive;
three waves do not characterize uncertainty across machines and days.

Token IDs and full logprob vectors are retained and exactly equal against both
controls. The pinned Core scoring input/checkpoint hashes match all six waves,
so forced replay was unnecessary. Core uses `e505356`, BF16 evaluation, natural
routing, torch attention, one full request per document, no EP.

| Metric against Core | Control and packed | Packed minus control |
|---|---:|---:|
| Mean absolute error, nats/token | 0.03694615 | 0 |
| Signed mean, serving minus Core, nats/token | +0.00201359 | 0 |
| p95 / p99 absolute error, nats | 0.13185763 / 0.23219654 | 0 / 0 |
| Maximum absolute error, nats | 3.29934740 | 0 |
| Ratio outside [0.8, 1.2] | 1.64719976% | 0 pp |

The ratio is `exp(Core logprob - serving logprob)`. Zero added cost means
unchanged disagreement with the trainer, not perfect trainer agreement. Exact
values on this checkpoint/runtime/panel do not prove equality for all models,
GPU architectures or GEMM backends. Raw route contents were not separately
hashed by this panel; token/logprob equality is the tested agreement claim.

The two 64-step batch-64 profiles show 625 → 610 graph kernels and 198 → 183
dense-GEMM/reduction calls per step. The combined output kernel runs 15 times
per step (0.02450 ms total). Summed graph kernel time falls 8.17732 → 8.13106 ms;
dense GEMMs/reductions fall 0.96817 → 0.91736 ms. Expert GEMMs remain essentially
unchanged at 5.56222 → 5.55576 ms. These are profiled GPU kernel sums, not
whole-wave wall time. The finalized trace bytes match the analyzed copies.

B300, composed options, admission tuning and real-RL quality remain outside
this individual H100 investigation. Do not add this independent percentage to
other option rows without measuring the combination.

Runs:

1. Completed panel, all phases exited successfully:
   [Beaker](https://beaker.org/ex/01M5R0J1B7358D7FZYRYYTE3C0).
2. Earlier fixture failure, no model loaded:
   [Beaker](https://beaker.org/ex/01M5R0BFAG3CAZYE9Y6SNNHTR9).

Result dataset: `01M5R0J1BJRW2JAY7HNRCR00C9`, finalized 2026-10-24 22:46:56 UTC.
Core reference: `01M5QH3K19XS37AFA3DTAJGW7F`.
Analysis, hashes, reproducible scripts and attempt ledger:
`01M5R1N2FN1CWF80EDCVMYGETG`.
Payload: `01M5R0H1X46C91QHCWQ2XQT2XP`.

The successful job used 982.46 seconds (16m22s), including a 16.84-second GPU
precheck. The before/candidate/after processes took 341.32, 312.85 and 308.58
seconds, including their startup, warmup and any profiling. Those process times
are not the unprofiled throughput measurement. It ran as high-priority
unallocated work on one Jupiter H100 with a 40-minute timeout and no automatic
restart. Its predecessor waited 9m12s with
35 minutes of protection and was canceled before ever starting. Unallocated
placement admitted the two subsequent attempts in about 14 and 10 seconds;
this is observed scheduling history, not a guarantee for future submissions.
