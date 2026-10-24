# Latent/shared-expert projection investigation

Status: GPU correctness checks passed; the whole-engine throughput and
same-token trainer-agreement panel is running. No performance gain or full-model
agreement claim is established yet.

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
The intended saving is 15 GEMM launches per decode step, retaining one output
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

## Measurement in progress

The run compares baseline `70abc4a`, candidate, then baseline on one H100 using
the frozen six-wave C64 panel: three fixed 1,024-token waves and three mixed RL
lengths up to 10,240 tokens. Memory fraction is 0.7, prefill chunk 8,192,
context 12,288, BF16 TP1/EP1, full decode graphs, with caching, route capture,
route return and original logprobs enabled. Startup, warmup, profiles and scoring
are excluded from throughput.

Token IDs and full logprob vectors are retained. If free trajectories change,
paired fresh engines force-score the same frozen tokens against the pinned
Core reference. The report will include mean absolute error, signed bias,
tails and ratio outliers, with prompt-cluster uncertainty. Forced-scoring timing
is diagnostic only. B300, composed options, admission tuning and real-RL quality
are outside this individual H100 investigation.

Runs:

1. Active panel, GPU precheck passed:
   [Beaker](https://beaker.org/ex/01M5R0J1B7358D7FZYRYYTE3C0).
2. Earlier fixture failure, no model loaded:
   [Beaker](https://beaker.org/ex/01M5R0BFAG3CAZYE9Y6SNNHTR9).

Active result dataset: `01M5R0J1BJRW2JAY7HNRCR00C9` (not finalized).
Payload: `01M5R0H1X46C91QHCWQ2XQT2XP`.
The active run is high-priority unallocated work on one Jupiter H100, with a
40-minute timeout and no automatic restart. Its predecessor waited 9m12s with
35 minutes of protection and was canceled before ever starting. Unallocated
placement admitted the two subsequent attempts in about 14 and 10 seconds;
this is observed scheduling history, not a guarantee for future submissions.
