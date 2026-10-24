# Grouped second-stage KDA gate investigation

Status: H100 GPU checks passed; whole-engine throughput and same-token trainer
agreement are being measured. A component timing improvement is established
under the microbenchmark conditions below, not an inference throughput gain.

## Option and implementation

Set `OLMO_SGLANG_KDA_GATE_PROJECTIONS=grouped` before starting the engine.
The default is `torch`; restart the engine when changing the setting. The
grouped path requires TP1, unquantized BF16 CUDA decode. Prefill and unsupported
layouts retain the original projections.

The forget and output gate second stages read different BF16 first-stage
outputs. One Triton launch assigns separate tiles to each GEMM, rather than
materializing a zero-filled block-diagonal matrix. It preserves both inputs
and the output-gate bias, accumulates in FP32 and returns two contiguous BF16
outputs. Original weight and bias tensors are read directly: checkpoint names,
parameter objects and loader mappings are unchanged, with no packed weight copy.

The tested model has 1,024-by-256 and 2,048-by-256 gate weights in each of
14 KDA layers. Combining two launches into one targets 14 fewer kernels per
decode step. Different reduction and bias accumulation can change last bits;
full-model agreement is measured, not assumed from the shapes or GPU unit tests.

This candidate is evaluated against the common baseline independently of
first-stage QKV/f1/beta/g1 packing and latent/shared packing. First-stage and
second-stage options modify the same forward path; a future combination needs
integration and its own throughput/agreement panel. Independent percentages
must not be added.

## GPU qualification and component timing

Tested runtime: `c98f11470255a54c94aad9cec813a97c307e6187`.
Pinned SGLang: `3145136dcd1238754e0ea2b2ffd546532119c71c`.
The H100 precheck passed in 22.51 seconds on 2026-10-24, covering:

- Empty and batch 1/7/16/64 outputs, tail columns, distinct strided inputs,
  bias, contiguous outputs and no input mutation. Integer-valued cases match
  native projections exactly.
- Random BF16 inputs against FP32 arithmetic rounded to BF16, using the stated
  numerical tolerance (`rtol=0.008`, `atol=0.004`), including changed-input and
  changed-weight/bias graph replay. This is not an exact-arithmetic assertion.
- Actual SGLang checkpoint loaders with two successive updates, unchanged
  parameter pointers/state-dict keys, and refreshed graph outputs.

Local Ruff/format/diff checks pass; the portable suite passes with 153 passed
and 105 skipped. GPU qualification supplements the local CUDA skips.

The diagnostic microbenchmark captures 100 same-input gate pairs per CUDA graph
and records nine event-timing samples per arm, after warmup. Weights remain warm
in cache. It excludes the rest of the engine and does not measure cache misses
after streaming expert weights. Arms run in fixed native-then-grouped order;
the full engine panel separately brackets the candidate with two controls.

| Batch | Native pair median, µs | Grouped pair median, µs |
|---|---:|---:|
| 1 | 5.3603 | 2.4134 |
| 16 | 5.4486 | 2.4390 |
| 64 | 5.6419 | 2.9120 |

Raw samples are in the run's `gpu-precheck.log`. Treat these as component
diagnostics, not an inference speedup or a reason to skip the divergence row.

## Whole-engine panel in progress

Baseline `70abc4a`, candidate, then baseline run on the same H100 with the
frozen six-wave C64 workload: three fixed 1,024-token waves and three mixed RL
length waves up to 10,240 tokens. Settings are BF16 TP1/EP1, context 12,288,
memory fraction 0.7, prefill chunk 8,192, full decode graphs, caching, route
capture/return and original logprobs. Startup, warmup, scoring and profiles are
excluded from throughput.

All output IDs and logprob vectors are saved. If free generations change, two
fresh engines force-score the same frozen tokens before either is compared to
the pinned Core reference. Forced-scoring timing is diagnostic only. The final
row will include throughput/control variation, mean absolute error, signed
bias, tails and probability-ratio outliers with prompt-cluster uncertainty.

Runs:

1. Active H100 panel, GPU precheck passed:
   [Beaker](https://beaker.org/ex/01M5R1JP7997A79YE6P4HGSKR0).

Payload: `01M5R02FHZ4YF9GEH701YP5NST`.
Active result: `01M5R1JP7JAVHGNX9KBRQGSCNY` (not finalized).
The job runs high-priority unallocated on one Jupiter H100, with a 40-minute
timeout and no automatic restart. No default, B300 qualification, composition,
admission setting or real-RL quality claim is included in this investigation.
