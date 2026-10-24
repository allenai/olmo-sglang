# Grouped second-stage KDA gate investigation

The independent H100 panel measures +0.400% fixed-length throughput and +0.669%
mixed-length throughput. All 384 requests / 1,035,090 output tokens and logprobs
match both controls exactly; every measured incremental Core-error metric is
zero. The option remains default-off, with no B300 or composition claim.

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
14 KDA layers. The full-engine trace confirms that replacing 28 native gate GEMMs with 14
grouped kernels removes 14 launches per decode step. Different reduction and bias accumulation can change last bits;
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

## Whole-engine results

Baseline `70abc4a`, candidate, then baseline run on the same H100 with the
frozen six-wave C64 workload: three fixed 1,024-token waves and three mixed RL
length waves up to 10,240 tokens. Settings are BF16 TP1/EP1, context 12,288,
memory fraction 0.7, prefill chunk 8,192, full decode graphs, caching, route
capture/return and original logprobs. Startup, warmup, scoring and profiles are
excluded from throughput.

| Panel | Before tok/s | After tok/s | Bracketed control tok/s | Grouped tok/s | Change |
|---|---:|---:|---:|---:|---:|
| Fixed | 5785.16 | 5799.11 | 5792.13 | 5815.27 | +0.400% |
| Mixed | 3617.65 | 3617.68 | 3617.67 | 3641.88 | +0.669% |

Fixed paired-wave changes are +0.038%, +0.928% and +0.122%; closing-control
variation is +0.241%. Mixed changes are +0.796%, +0.656% and +0.564%; control
variation is +0.00069%. Three-wave percentile bootstrap intervals are
[+0.038%, +0.928%] and [+0.564%, +0.796%]. All six paired changes are positive,
but the fixed gain is small and varies substantially between waves. Three waves
do not characterize variation across machines and days.

All token IDs and full logprob vectors are retained and equal against both
controls. The pinned Core input/checkpoint hashes match all six waves, so no
forced replay was needed. Core uses `e505356`, BF16 evaluation, natural routing,
torch attention, one full request per document, no EP.

| Metric against Core | Control and grouped | Grouped minus control |
|---|---:|---:|
| Mean absolute error, nats/token | 0.03694615 | 0 |
| Signed mean, serving minus Core, nats/token | +0.00201359 | 0 |
| p95 / p99 absolute error, nats | 0.13185763 / 0.23219654 | 0 / 0 |
| Maximum absolute error, nats | 3.29934740 | 0 |
| Ratio outside [0.8, 1.2] | 1.64719976% | 0 pp |

The ratio is `exp(Core logprob - serving logprob)`. Zero added cost means
unchanged trainer disagreement on this sample, not perfect agreement or a
universal bitwise guarantee. Raw route contents were not separately hashed by
this panel; token/logprob equality is the tested claim.

The 64-step batch-64 profiles have 625 → 611 graph kernels. Native dense-GEMM/
reduction calls fall 198 → 170, with 14 new `_grouped_gates` calls: 28 native
GEMMs become 14 grouped kernels. Summed graph kernel time falls
8.17502 → 8.14089 ms/step. The remaining native dense-GEMM/reduction time plus
new grouped kernels is 0.87786 + 0.05078 = 0.92864 ms/step, versus 0.96711 ms
for the original family. These are profiled kernel sums, not whole-wave wall
time; the analyzed trace bytes match the finalized result.

Runs:

1. Completed H100 panel, GPU checks and all timing arms passed:
   [Beaker](https://beaker.org/ex/01M5R1JP7997A79YE6P4HGSKR0).

Payload: `01M5R02FHZ4YF9GEH701YP5NST`.
Finalized result: `01M5R1JP7JAVHGNX9KBRQGSCNY` (2026-10-24 23:04:36 UTC).
Core reference: `01M5QH3K19XS37AFA3DTAJGW7F`.
Analysis, hashes, reproducible scripts and microbenchmark samples:
`01M5R2M2P60RTV1QTA0FHJ0A93`.

The successful job ran for 976.35 seconds (16m16s), including a 22.51-second
GPU precheck. Before/candidate/after processes took 325.90, 314.60 and 310.44
seconds, including startup, warmup and any profiling; these process durations
are not throughput measurements. Scheduling took 8.71 seconds. The job used
one Jupiter H100 at high priority, unallocated, with a 40-minute timeout and no
automatic restart. No default, B300 qualification, composition, admission setting
or real-RL quality claim is included in this investigation.
