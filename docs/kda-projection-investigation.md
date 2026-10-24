# KDA first-stage projection packing

Packing the four KDA projections that read the same hidden state improves the
frozen H100 panel by 1.786% fixed-length and 2.207% mixed-length throughput.
All generated token IDs and logprobs match both controls exactly. This is an
opt-in building block, not a default change or a composition result.

## Implementation and scope

Set `OLMO_SGLANG_KDA_PROJECTIONS=packed` before starting the engine. The default
is `torch`; restart to change it. The packed path requires TP1, unquantized
weights and BF16 CUDA decode. Prefill and unsupported configurations retain
the original projection calls.

QKV, f1, beta and g1 read the same hidden tensor. Their existing named parameters
become contiguous row views into one persistent packed buffer. The original
parameter objects, checkpoint mappings and loader attributes remain in place;
in-place updates write through those views. Device/dtype moves rebuild them
before graph capture. There is no per-step weight concatenation or stale duplicate.

The combined call uses SGLang's unquantized linear dispatch. It retains the
configured dispatch setting, but changed shapes can select different underlying
kernels. The f1→f2 and g1→g2 boundaries still materialize BF16, and g2 retains its
bias. Packed beta has a padded row stride, which the existing recurrence now
accepts directly; non-unit head stride still takes a contiguous fallback.

For this checkpoint, the first-stage output has 4,616 rows:
4,096 QKV + 256 f1 + 8 beta + 256 g1, with input width 1,024.
Four GEMMs become one in each of 14 KDA layers. The dependent second-stage gates
read different inputs and are a separate investigation. Latent/shared-expert
projection packing is also separate. This change does not implement all of the
earlier roughly 75-call packing proposal.

## Whole-engine measurement

The before/candidate/after panel uses the same H100, checkpoint, frozen six waves,
C64 admission, TP1/EP1, BF16, 0.7 memory fraction, 8,192-token prefill chunks,
12,288-token context, radix cache with extra-buffer Mamba state, and full decode
graphs. Route capture and return, original logprobs, and PyTorch sampling remain
enabled. Sampling is temperature 1, top-p 1, top-k -1, with EOS ignored. Each panel
retains all 384 requests and 1,035,090 output tokens. Three fixed waves request
1,024 tokens; three mixed waves use frozen RL lengths up to 10,240 tokens.

| Workload | Opening control tokens/s | Closing control tokens/s | Bracketing control tokens/s | Packed tokens/s | Change |
|---|---:|---:|---:|---:|---:|
| Fixed | 5,774.05 | 5,744.37 | 5,759.17 | 5,862.03 | +1.786% |
| Mixed | 3,614.58 | 3,611.56 | 3,613.07 | 3,692.80 | +2.207% |

Fixed paired-wave changes are +0.948%, +2.510%, +1.755%; mixed changes are
+2.227%, +2.343%, +2.075%. All six improve. Closing-control drift is -0.514%
fixed and -0.083% mixed. Three-wave bootstrap intervals are [+0.948%, +2.510%]
and [+2.075%, +2.343%], respectively; these describe this short panel, not
variation across machines, days, checkpoints or workloads.

Throughput includes admission and drain but excludes engine startup, warmup,
profiling and scoring. The three phase runtimes, including those overheads, are
348.786 / 335.111 / 336.428 seconds. The complete process takes 1,044.898 seconds
(17m25s), including a 21.851-second GPU precheck. No forced replay was needed
because the free trajectories and logprobs already match exactly.

## Trainer agreement

Both the candidate and closing control have zero changed token IDs and zero
changed logprobs against the opening control on all 1,035,090 output tokens.
The existing Core reference is reused only after all six input-file hashes and
checkpoint inventories match. Its contract is Core `e505356`, BF16 evaluation,
natural routing, torch attention, one full request per document, and no EP.

| Metric against Core | Control | Packed | Change |
|---|---:|---:|---:|
| Mean absolute logprob error | 0.03694615 nats | 0.03694615 nats | 0 |
| Signed mean error | +0.00201359 nats | +0.00201359 nats | 0 |
| p95 absolute error | 0.13185763 nats | 0.13185763 nats | 0 |
| p99 absolute error | 0.23219654 nats | 0.23219654 nats | 0 |
| Maximum absolute error | 3.29934740 nats | 3.29934740 nats | 0 |
| Probability ratio outside [0.8, 1.2] | 1.64719976% | 1.64719976% | 0 pp |

Ratios are exp(Core logprob − serving logprob). Exact agreement on this panel
does not prove bitwise identity for every GPU, GEMM shape, checkpoint or kernel
version. A changed GEMM shape is a reason to measure divergence, not proof that
divergence must occur. Real-RL routing-replay scoring and learning effects remain
part of the later workload phase.

## Trace and validation

Both profiles contain 64 annotated batch-64 decode steps. Graph kernels drop
from 625 to 583 per step, exactly 42. Summed graph kernel time falls from
8.171727 to 8.019894 ms, a 0.151833 ms reduction. The dense-GEMM/reduction family
falls from 0.965453 to 0.809626 ms. Expert GEMM time is essentially unchanged
(5.561755 versus 5.560818 ms); recurrence time is also unchanged
(0.620867 versus 0.620655 ms). Profiles are diagnostic; unprofiled waves determine
the throughput result.

The GPU precheck exercises actual SGLang checkpoint loaders, repeated partial
updates, CUDA graph replay with updated weights, stable parameter/storage
mappings, and six contiguous-versus-strided beta recurrence cases (C1/C16/C64,
both eigenvalue modes). Those recurrence outputs and states match exactly.
Portable tests pass: 156 passed, 104 CUDA-dependent skipped; Ruff and formatting
checks pass. The actual GPU checks run before the benchmark on the same image.

Baseline source is `70abc4`; tested candidate source is
`edc0ae582defcd9241d83f544e12393d7f3b8a2d`. Runtime inventories match all
manifested Python/configuration files in all three phases. SGLang is
`3145136dcd1238754e0ea2b2ffd546532119c71c`, PyTorch 2.13.0+cu130 and Triton 3.7.1.
The tested checkpoint is the frozen Olmo 3.5 KDA MoE SFT step5402 HF export;
the result reports include the full path and file hashes.

Runs:

1. [H100 before/packed/after](https://beaker.org/ex/01M5QWZCQ5DAYCYNEG74N8GC6X):
   one H100 on Jupiter, `open-instruct-dev`, high priority, 35-minute protection,
   40-minute timeout, no automatic retry. Started 2026-10-24 21:27:45 UTC;
   finalized 21:45:14 UTC, exit 0.

Raw result: `01M5QWZCQJ3V3GBTY6TK32DM8Q`. Immutable payload:
`01M5QNQNGVTASYGX57M7642RXJ`. Core result: `01M5QH3K19XS37AFA3DTAJGW7F`.
Local evidence is under
`open-instruct/runs/issue31-decode-menu-20261024/05-kda-packing/`.
Earlier payload versions were superseded before launch and provide no measurements.

The [analysis archive](https://beaker.org/ds/01M5QY537AVHBBVMDZT6VM9TBQ) contains
the throughput/trainer audits, trace summaries, source manifests, phase runtimes,
precheck log, analysis scripts and SHA-256 checksums. Its README gives commands
to regenerate the reports from the raw and Core datasets.
