# Q/k rounding on the common H100 panel

PR #21 rounds normalized KDA q/k to BF16 before converting back to FP32 for
the decode recurrence. On this panel it reduces signed serving/trainer mismatch
by about 94.7%, leaves mean absolute error approximately unchanged, and raises
the probability-ratio outlier rate by 0.038 percentage points. There is no
consistent throughput improvement. This is a measured numerical tradeoff,
not an adopted default or a speed optimization.

The frozen tokens were sampled from H100 main. A smaller signed mismatch on
those tokens does not by itself prove removal of a systematic numerical bias:
scoring a model's own samples tends to produce a positive log-ratio relative
to another model.
See the [composition results](decode-composition.md#sampling-distribution-and-signed-error)
for the cross-hardware evidence and interpretation. MAE and outlier changes
remain valid paired measurements on this fixed workload distribution.

## Source and workload

Baseline: `70abc4adf7b0993e95cb2fef4a7a63db998ecef9`.
Candidate: `019b82270e48f572a8e2e2c4756fc00ecb80efd8` from
[PR #21](https://github.com/allenai/olmo-sglang/pull/21).
SGLang: `3145136dcd1238754e0ea2b2ffd546532119c71c`.

The candidate changes source arithmetic unconditionally. `ISSUE31_KDA_QK_ARM`
is only a harness provenance label; it is not a production feature switch.
Runtime source inventories were checked in all three timed arms and both
forced-replay arms, along with checkpoint and trajectory hashes.

The [common panel contract](decode-option-results.md#common-measurement-contract)
uses one H100, BF16 TP1/EP1, admission 64, full decode graphs, memory fraction
0.7, chunked prefill 8,192, context 12,288, route capture/return, original
logprobs, and three fixed plus three mixed-length waves. The checkpoint and
requests are frozen. Startup, warmup, scoring and profiling are excluded from
the throughput rows.

## Throughput

| Panel | Before tok/s | After tok/s | Bracketed control tok/s | Candidate tok/s | Change |
|---|---:|---:|---:|---:|---:|
| Fixed, 1,024 output tokens/request | 5685.75 | 5707.56 | 5696.64 | 5720.48 | +0.419% |
| Mixed, up to 10,240 output tokens/request | 3569.39 | 3563.39 | 3566.39 | 3558.87 | -0.211% |

The paired fixed-wave changes are +0.661%, +0.087% and +0.581%; closing-control
drift is +0.384%. Mixed changes are +0.066%, -0.056% and -0.594%; control drift
is -0.168%. Three-wave percentile bootstrap intervals are [+0.087%, +0.661%]
and [-0.594%, +0.066%], respectively. These small panels do not establish a
consistent speed benefit or characterize variation across machines and days.

All 384 candidate free-generation trajectories change, whereas both controls
have identical token IDs and logprobs. Token counts and requested lengths are
fixed, but changed text can change expert routing and GPU work. Therefore the
small timing differences cannot be attributed solely to the added casts.

The ordinary profiles each contain 64 batch-64 decode steps and 625 graph
kernels per step. Summed graph kernel time is 8.32461 versus 8.34414 ms/step;
KDA recurrence time is 0.62610 versus 0.62746 ms/step. Profile bytes were verified
against the finalized result dataset. These explain work, not unprofiled
throughput, and they use different generated trajectories.

## Same-token trainer agreement

Two fresh serving engines force-score the same 1,035,090 frozen output tokens
from 384 requests / 128 prompt clusters, without generation warmup. Both use
the same ordered waves and cache protocol. The forced token selector preserves
the original model logprobs and performs CPU synchronization, so its timing is
diagnostic only. The pinned Core reference is `e505356`, BF16 evaluation,
natural routing, torch attention, one full request per document, no EP.

The replay control is its own numerical baseline. Its cache history differs
from the warmed free-generation control, so do not subtract the latter's
absolute metrics from this candidate.

| Metric | FP32 normalized q/k | BF16 normalized q/k | Candidate minus control |
|---|---:|---:|---:|
| Mean absolute error, nats/token | 0.03695125 | 0.03696627 | +0.00001502 |
| Signed mean, serving minus Core, nats/token | +0.00178905 | +0.00009522 | -0.00169383 |
| p95 absolute error, nats | 0.13192309 | 0.13197544 | +0.00005234 |
| p99 absolute error, nats | 0.23230754 | 0.23197025 | -0.00033729 |
| Maximum absolute error, nats | 3.29934740 | 1.25241661 | -2.04693079 |
| Ratio outside [0.8, 1.2] | 1.66400989% | 1.70197761% | +0.03796771 pp |

The ratio is `exp(Core logprob - serving logprob)`. The prompt-clustered
95% intervals for candidate-minus-control changes are:

- Mean absolute error: [-0.00008758, +0.00012003] nats/token.
- Signed mean: [-0.00182597, -0.00155124] nats/token.
- Outlier rate: [+0.00722772, +0.06889621] percentage points.

The outlier interval is above zero on this sample; the result should not be
described as an observed zero cost. Maximum and percentile tail metrics do not
have bootstrap intervals here. The full analysis retains per-prompt, domain
and output-position breakdowns. This is not a real-RL quality or routing-replay
training evaluation.

The older PR #21 study uses 1,185,618 tokens and different cache/memory/prefill
settings. Its audited result independently shows the same direction: lower
signed mismatch, nearly unchanged mean absolute error, and +0.04428 pp outliers. It is
corroborating evidence, not part of this common-panel row. Similar pairwise
BF16 errors are consistent with accumulation-order noise; they do not by
themselves establish an irreducible error floor for every implementation.

## Runtime and artifacts

The job ran for 1,788.37 seconds (29m48s), excluding queue/finalization. Timed
panel processes, including their startup and profiling, took 352.59, 332.32
and 326.06 seconds. Replay control and candidate took 383.82 and 392.15 seconds.
All five phases exited successfully; finalization was 2026-10-24 22:16:27 UTC.

Runs:

1. Common H100 panel and paired replay:
   [Beaker](https://beaker.org/ex/01M5QY1TE4BQRW8BYFCVHFFACB).
2. Earlier, separate-reference study:
   [Beaker](https://beaker.org/ex/01M5QPYX8EQ1WAWEWTBVN1JFVN).

Datasets:

- Raw common-panel outputs, full token/logprob vectors and profiles:
  `01M5QY1TEBVDFMYBSK3TKWHTHR`.
- Common Core scoring reference: `01M5QH3K19XS37AFA3DTAJGW7F`.
- Analysis, source/profile hashes and reproducible analyzers:
  `01M5QZWDFAQGCP8Y2A5HR2MKME`.
- Payload: `01M5QXWC77371QP56B9H2GTSZ2`.

The [machine-readable menu](data/decode-option-results-20261024.json) retains
the throughput waves, controls, numerical intervals, source revisions and
terminal attempt. No default, merge or deployment is implied by this report.
