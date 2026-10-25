# Route capture and return attribution

This investigation separates the cost of recording expert routes from the cost
of returning request histories. On the frozen H100 C64 workload, disabling capture
improves throughput by about 0.3%; omitting returned histories adds 3.53% on fixed
lengths and 0.68% on mixed lengths. These are attribution controls, not equivalent
replacements for rollout routing replay. A subsequent route-preserving panel
tests the existing MessagePack IPC switch: +0.85% fixed-length throughput and
+0.19% mixed (inconclusive), with identical tokens, logprobs, and route histories.

## Whole-engine results

All five uninstrumented panels completed on the same H100 with the same source,
checkpoint, requests, memory/cache settings, graphs, sampling, and logprobs.
Each panel contains three fixed-length and three mixed-length waves, retaining
384 requests and 1,035,090 generated tokens.

| Configuration | Fixed tokens/s | Mixed tokens/s |
|---|---:|---:|
| Capture and return, mean bracketing time | 5,783.52 | 3,629.14 |
| Capture only, mean bracketing time | 5,987.85 | 3,653.66 |
| Capture off | 6,005.77 | 3,666.39 |

| Contrast | Fixed throughput change | Mixed throughput change |
|---|---:|---:|
| Disable capture, relative to capture-only controls | +0.299% | +0.348% |
| Omit histories, retaining capture | +3.533% | +0.675% |
| Disable both, relative to full return controls | +3.843% | +1.026% |

All three paired waves improve in each contrast. The capture-only closing-control
drift is -0.026% fixed / -0.185% mixed; the full-return control drift is +0.159%
fixed / +0.024% mixed. Per-wave throughput changes for omitting histories range
from +1.935% to +4.935% fixed and +0.306% to +0.922% mixed. These short panels
establish a workload-specific cost; they are not a forecast for a route-preserving
change or a confidence interval over other machines and days.

All five panels have exactly identical output token IDs and logprob values.
The existing trainer reference was reused only after all six opening-control
input-file hashes and checkpoint identities matched. Every incremental mean,
bias, p95, p99, maximum-error, and ratio-outlier cost is zero on these trajectories.
Absolute agreement remains 0.03694615 nats/token mean absolute error, 0.13185763
p95, 0.23219654 p99, and 1.64719976% of token probability ratios outside [0.8, 1.2].
The reference is pinned Core BF16 evaluation with natural routing, torch attention,
one full request/document, and no EP; it is not a real-RL routing-replay evaluation.

The full-return panel returns 1,155,807,232 raw route bytes (1.076 GiB), including
prompt histories and dense-layer slots. That volume is not itself evidence that
encoding is the bottleneck.

## Pinned implementation

The source under test is SGLang
`3145136dcd1238754e0ea2b2ffd546532119c71c`, with Olmo-SGLang
`70abc4adf7b0993e95cb2fef4a7a63db998ecef9`.

The existing overlap path already does the following:

1. `BaseDeviceCache.capture` writes each layer's selected experts to a GPU buffer.
2. `BaseTopkCapturer.on_forward_end` clones the current routes and cache locations
   before the next overlapping forward can overwrite them.
3. `TopkCaptureOutput.map_device_tensors` uses `_async_d2h`, which allocates pinned
   CPU storage, copies nonblocking, and records the source's stream lifetime.
4. After the result-copy event completes, `TopkCaptureOutput.finalize` scatters
   routes into the host history indexed by cache locations.
5. For a completed request that asks for routes,
   `_maybe_collect_routed_experts` calls `get_topk`. It copies the request's cache
   indices to the CPU and gathers the corresponding history. The requested range
   is `[start_len, prompt_length + output_length - 1)`.
6. With `skip_tokenizer_init=True`, as in this benchmark, the tokenizer manager
   converts populated route tensors to bytes and base64. The text-tokenizer path
   instead uses the detokenizer's `_b64_encode_per_request`. Full histories are
   gathered and encoded at completion, not on every decode step.

Therefore, “make copies asynchronous and return routes once” describes existing
behavior. Possible remaining costs include buffer allocation/cloning, host
scatter, completion index transfer/gather, serialization, and IPC. Their exposed
latency must be measured before selecting a change.

## Matched experiment

The immutable payload is `01M5QN3ND8200X2NR706QE440P`. It verifies the relevant
installed SGLang source hashes before any timing. Five fresh engines execute the
same frozen six C64 waves in this order:

| Panel | Capture | Return histories |
|---|---|---|
| return-before | On | On |
| capture-before | On | Off |
| capture-off | Off | Off |
| capture-after | On | Off |
| return-after | On | On |

The checkpoint, BF16 TP1/EP1, memory fraction 0.7, context 12288, prefill chunk
8192, decode graphs, radix/Mamba cache configuration, original logprobs, and
sampling settings remain fixed. Three fixed-length and three mixed-length waves
retain every request's output tokens and logprobs. Startup, warmup, and profiling
are excluded from timed throughput. Whole-wave time includes admission and drain.
The benchmark's return path also decodes base64 in the client to validate the
full route envelope, as a route-consuming client must do. The return contrast
therefore includes that client cost as well as server gather/encoding and IPC.

The middle panel is compared with its bracketing capture-only controls. To
estimate the return cost, the two capture-only timings are averaged for each
corresponding wave and compared with the outer return controls. Report individual
waves and control drift alongside aggregate changes. A short three-wave panel
does not establish day-to-day or cross-machine uncertainty.

Compare numerical vectors to the common Core reference only after validating
checkpoint, input, and trajectory hashes. Capture-off and no-return have different
routing-replay capabilities even if their token/logprob results match exactly.

## Separate diagnostic profiles

Only after all five uninstrumented timing panels, the job decorates the pinned
source with CPU timers and profiler ranges. It then runs separate return,
capture-only, and capture-off engines over a full C64 128-token wave, including
completion. It records route payload bytes and hashes, per-process probe samples,
and CPU/GPU traces. Modified source and hashes are retained with the results.
In these diagnostic profiles, client-side base64 decoding and envelope validation
occur after profiling stops. Server probe timings alone cannot explain that
portion of the unprofiled return contrast.

The analysis distinguishes graph kernel work, outside-graph work, and true idle
between replays of the same graph shape. CPU ranges are correlated to GPU calls
and checked for overlap with recorded GPU activity. A CPU event wait or a long
CPU function is not automatically a GPU stall. Nested labels must not be added
together. Graph-captured per-layer copies need not have repeated Python labels
during replay; compare their GPU kernel counts as well.

The timers and counter snapshots perturb execution. Profiled wall times are
diagnostics, not throughput measurements. Serialization in another process may
appear in its probe counters without appearing in the scheduler's trace. Temporal
overlap with a GPU gap does not by itself establish causality.

## Status and artifacts

The protected repeat completed successfully at 2026-10-24 20:22:31 UTC and
finalized at 20:22:39 UTC. Process runtime was 1,773.873 seconds (29m34s), including
all five timing panels and three separate profiles. Engine startup ranged from
31.1 to 57.9 seconds and warmup from 1.2 to 6.7 seconds per timing panel; neither
is included in throughput. The first engine's longer startup accounts for most
of its longer process duration.

### Profile findings

Each profile contains 128 graph replays explicitly correlated to CPU annotations
`step[DECODE bs=64]`. All three profile modes produce the same 8,192 output token
IDs and logprobs. The return profile includes 24,498,176 raw route bytes.

| Diagnostic quantity | Capture + return | Capture only | Capture off |
|---|---:|---:|---:|
| Graph kernels per decode step | 625 | 625 | 610 |
| Summed graph kernel time | 8.553 ms | 8.555 ms | 8.534 ms |
| Median gap between graph replays | 338.5 us | 333.7 us | 328.8 us |
| Median GPU work in that gap | 277.4 us | 277.4 us | 273.3 us |
| Median true idle in that gap | 60.7 us | 56.6 us | 55.4 us |

The 15 extra graph kernels are route-buffer copies, totaling 21.205 us per step.
This agrees with the small whole-engine capture cost. The gap components are
separately computed medians and need not add exactly to the median total gap.
Profiled kernel time is not an unprofiled step-latency measurement.

In the return trace, CPU range unions for forward-end clones, route D2H, and host
scatter total 14.2, 23.8, and 7.7 ms across the wave. Approximately 97%, 90%, and
91% overlap GPU work, respectively. They are nested/concurrent diagnostics,
not additive costs. There is no evidence here for a recoverable 0.3 ms of
per-step idle caused by capture.

Completion has a more visible cost: 64 `get_topk` calls total 22.1 ms, with
19.9 ms outside recorded GPU activity. Scheduler `sock_send` ranges total
50.1 ms, dominated by one 48.7 ms send; capture-only sends total 2.1 ms.
This identifies completion gather/IPC as a better target than per-step transport,
but the initial send range does not distinguish serialization from socket work.

The initial base64 probe was on the detokenizer path, which this token-ID workload
bypasses. Its zero counter is missing coverage, not zero encoding cost. A short
profile-only follow-up instruments the actual tokenizer-manager encoding and
separates pickle serialization/deserialization from socket operations. It retains
the exact tensor-to-bytes/base64 expression and pickle protocol. It does not rerun
or alter the completed uninstrumented timing panels.

The trace analyzer excludes Kineto's `gpu_user_annotation` mirrors when counting
CPU ranges; otherwise some function-call counts would incorrectly double. It
correlates CPU graph launches to GPU replays and validates interval-union logic
with synthetic overlap cases and an existing 64-step baseline trace.

The first started run was [route attribution](https://beaker.org/ex/01M5QPFAEY0QSWCHD8QCJDVPPV),
one H100 on Jupiter, high priority, unallocated, with a 40-minute timeout and no
automatic resume. It started at 2026-10-24 19:37:03 UTC and passed the source
preflight, then was preempted at 19:41:30 UTC by an urgent allocated one-GPU job.
It exited at 19:41:37 UTC after 273.834 seconds. Five of six opening-control waves
were saved; no complete timing comparison or profile was obtained. Its finalized
partial dataset is `01M5QPFAF5FP2PQS1S68G5TDHA` and is retained separately. Two preceding protected attempts were canceled before starting after
scheduler events identified an urgent eight-GPU queue blocker. Partial or
preempted timing ladders will not count as complete comparisons.

Local evidence lives under
`open-instruct/runs/issue31-decode-menu-20261024/04-route-capture/`.
The finalized full result dataset is `01M5QQ8T50ZCT60QAZ4CW9Z853`.
It contains the five timing panels, all token/logprob vectors, three traces,
per-process probe counters, and verified/instrumented source hashes.

The reproducible [analysis archive](https://beaker.org/ds/01M5QV1R3CAGTXCEEE2HHP6FZY)
contains the timing/trainer audit, all three trace summaries, both probe reports,
analysis scripts, source manifests, an attempt ledger, and SHA-256 checksums.
Its README gives the commands and raw/reference dataset IDs needed to regenerate
the reports. The corrected follow-up is retained separately from the initial
profile so the missing initial base64 coverage remains visible.

A protected replacement, [same-payload repeat](https://beaker.org/ex/01M5QQ8T4TDW5WWD80YPVG6HXA),
was submitted at 19:47:42 UTC and scheduled at 19:49:40 UTC. It uses one H100
on Jupiter, high priority, 35-minute protection, a 40-minute timeout, and no
automatic resume. Its result dataset is `01M5QQ8T50ZCT60QAZ4CW9Z853`.

The [completed profile-only follow-up](https://beaker.org/ex/01M5QT4BG1AK64TMXH5PGPAG4M)
uses payload `01M5QSN5J4FWVFG40M11NZHQ54`, one H100 on Jupiter, high priority,
unallocated, a ten-minute timeout, and no automatic resume. Process runtime was
159.310 seconds (2m39s); the job finalized at 20:40:50 UTC. The earlier protected
attempt `01M5QSPGHRNVHAB0CMDK999AVE` was canceled before starting after scheduler
events identified a queue blocker. No duplicate profile execution occurred.

Its tokenizer/pickle probes passed local contiguous/strided base64 parity,
synchronous/asynchronous pickle round-trip, flag, and byte-counter checks.
All six installed source hashes matched the pin, and all child processes exited
successfully. Both profile modes exactly reproduce the original profiles' 8,192
output token IDs and logprobs; the returned route bytes and SHA-256 digests also
match. The 64 tokenizer base64 calls account for all 24,498,176 returned raw bytes.

| CPU diagnostic across the full profile wave | Capture + return | Capture only |
|---|---:|---:|
| Completion gather, 64 calls when returning routes | 23.262 ms | None |
| Tokenizer base64 conversion, 64 calls | 15.369 ms | None |
| Pickle encode, 8 calls across processes | 34.739 ms | 3.075 ms |
| Pickle decode, 8 calls across processes | 17.976 ms | 4.467 ms |
| IPC send, including encoding, 8 calls | 54.298 ms | 10.121 ms |
| Serialized bytes, independently counted at encode and decode | 24,941,429 | 423,407 |

These are CPU durations across processes, with nesting and overlap; they must not
be summed into elapsed latency. The largest return encode and decode calls take
31.882 ms and 14.499 ms, respectively. Serialization is a concrete candidate to
measure, but the profile does not establish a recoverable throughput gain.
The finalized follow-up dataset is `01M5QT4BGAZED8TAKBKXJJ3MRS`.

### Route-preserving MessagePack option

The pinned runtime already supports `SGLANG_USE_PICKLE_IPC=false`, which switches
its typed interprocess messages to MessagePack. Its tensor codec preserves shape,
dtype, and raw contiguous tensor bytes. This is an existing runtime option, not a
new model kernel. Pickle remains the default.

The completed controlled panel compares pickle / MessagePack / pickle with identical
model source and the same frozen six-wave workload. It retains capture and full
returned histories, records the actual codec dispatch, and compares SHA-256
of every returned history alongside all output tokens and logprobs. Hashing runs
in all three timed paths; writing the sidecar occurs outside timing. This panel
qualifies inference messages only. Live weight updates, other control messages,
and real-RL integration require their own later validation.

| Workload | Bracketing pickle tokens/s | MessagePack tokens/s | Throughput change | Individual paired waves |
|---|---:|---:|---:|---|
| Fixed | 5,748.35 | 5,797.50 | +0.855% | +1.322%, +0.616%, +0.680% |
| Mixed | 3,609.02 | 3,615.95 | +0.192% | +0.227%, -0.016%, +0.334% |

Closing-control drift is +0.388% fixed and -0.045% mixed. Three-wave bootstrap
intervals are [+0.616%, +1.322%] fixed and [-0.016%, +0.334%] mixed; these describe
this short panel, not variation across machines/days. The mixed result is
inconclusive. All 384 requests, 1,035,090 output token IDs/logprobs, and SHA-256
digests of all 1,155,807,232 raw route bytes match both controls exactly. All six
Core reference input hashes match; incremental mean error, signed bias, tails,
maximum error, and ratio-outlier costs are zero. The absolute trainer metrics
remain those reported above.

Both arms use Olmo-SGLang `70abc4`; the only treatment is
`SGLANG_USE_PICKLE_IPC=false`. Installed dispatch was verified as true/false/true.
Set the environment variable before starting the engine and all its subprocesses;
restart to change it. Pickle remains the default. Completion base64 and the client
API are unchanged. The ordinary 64-step profiles retain 625 graph kernels per
step and approximately 8.17 ms of graph kernel work. Those profiles stop before
request completion, so they do not measure completion serialization itself.

The [completed panel](https://beaker.org/ex/01M5QVQPB46F2J210XJR9PDY31)
ran on one H100 on Jupiter, high priority, with 22-minute protection, a 25-minute
timeout and no automatic resume. Process runtime was 980.647 seconds (16m21s);
opening control/candidate/closing control took 343.527/314.239/309.911 seconds,
including their startup/warmup and applicable profiling. Throughput above excludes
that overhead. Raw dataset: `01M5QVQPBFVAZB1P5SPC6VKJQ0`; payload:
`01M5QVP9M3C7XPWTKSADKHNPT3`.
The [first attempt](https://beaker.org/ex/01M5QTWWMHG6N1FFD4GSK9434B)
failed after 11.917 seconds, before model loading, because the harness's custom
probe used a map-style struct in an array-style MessagePack union. Correcting
the probe to match the installed schema resolved it; no partial timings were used.

### Local CPU characterization

A local test isolates the codecs using the pinned `BatchTokenIDOutput` definition
and tensor/array hooks, extracted unchanged from SGLang source. Token IDs and
logprobs come from the frozen panel; route values are synthetic contiguous int32
tensors with the same per-request byte counts, including prompt history. The
decoder union includes the tested message, explicit pickle wrapper, and primitives;
unused message tags are omitted. This is codec work, excluding sockets, base64,
scheduling and GPU transfers. It is not another engine throughput measurement.

Two fresh Python processes each run 21 randomized encode/decode timing rounds,
after five warmups, with calibrated blocks of at least approximately 20 ms. The
process is pinned to one CPU; GC stays enabled, with collection outside each block.
Results include destruction of returned objects. Hardware: Ryzen 9 9950X;
Python 3.12, PyTorch 2.11.0+cu130, msgspec 0.19.0, pickle protocol 5. The GPU panel
uses PyTorch 2.13, so these are local characterization results, not server CPU
latency predictions. Every populated field and route tensor round-trips exactly.

| Synthetic message using frozen token histories | Pickle encode / decode, µs (run 1 median) | MessagePack encode / decode, µs (run 1 median) | Encode + decode speed ratio, two runs |
|---|---:|---:|---:|
| 64 incremental results, no routes | 31.75 / 29.46 | 33.52 / 26.84 | 1.01× / 1.00× |
| One 1,024-token completion, no routes | 15.96 / 26.75 | 6.73 / 18.07 | 1.72× / 1.73× |
| One 1,024-token completion, 1.24 MB routes | 75.51 / 94.16 | 45.44 / 57.38 | 1.65× / 1.65× |
| Eight 1,024-token completions, 9.99 MB routes | 2,748.86 / 1,072.27 | 445.59 / 560.82 | 3.80× / 3.93× |
| 64 1,024-token completions, 83.22 MB routes | 31,385.97 / 33,164.81 | 21,492.45 / 18,617.32 | 1.61× / 1.73× |
| One 10,240-token completion, 10.54 MB routes | 837.01 / 946.95 | 682.30 / 803.78 | 1.20× / 1.29× |

The ratio is the sum of separate median encode/decode times for pickle divided
by that sum for MessagePack, not a measured socket round trip. Large allocations
show more variation: in run 1, the 83 MB case's encode/decode p10–p90 intervals are
29.08–34.95 / 23.50–36.19 ms for pickle and 18.65–24.13 / 10.82–22.30 ms for
MessagePack. Both fresh-process medians favor MessagePack on completed histories.
The whole 64-request message is a completion-burst stress case, not a claim that
the scheduler always emits all completions in one message. Batch size and allocation
behavior materially affect the ratio; the eight-request peak is not universal.

Serialized size falls from 7,746 to 4,319 bytes for the small incremental message,
but only from about 85.070 MB to 85.037 MB for the largest route-bearing message
(0.039%). Raw int32 route bytes dominate, so the meaningful benefit there is CPU
serialization cost, not compression. Tiny incremental messages are essentially
tied overall; MessagePack encoding alone is slightly slower in that case.

Useful primary references:

- [msgspec's benchmarks](https://msgspec.dev/benchmarks) compare typed/untyped
  messages and encoding libraries, and explicitly discuss hot-loop limitations.
  They do not establish performance for these PyTorch payloads.
- [Quickle's published benchmark and source](https://jcristharif.com/quickle/benchmarks.html)
  directly compare pickle and MessagePack over 1, 1,000, and 10,000 structured
  objects. They explain how pickle memoization can win for repeated objects and
  strings; a format name alone does not establish a universal winner.
- [PyZMQ's serialization documentation](https://pyzmq.readthedocs.io/en/latest/howto/serialization.html)
  confirms `send_pyobj` uses ordinary pickle serialization. Multipart buffer
  transport is a separate design, not what this switch implements.

The smaller whole-engine gain is consistent with removing CPU work that is only
a small part of the current critical path. This makes MessagePack a measured,
route-preserving option that can reduce future CPU pressure as decode improves;
it does not establish that serialization is the next dominant bottleneck.

The [MessagePack analysis archive](https://beaker.org/ds/01M5QXH778MVPVWRQEC3H3SZYF)
contains both CPU runs with every timing sample, p10/median/p90 summaries, source
and input hashes, full engine/trainer/route audits, and reproduction commands.
The benchmark is also checked in as
[`tools/benchmark_ipc_codecs.py`](../tools/benchmark_ipc_codecs.py). With the raw
panel downloaded to `./msgpack-result` and the pinned SGLang commit available in
`../sglang`:

```bash
python tools/benchmark_ipc_codecs.py \
  --sglang-repo ../sglang --panel ./msgpack-result \
  --output cpu-codecs.json --seed 31
```

Repeat with `--seed 73` and a different output file for the second randomized
ordering. The script checks all populated message fields, exact tensor values,
and every request's route byte count before timing. It needs `torch`, `numpy`,
and `msgspec`; no installed SGLang runtime or GPU is required.
