# Decode option measurements

This is the second-pass menu of inference building blocks. It records measured
throughput and trainer agreement separately. The baseline is `70abc4a`, after
the first-pass defaults in #15 and the default-off fused router in #10. The new
kernel switches below are on their linked draft branches, not yet on main as of
2026-10-24. MessagePack is already an option in the pinned SGLang runtime.

This document is in progress: the q/k rounding, latent/shared projection and
grouped second-stage gate investigations are not yet complete. No combined
speedup or real-RL quality result is claimed.

## Measured options

Each row compares its option independently with a before/after control on one
H100. All six completed rows have exactly matching output tokens and logprobs
on 384 requests / 1,035,090 output tokens, against both controls. Therefore every
incremental trainer-error and ratio-outlier metric is zero on those trajectories.
This is observed equality on the tested checkpoint/runtime/workload, not a
universal arithmetic guarantee.

| Building block | Opt-in setting | Fixed throughput change | Mixed throughput change | Evidence |
|---|---|---:|---:|---|
| Shared-expert add into post-MLP norm | `OLMO_SGLANG_SHARED_ADD_NORM=fused` | +0.843% | +0.169% | [PR #17](https://github.com/allenai/olmo-sglang/pull/17) |
| Routing normalization/divide/scale/cast tail | `OLMO_SGLANG_ROUTING_TAIL=fused` | +0.059%, inconclusive | +0.985% | [PR #18](https://github.com/allenai/olmo-sglang/pull/18) |
| FP32 router input emitted by pre-MLP norm | `OLMO_SGLANG_ROUTER_INPUT=fused` | +0.860% | +0.623% | [PR #19](https://github.com/allenai/olmo-sglang/pull/19) |
| Sampler selection kernels | `OLMO_SGLANG_SAMPLER=fused` | -0.339%, inconclusive | +0.733% | [PR #20](https://github.com/allenai/olmo-sglang/pull/20) |
| MessagePack IPC, preserving route return | `SGLANG_USE_PICKLE_IPC=false` | +0.855% | +0.192%, inconclusive | [PR #22](https://github.com/allenai/olmo-sglang/pull/22) |
| KDA first-stage QKV/f1/beta/g1 packing | `OLMO_SGLANG_KDA_PROJECTIONS=packed` | +1.786% | +2.207% | [PR #23](https://github.com/allenai/olmo-sglang/pull/23) |

All new kernel switches default to `torch`; SGLang IPC defaults to pickle. Set
the chosen variable before engine/subprocess startup and restart to change it.
The linked investigations document guards and fallback behavior. For example,
the sampler retains native RNG and probability kernels and falls back for
unsupported sampling modes; KDA packing is TP1/unquantized/BF16 CUDA decode only.

The shared-add result is small and sensitive to drift: its fixed-length closing
control moves +0.910%, versus the observed +0.843% treatment gain. The other
fixed-length drifts are +0.168% routing tail, -0.538% router input, +0.177% sampler,
+0.388% MessagePack and -0.514% KDA packing. The linked rows and
[machine-readable measurements](data/decode-option-results-20261024.json)
retain individual waves, bracketing throughput, drift and three-wave bootstrap
intervals. Three waves do not characterize variation across machines and days.

## Common measurement contract

The panel fixes the step5402 Olmo 3.5 KDA MoE HF checkpoint, requests, sampling,
admission C64, memory fraction 0.7, prefill chunk 8,192 and context 12,288. It uses
BF16, TP1/EP1, full decode graphs, radix caching with extra-buffer Mamba state,
route capture/return and original logprobs. H100 uses the existing default MoE
tiles; the shipped B300 tuning file does not select an H100 configuration.

Three fixed waves generate 1,024 tokens per request; three mixed waves use frozen
RL lengths up to 10,240 tokens. All outputs are retained. Whole-wave throughput
includes admission and drain, excluding startup, warmup, profiling and scoring.
Each option has its own opening and closing controls on the same physical GPU;
different option panels can use different H100s. Profile timings explain changed
work; unprofiled waves supply the throughput rows.

The common Core reference uses `e505356`, BF16 evaluation, natural routing,
torch attention and one full request per document, without EP. Reuse requires
matching checkpoint and input-file hashes. Absolute baseline disagreement is:

| Metric | Common reference value |
|---|---:|
| Mean absolute error | 0.03694615 nats/token |
| Signed mean error | +0.00201359 nats/token |
| p95 / p99 absolute error | 0.13185763 / 0.23219654 nats |
| Maximum absolute error | 3.29934740 nats |
| Ratio outside [0.8, 1.2] | 1.64719976% |

The ratio is exp(Core logprob − serving logprob). “Zero agreement cost” means no
increase over that disagreement; it does not mean serving matches the trainer.
If free generations change, two fresh engines force-score the same frozen tokens
before comparing either against Core. Forced-replay timing is diagnostic only.

## Interactions and limits

Do not add these percentages. They are independent measurements against the same
baseline behavior, not sequential additions to one implementation. Some options
touch the same code or work; others change how much CPU work overlaps the GPU.
The fused router from #10 takes precedence over the routing-tail path, so those
two savings cannot simply stack. A combined branch needs its own full benchmark,
source audit and same-token agreement row.

MessagePack illustrates the distinction between component and whole-engine
speed: local completed-history encoding/decoding improves by roughly 1.2–3.9×
depending on message size/batching, while whole-engine throughput improves much
less. Tiny incremental messages are approximately tied. Its observed benefit
preserves returned routes; disabling route return was an attribution control
that removes routing-replay capability and is not an equivalent option.

The original #10 fused-top-k result belongs to the earlier panel and reference;
see [its documented tradeoff](core-compat.md#optional-fused-router-top-k). The
incoming [PR #21](https://github.com/allenai/olmo-sglang/pull/21) q/k rounding
result also has a separate older reference. Its common-panel measurement is in
progress. That PR changes source arithmetic unconditionally; the experiment's
arm label is not a production feature switch, and the PR has not been adopted.

Admission/concurrency tuning, MILES lag/drain behavior, real-workload RL quality,
B300 qualification and final production selection remain later work. The present
phase measures the inference building blocks without changing those settings.
