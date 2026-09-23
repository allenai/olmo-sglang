# Numerical findings from the September 10, 2026 investigation

The original reports and Core/HF diagnostic tools are retained at
[commit 02ccb5d](https://github.com/allenai/olmo-sglang/tree/02ccb5dcf641cbabc9b78a5bc65dacf8690707a7).
They describe the model, backend, and source combinations tested then.
Current serving checks are documented in [attention](attention.md).

## Full-checkpoint probability mismatch

The step-75500 checkpoint failed its 0.1 absolute log-probability threshold.
Eight greedy predictions matched, but the maximum checked SGLang top-token
error was 0.3118 with unchunked prefill and 0.5619 with chunked prefill plus
decode graphs. Core/HF full-vocabulary error reached 0.5473 (mean 0.08255).
Greedy-token agreement did not establish probability parity. The tiny-model
checks do not resolve this full-checkpoint mismatch.

## Recovered full-checkpoint controls (audited September 23)

The retained Beaker results go beyond the original failed comparison. The
[compact evidence record](measurements/readiness-limits-20260923.json) includes
experiment IDs and report hashes.

- The [exhaustive conversion audit](https://beaker.org/ex/01M26YP78T0JJ545H914AEYDDZ)
  matched all 23,441 tensors after the declared export cast, both native-to-HF
  and HF-to-Core-to-HF. Checkpoint conversion itself passed.
- The [matched Core/HF control](https://beaker.org/ex/01M274P8Z2D9A6FQ0BSHGBX6NK)
  used SDPA and `OLMO_HF_MOE_CORE_REFERENCE=1`, which matches Core's expert
  permutation, grouped GEMMs, SwiGLU layout, and accumulation. Full-vocabulary
  logits and log probabilities were identical on both initial prefixes
  (16 and 81 tokens). This establishes that the earlier Core/HF drift can be
  removed by aligning execution paths for those inputs. It does not establish
  equality with the ordinary HF or SGLang kernels.
- The [four-mode serving matrix](https://beaker.org/ex/01M27118R2ER9QG0PYP3ARCHBE)
  matched all eight greedy predictions in every mode. Switching CUDA graphs
  off/on changed none of the checked forced-prefix probabilities. Switching
  prefill chunks from 128 to 32 changed common top-token log probabilities by
  up to 0.571168. The strict HF comparison still failed: maximum top-token
  errors were 0.323586 and 0.385278 respectively, against 0.1. Chosen decode
  token errors stayed below 0.09; maximum absolute probability difference
  among checked top tokens was 0.015821.

These results classify the remaining issue as **probability sensitivity to
execution paths**, rather than an untested checkpoint conversion or a proven
CUDA-graph defect. The matrix does not isolate which chunk-dependent operation
causes the drift. If close trainer/serving probabilities are required, the next
useful investigation is matched-prefix KDA/MoE output and routing comparison
across chunk sizes. The existing tolerance has not been relaxed. Ordinary
generation has greedy-parity evidence; this small sample is not a quality or
long-context evaluation.

## Current-source checkpoint audit (September 23)

The [fresh TP1 B300 audit](https://beaker.org/ex/01M37SJ6677TESB1CDWXPG42WR)
used extension `156549d` with the accompanying validation-tool changes and
SGLang source `3145136dcd1238754e0ea2b2ffd546532119c71c`. It loaded the same
step-75500 checkpoint without missing, unexpected, or mismatched weights.
Both reference variants used BF16 weights and recurrent HF prefill. Generation
was bounded to four tokens for each of the same 16/81-token synthetic prefixes.

| Reference | Prefill chunk | Max checked top-token logprob error | Max chosen decode-token error |
|---|---:|---:|---:|
| Ordinary HF, eager attention | 128 | 0.311782 | 0.089736 |
| Ordinary HF, eager attention | 32 | 0.561879 | 0.086706 |
| Core-matched HF experts, SDPA | 128 | 0.268488 | 0.149366 |
| Core-matched HF experts, SDPA | 32 | 0.493953 | 0.100297 |

Every row ran with decode graphs both off and on. All eight greedy tokens
matched both references in all modes. Graphs changed neither checked
forced-prefix nor chosen decode-token probabilities. Changing chunk size
produced a maximum common-top-token logprob difference of 0.571168 in both
reference runs. Both full probability checks failed the unchanged 0.1 threshold.
The largest checked absolute probability difference from ordinary HF was
0.034108; this is a top-20-token check, not a full-distribution distance.

Thus the probability difference persists with current code and with a
Core-matched reference. It is not explained away solely by ordinary HF's expert
layout. This audit supports short greedy inference and graph execution, while
strict probability equivalence remains a distinct numerical investigation.
It does not measure task quality, long contexts, or exact router agreement.
The [evidence record](measurements/readiness-limits-20260923.json) includes
source-bundle identity, reference source hashes, per-mode results, and report
hashes. The retained image runtime lock describes the base image; the staged
extension and SGLang sources override its original versions.

## SwiGLU rounding

On a tiny latent-MoE model, routing IDs, FP32 combine weights, latent down
projection, and the controlled first grouped-GEMM inputs/outputs matched
exactly. Native Core no-grad execution first differed at SwiGLU: its kernel
evaluated in FP32 and cast once, whereas the controlled HF path rounded
`silu(gate)` to BF16 before multiplying by `up`. Core's grad-enabled path used
the latter expression and matched controlled HF at every captured branch,
including the final sum, for the saved 16- and 81-token inputs.

Native no-grad activation matched the FP32 single-cast control exactly. Its
relative L2 difference from HF eager activation was 0.003047 and 0.002652.
Ordinary HF still differed from grad-enabled Core at the combined output
(relative L2 0.000772 and 0.001232), so the activation split did not explain all
observed discrepancies. These were forward-only operator checks, with attention
bypassed and auxiliary losses zero; they did not qualify training or the full
checkpoint.
