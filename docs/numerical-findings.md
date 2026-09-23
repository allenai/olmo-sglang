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
