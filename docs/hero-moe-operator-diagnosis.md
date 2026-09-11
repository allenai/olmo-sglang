# Isolated hero latent-MoE operator diagnosis

`tools/diagnose_hero_moe.py` loads only one checkpoint layer and uses the frozen HF activations from `diagnose_hero_core.py`. It bypasses attention at the saved attention-residual boundary, then executes the real native no-EP MoE path. It compares router IDs/FP32 combine weights, latent projections, shared output and the final sum against both ordinary HF and the existing explicitly selected `OLMO_HF_MOE_CORE_REFERENCE` path. The controlled HF path additionally exposes the same two grouped-GEMM boundaries, making row order and packed projection layout directly comparable.

```bash
python tools/diagnose_hero_moe.py \
  --model /weka/olmo-3p5-checkpoints/scratch/hero-hf-20260909/non-emo/step75500/hf \
  --activations /reference/hf-activations.pt \
  --reference-report /reference/diagnosis.json \
  --layer 1 --output /output/moe-operators.json
```

Use the matching frozen-activation dataset, including its HF source hashes. The tool reads only the selected layer from safetensors shards, loads both representations strictly and records source/activation hashes. It compares actual no-grad and grad-enabled single-block forward calls, with the module in eval mode and no backward or optimizer. Auxiliary losses are zero; attention is bypassed, and the input boundary is explicit. This is an operator diagnostic, not a replacement qualification gate or a full training test.

The tiny local RTX4090 run identified a specific arithmetic split. Router IDs, combine weights, latent down projection, and the controlled first grouped-GEMM inputs/outputs matched exactly. Native no-grad execution first differed at SwiGLU: its valid-prefix kernel evaluates in FP32 and casts once, whereas HF's controlled path rounds `silu(gate)` to BF16 before multiplying by `up`. Core's grad-enabled path uses that latter expression. For both saved 16/81-token inputs, the **actual grad-enabled Core MoE forward matched controlled HF exactly at every captured branch and both grouped GEMMs**, including the final sum. Native no-grad activation matched the explicit FP32 single-cast control exactly; its relative L2 difference from HF eager activation was 0.003047 and 0.002652.

Ordinary HF retained a separate difference from grad-enabled Core (combined-output relative L2 0.000772 and 0.001232), so this result does not explain every ordinary-HF discrepancy. Exact tiny evidence is in `measurements/hero-tiny-moe-operators-20260910.json`. Three focused tests check operator capture without arithmetic changes, cleanup, exact integer routing comparisons and grouped-stage coverage. The full step-75500 operator run remains pending. No runtime arithmetic or acceptance threshold changed.
