#!/bin/bash
# Benchmark the fused MoE kernel at every batch size with the default config and with the tuned file.
set -euo pipefail
export PYTHONUNBUFFERED=1
echo "== GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
WORK=/tmp/olmo-moe-bench
rm -rf "$WORK" && mkdir -p "$WORK" && cp -r /tune/benchmark/kernels/fused_moe_triton/. "$WORK/" && cd "$WORK"
cat > hero_config.json <<'JSON'
{"architectures": ["Olmo3MoeForCausalLM"], "hidden_size": 1024, "n_routed_experts": 512, "num_experts_per_tok": 16, "moe_intermediate_size": 1024, "latent_moe_dim": 512, "torch_dtype": "bfloat16"}
JSON
export OLMO_MOE_TUNE_CONFIG=$WORK/hero_config.json
SIZES="1 2 4 8 16 24 32 48 64 96 128 256 512 1024 2048 4096 8192"
echo "== DEFAULT config"
python tuning_fused_moe_triton.py --model dummy --tp-size 1 --ep-size 1 --dtype auto --batch-sizes $SIZES 2>&1 | grep -E "^Batch size|^Kernel time" | paste - - | sed 's/^/default  /'
echo "== TUNED config from /moe_configs"
SGLANG_MOE_CONFIG_DIR=/moe_configs python tuning_fused_moe_triton.py --model dummy --tp-size 1 --ep-size 1 --dtype auto --batch-sizes $SIZES 2>&1 | grep -E "^Batch size|^Kernel time|Using MoE kernel config" | paste - - | sed 's/^/tuned    /'
