#!/bin/bash
# Tune SGLang's fused-MoE Triton kernel for the hero (12.5B hybrid latent-KDA MoE) expert shape.
set -euo pipefail
export PYTHONUNBUFFERED=1
echo "== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
python -c "import sglang, torch, triton; print('sglang', sglang.__version__, 'torch', torch.__version__, 'triton', triton.__version__)"
python -c "import ray" 2>/dev/null || { echo "ray missing; installing"; uv pip install --system ray || pip install ray; }

WORK=/tmp/olmo-moe-tune
rm -rf "$WORK" && mkdir -p "$WORK" && cp -r /tune/benchmark/kernels/fused_moe_triton/. "$WORK/" && cd "$WORK"

# Resolve the hero HF config: prefer the C11 checkpoint lineage on weka, fall back to the
# qualified rounding-profile geometry (identical expert shape) if no HF config is found.
python - <<'PY'
import json, os, pathlib
candidates = [
    "/weka/olmo-3p5-post-checkpoints/abhishekr/hero-sft-4t/runs/c11-4t-s34521",
    "/weka/olmo-3p5-post-checkpoints/abhishekr/hero-sft-4t/runs/c11-4t-s34522",
    "/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/hero-sft-4t/emo/step5961",
]
found = None
for root in candidates:
    p = pathlib.Path(root)
    if not p.exists():
        print(f"missing: {root}")
        continue
    for cfg in sorted(p.rglob("config.json")):
        if len(cfg.relative_to(p).parts) > 4:
            continue
        try:
            data = json.load(open(cfg))
        except Exception:
            continue
        if "Olmo3MoeForCausalLM" in (data.get("architectures") or []):
            found = (cfg, data)
            break
    if found:
        break
if found:
    cfg, data = found
    print(f"using HF config from {cfg}")
else:
    print("WARNING: no HF Olmo3Moe config found on weka; using the qualified hero rounding profile geometry")
    data = {"architectures": ["Olmo3MoeForCausalLM"], "hidden_size": 1024, "n_routed_experts": 512,
            "num_experts_per_tok": 16, "moe_intermediate_size": 1024, "latent_moe_dim": 512,
            "torch_dtype": "bfloat16"}
keys = ["hidden_size", "n_routed_experts", "num_experts_per_tok", "moe_intermediate_size", "latent_moe_dim", "num_hidden_layers"]
print("geometry:", {k: data.get(k) for k in keys})
expected = {"n_routed_experts": 512, "num_experts_per_tok": 16, "moe_intermediate_size": 1024, "latent_moe_dim": 512}
mismatch = {k: data.get(k) for k, v in expected.items() if data.get(k) != v}
if mismatch:
    print(f"WARNING: expert geometry differs from the hero profile: {mismatch}")
json.dump(data, open("/tmp/olmo-moe-tune/hero_config.json", "w"))
PY
export OLMO_MOE_TUNE_CONFIG=/tmp/olmo-moe-tune/hero_config.json

START=$(date +%s)
python tuning_fused_moe_triton.py --model dummy --tp-size 1 --ep-size 1 --dtype auto --tune --search-space-file /tune/search_space.json --batch-sizes 1 2 4 8 16 24 32 48 64 96 128 256 512 1024 2048 4096 8192 2>&1 | tee "$WORK/tune.log"
echo "== tuning wall time: $(( $(date +%s) - START )) s"

mkdir -p /output
cp -v "$WORK"/E=*.json /output/ 
cp -v "$WORK/tune.log" /output/
cp -v /tmp/olmo-moe-tune/hero_config.json /output/
echo "== tuned config:"
cat /output/E=*.json
