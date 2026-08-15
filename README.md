# olmo-sglang

Installable, out-of-tree native SGLang model support for
`Olmo3MoeForCausalLM`. Applications can call `olmo_sglang.register()` before
constructing an embedded SGLang engine. Server processes discover the same
implementation through SGLang's external-model registry, so the SGLang checkout
does not need to be patched.

## Current milestone

- Native full and sliding-window attention.
- Headwise Q/K normalization and optional elementwise attention output gate.
- Peri-LN residual ordering.
- Dense SwiGLU and routed MoE layers.
- OLMo-specific expert-weight normalization, restored top-k scale, and optional
  original-top-k correction.
- Optional latent down/up projections and a full-width shared expert.
- HF-layout weight loading into SGLang's fused QKV, SwiGLU, and MoE tensors.
- Token-in/token-out HF versus SGLang greedy inference harness.

The production model's `linear_attention` layers are deliberately rejected for
now. SGLang already has recurrent KDA infrastructure, but the target Olmo model
uses `beta = 2 * sigmoid(beta_proj(x))` when negative eigenvalues are enabled.
Current SGLang decode kernels apply the sigmoid internally and assume the usual
`[0, 1]` beta range. Treating that as ordinary Kimi KDA would silently change
the model.

## Install

The package expects an existing SGLang runtime and deliberately does not install
or pin a second copy. To use the local SGLang checkout in a fresh environment:

```bash
git clone git@github.com:allenai/olmo-sglang.git ~/proj/olmo-sglang
cd ~/proj/olmo-sglang
uv venv --python 3.12
uv pip install --python .venv/bin/python -e ~/proj/sglang/python
uv pip install --python .venv/bin/python -e .
```

After installation, any Python process in that environment can import the
package without setting `PYTHONPATH`:

```python
import olmo_sglang

print(olmo_sglang.MODEL_ARCHITECTURE)
olmo_sglang.register()
```

`register()` registers the model in the current process and sets
`SGLANG_EXTERNAL_MODEL_PACKAGE=olmo_sglang.models` for SGLang worker processes.
It raises an error rather than replacing a different configured external-model
package.

## Embedded SGLang engine

For a checkpoint with full/sliding attention:

```python
from olmo_sglang import register

register()

import sglang as sgl

engine = sgl.Engine(
    model_path="/path/to/olmo-hf-checkpoint",
    trust_remote_code=True,
)
try:
    result = engine.generate(
        prompt="The future of language modeling is",
        sampling_params={"temperature": 0, "max_new_tokens": 32},
    )
    print(result["text"])
finally:
    engine.shutdown()
```

If the checkpoint does not contain tokenizer files, construct the engine with
`skip_tokenizer_init=True` and pass `input_ids` instead of `prompt`.

## Start an SGLang server

The SGLang CLI starts a persistent HTTP and OpenAI-compatible server. The
environment variable is necessary because the CLI does not import
`olmo_sglang` before it initializes the model registry:

```bash
export MODEL_PATH=/path/to/olmo-hf-checkpoint

SGLANG_EXTERNAL_MODEL_PACKAGE=olmo_sglang.models \
  .venv/bin/sglang serve \
  --model-path "$MODEL_PATH" \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 30000
```

Query a checkpoint that has a tokenizer through the OpenAI-compatible API:

```bash
curl http://127.0.0.1:30000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "/path/to/olmo-hf-checkpoint",
    "prompt": "The future of language modeling is",
    "temperature": 0,
    "max_tokens": 32
  }'
```

Press `Ctrl-C` in the server terminal to shut it down.

## Tiny local inference smoke test

The local attention-only reference checkpoint is small enough for a fast GPU
smoke test and carries its own Transformers implementation:

```bash
.venv/bin/olmo-sglang-smoke \
  --model "$HOME/proj/olmo3-megatron-spike/reference/base" \
  --input-ids 2 3 4 5 \
  --max-new-tokens 4
```

The same tiny checkpoint can be served with token IDs. Its 8-wide attention
heads require diagnostic backends rather than SGLang's production FlashInfer
path:

```bash
SGLANG_EXTERNAL_MODEL_PACKAGE=olmo_sglang.models \
  .venv/bin/sglang serve \
  --model-path "$HOME/proj/olmo3-megatron-spike/reference/base" \
  --trust-remote-code \
  --skip-tokenizer-init \
  --attention-backend torch_native \
  --disable-radix-cache \
  --cuda-graph-backend-decode disabled \
  --cuda-graph-backend-prefill disabled \
  --context-length 32 \
  --max-total-tokens 64 \
  --mem-fraction-static 0.15 \
  --host 127.0.0.1 \
  --port 30000
```

Send a native generation request from another terminal:

```bash
curl http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "input_ids": [2, 3, 4, 5],
    "sampling_params": {"temperature": 0, "max_new_tokens": 4}
  }'
```

The harness uses matched FP16 precision, torch-native attention, and no radix
cache because the tiny checkpoint's 8-wide attention heads are below the shapes
supported by the optimized FlashInfer kernel. It requires first-token agreement
and prints both output sequences. Use `--require-token-parity` to make any later
greedy-token divergence fatal.

The current tiny-checkpoint result is HF `[37, 18, 12, 12]` versus SGLang
`[37, 12, 37, 12]`. The first token agrees. Layer probes localized the later
divergence to fused-MoE rounding: attention through the first sparse layer has
cosine similarity of at least 0.99997, while the raw MoE output differs by about
`3.1e-4` mean absolute error. This synthetic checkpoint initializes that output
near zero and immediately RMS-normalizes it, amplifying the small kernel-level
difference. Treat this as a working inference smoke test, not strict numerical
parity.

Strict validation for a production checkpoint should compare prompt logits and
router selections at its serving dtype before enabling RL rollouts.

## Tests

```bash
PYTHONPATH=src python -m pytest tests -q
ruff format --check src tests
ruff check src tests
```
