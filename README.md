# olmo-sglang

Installable, out-of-tree native SGLang model support for
`Olmo3MoeForCausalLM`. Applications can call `olmo_sglang.register()` before
constructing an embedded SGLang engine. Server processes discover the same
implementation through SGLang's external-model registry, so the SGLang checkout
does not need to be patched.

## Current milestone

- Native full and sliding-window attention.
- Native OLMo KDA prefill and cached decode through FLA 0.5.2.
- Exact negative-eigenvalue semantics: raw beta logits use `2 * sigmoid(beta)`.
- Unequal K/V head widths in the recurrent-state cache.
- A package-local FLA 0.5.2 compatibility shim for Triton 3.6 and newer.
- Headwise Q/K normalization and optional elementwise attention output gate.
- Peri-LN residual ordering.
- Dense SwiGLU and routed MoE layers.
- OLMo-specific expert-weight normalization, restored top-k scale, and optional
  original-top-k correction.
- Optional latent down/up projections and a full-width shared expert.
- HF-layout weight loading into SGLang's fused QKV, SwiGLU, and MoE tensors.
- Token-in/token-out HF versus SGLang greedy inference harness.

The KDA implementation is a correctness-first serving path. It reuses SGLang's
hybrid scheduler, convolution cache, and recurrent-state pool, but calls FLA
0.5.2 for both prefill and decode so the model's beta semantics stay exact.
Current limits are tensor parallel size 1, radix cache disabled, and no
speculative target verification. This is suitable for parity work and initial
RL rollouts, not yet the final optimized serving kernel.

FLA is optional; attention-only checkpoints do not import it.

## Roadmap and TODOs

These items are intentionally explicit so a working correctness path is not
confused with complete production serving support.

### Correctness gate for MILES and Megatron Bridge

- [ ] Load a real Megatron-Bridge-exported production checkpoint, with no
  synthetic or attention-only substitutions.
- [ ] Compare prompt logits, greedy tokens, and MoE router selections against
  the reference implementation at the intended serving dtype.
- [ ] Exercise short multi-request rollouts through the same HTTP or embedded
  interface that MILES will use, including stop conditions and request
  cancellation.
- [ ] Validate the RL weight-refresh workflow: export or transfer a new policy
  checkpoint, reload it in SGLang, and confirm that generation changes without
  stale KDA state crossing policy versions.

### KDA serving features

- [ ] **Radix cache:** snapshot and restore both the convolution window and
  recurrent KDA matrix at prefix-tree branch points; validate reuse, branching,
  eviction, and request isolation.
- [ ] **Speculative decoding:** implement KDA target verification with
  per-proposal intermediate states and commit only the accepted prefix,
  including rollback and tree-branch behavior.
- [ ] **Tensor parallelism greater than one:** shard the KDA projections,
  unequal-width K/V heads, convolution windows, and recurrent state correctly;
  add TP=1 versus TP=2 numerical comparisons.
- [ ] Validate batched and long-context KDA execution, request retraction, and
  scheduler edge cases under sustained concurrent load.

### Hardening and performance

- [ ] Replace the correctness-first FLA chunk call for one-token decode with an
  optimized OLMo-semantics kernel, then re-enable and validate CUDA graphs.
- [ ] Replace or upstream the narrow FLA 0.5.2/Triton 3.6+ source shim.
- [ ] Add BF16 and production-dimension coverage, including real sparse-MoE
  layers rather than only the tiny dense smoke checkpoint.
- [ ] Optionally add a slow PyTorch CPU reference recurrence for portable unit
  tests; production KDA serving remains GPU-oriented.

Radix caching, speculative decoding, and TP>1 are not prerequisites for the
first MILES integration. The immediate gate is real-checkpoint numerical
correctness using TP=1, ordinary autoregressive decoding, and radix caching
disabled.

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

Install FLA when the checkpoint contains `linear_attention` layers:

```bash
uv pip install --python .venv/bin/python flash-linear-attention==0.5.2
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
    tp_size=1,
    disable_radix_cache=True,
    cuda_graph_backend_decode="disabled",
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
  --tp-size 1 \
  --disable-radix-cache \
  --cuda-graph-backend-decode disabled \
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

## Exercise KDA locally

The repository includes a deterministic two-layer checkpoint with one KDA
layer and one full-attention layer. It has no tokenizer and is intended only
for fast GPU smoke tests:

```bash
PYTHONPATH=src .venv/bin/python examples/create_tiny_kda_checkpoint.py \
  /tmp/olmo-sglang-tiny-kda

PYTHONPATH=src .venv/bin/python -m olmo_sglang.infer \
  --model /tmp/olmo-sglang-tiny-kda \
  --skip-hf \
  --input-ids 2 3 4 5 \
  --max-new-tokens 4
```

The deterministic result on an RTX 4090 is:

```text
input_ids=[2, 3, 4, 5]
sglang_output_ids=[53, 10, 45, 39]
```

The recurrence harness verifies that prompt prefill followed by cached
single-token decode matches one uninterrupted FLA sequence:

```bash
PYTHONPATH=src .venv/bin/python examples/check_kda_recurrence.py
```

It also confirms that enabling negative eigenvalues produces a nonzero result
delta from ordinary KDA, so the test would catch accidentally dropping the
`2 * sigmoid(beta)` behavior.

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
ruff format --check src tests examples
ruff check src tests examples
PYTHONPATH=src python examples/check_kda_recurrence.py
```
