# Getting started

`olmo-sglang` is an out-of-tree model implementation loaded by SGLang. It is
independent of `olmo-miles`: applications can embed an engine or start SGLang's
HTTP server directly. It does not install SGLang, choose a GPU runtime, or
convert a training checkpoint into Hugging Face layout.

## Install

Start with a compatible SGLang source checkout. The exact runtime tested by the
OLMo integration is recorded in [compatibility](compatibility.md).

```bash
git clone https://github.com/allenai/olmo-sglang.git
cd olmo-sglang
uv venv --python 3.12
uv pip install --python .venv/bin/python -e /path/to/sglang/python
uv pip install --python .venv/bin/python -e .
```

KDA models also require FLA:

```bash
uv pip install --python .venv/bin/python flash-linear-attention==0.5.2
```

Attention-only checkpoints do not import FLA. The package intentionally leaves
SGLang and GPU-library resolution to the containing runtime because those
versions must be selected together.

## Embedded engine

Call `register()` before constructing an engine. Registration installs the
model in the current SGLang registry and sets
`SGLANG_EXTERNAL_MODEL_PACKAGE=olmo_sglang.models` for worker processes.

```python
from olmo_sglang import register

register()

import sglang as sgl

engine = sgl.Engine(
    model_path="/path/to/olmo-hf-checkpoint",
    trust_remote_code=True,
    attention_backend="triton",
    sampling_backend="pytorch",
    tp_size=1,
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

The same flow is available as a source-checkout example:

```bash
.venv/bin/python examples/embedded_inference.py /path/to/olmo-hf-checkpoint
```

The example selects conservative torch-native attention, disables radix reuse
and CUDA graphs, and caps its token pool at 128 tokens so it can also exercise
the repository's tiny local fixtures without reserving most of a GPU. Remove
those diagnostic settings when selecting production backends and capacity.

If a checkpoint does not include tokenizer files, create the engine with
`skip_tokenizer_init=True` and call `generate(input_ids=[...])` instead of
passing `prompt`.

## HTTP server

The upstream CLI does not import `olmo_sglang` before model discovery, so set
the external-package variable explicitly:

```bash
SGLANG_EXTERNAL_MODEL_PACKAGE=olmo_sglang.models \
  .venv/bin/sglang serve \
  --model-path /path/to/olmo-hf-checkpoint \
  --trust-remote-code \
  --attention-backend triton \
  --sampling-backend pytorch \
  --tp-size 1 \
  --host 127.0.0.1 \
  --port 30000
```

For a checkpoint with a tokenizer, query the OpenAI-compatible endpoint:

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

For a tokenizer-free fixture, add `--skip-tokenizer-init` to the server and use
SGLang's native `/generate` endpoint with `input_ids`.

## Registration conflicts

SGLang currently selects one external model package through
`SGLANG_EXTERNAL_MODEL_PACKAGE`. `register()` accepts an unset value or
`olmo_sglang.models`; it raises rather than silently replacing another external
package.
