# olmo-sglang

`olmo-sglang` adds native `Olmo3MoeForCausalLM` support to
[SGLang](https://github.com/sgl-project/sglang) without patching the SGLang
checkout. It can be used directly for inference or as the rollout runtime in
[`olmo-miles`](https://github.com/allenai/olmo-miles).

> [!IMPORTANT]
> This package is an OLMo extension for an existing, compatible SGLang runtime;
> it is not a standalone inference framework. The implementation is usable for
> experimentation, but several production correctness and performance gates
> remain open. See [status](docs/status.md) and
> [compatibility](docs/compatibility.md).

The frozen compatibility target is the official SGLang `sglang-miles` commit
`3bbb2812e2ca1defdee76f6ec09dbb2456c21c69`; this package does not require an
SGLang fork. See [compatibility](docs/compatibility.md) for the complete runtime
contract.

## Supported model features

- Full and sliding-window attention.
- OLMo KDA prefill and cached decode through FLA 0.5.2.
- Dense SwiGLU and routed MoE layers, including latent and shared experts.
- HF-layout checkpoint loading into SGLang's fused tensors.
- Tensor-parallel KDA execution with documented head-count constraints.
- Experimental branch-capable KDA radix caching and speculative verification.

FLA is optional for attention-only checkpoints. KDA serving is GPU-oriented.

## Quick start

Create an environment containing this package and a compatible SGLang checkout:

```bash
git clone https://github.com/allenai/olmo-sglang.git
cd olmo-sglang
uv venv --python 3.12
uv pip install --python .venv/bin/python -e /path/to/sglang/python
uv pip install --python .venv/bin/python -e .
```

For checkpoints containing `linear_attention` layers, also install the pinned
FLA release:

```bash
uv pip install --python .venv/bin/python flash-linear-attention==0.5.2
```

Start an OpenAI-compatible SGLang server:

```bash
SGLANG_EXTERNAL_MODEL_PACKAGE=olmo_sglang.models \
  .venv/bin/sglang serve \
  --model-path /path/to/olmo-hf-checkpoint \
  --trust-remote-code \
  --tp-size 1
```

Or run the barebones embedded example:

```bash
.venv/bin/python examples/embedded_inference.py \
  /path/to/olmo-hf-checkpoint \
  --prompt "The future of language modeling is"
```

Applications embedding SGLang directly must call `register()` before creating
the engine:

```python
from olmo_sglang import register

register()
```

See [getting started](docs/getting-started.md) for complete server, tokenizer,
and embedded-engine examples.

## Documentation

| Document | Contents |
|---|---|
| [Getting started](docs/getting-started.md) | Installation, registration, embedded inference, and serving |
| [Compatibility](docs/compatibility.md) | Tested runtime, checkpoint contract, requirements, and constraints |
| [Design](docs/design.md) | Package boundaries, model integration, KDA state, radix caching, and speculation |
| [Validation](docs/validation.md) | Tiny fixtures, parity tools, smoke tests, and developer checks |
| [Status](docs/status.md) | Current capabilities, remaining gates, and production caveats |
| [Development](docs/development.md) | Repository layout, test commands, and contribution guidance |

## Development check

For the CPU-only GitHub Actions environment and dependency installation, see
[local checks](docs/development.md#local-checks). With a compatible SGLang runtime,
run the full suite:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests -q
.venv/bin/ruff format --check src tests tools examples
.venv/bin/ruff check src tests tools examples
```

Licensed under the [Apache License 2.0](LICENSE).
