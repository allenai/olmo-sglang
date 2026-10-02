# olmo-sglang

> **Development status:** This repository supports OLMo model architectures under
> active development. No compatible model checkpoints are publicly available yet.
> It is intended for development and collaboration; synthetic fixtures are
> provided for testing.

`olmo-sglang` adds native `Olmo3MoeForCausalLM` support to
[SGLang](https://github.com/sgl-project/sglang) without patching the SGLang
checkout. It can be used directly for inference or as a rollout runtime in
training applications.

Install the extension into the [tested SGLang runtime](docs/compatibility.md).
The pinned source revision is `3145136dcd1238754e0ea2b2ffd546532119c71c`.
See [validation status](docs/status.md) for measured coverage and known numerical
limits, including measured full-checkpoint probability differences across
execution paths.

## Supported model features

- Full and sliding-window attention.
- OLMo KDA prefill through FLA 0.5.2 and packed Triton cached decode.
- Headwise Q/K normalization, per-head gains, scalable softmax, and elementwise
  attention output gates.
- Dense SwiGLU and routed MoE layers, including latent and shared experts.
- HF-layout checkpoint loading into SGLang's fused tensors.
- Tensor-parallel KDA execution with documented head-count constraints.
- Branch-capable KDA radix caching and opt-in speculative verification, with
  [feature-specific validation](docs/status.md).

FLA is optional for attention-only checkpoints. KDA serving is GPU-oriented.

## Development setup

Use Python 3.12 and a GPU environment matching the
[tested runtime](docs/compatibility.md#integration-baseline). For CPU-only
development, follow the [local checks](docs/development.md#local-checks).
Install this package alongside the pinned SGLang checkout:

```bash
git clone https://github.com/sgl-project/sglang.git
git -C sglang checkout 3145136dcd1238754e0ea2b2ffd546532119c71c
git clone https://github.com/allenai/olmo-sglang.git
cd olmo-sglang
uv venv --python 3.12
uv pip install --python .venv/bin/python -e ../sglang/python
uv pip install --python .venv/bin/python -e .
```

For checkpoints containing `linear_attention` layers, also install the pinned
FLA release:

```bash
uv pip install --python .venv/bin/python flash-linear-attention==0.5.2
```

To exercise the integration without model weights, use the
[synthetic checkpoint walkthrough](docs/validation.md#deterministic-kda-fixture).
These fixtures test execution and do not produce meaningful language.

If you have a compatible development checkpoint, start an OpenAI-compatible
SGLang server:

```bash
SGLANG_EXTERNAL_MODEL_PACKAGE=olmo_sglang.models \
  .venv/bin/sglang serve \
  --model-path /path/to/olmo-hf-checkpoint \
  --trust-remote-code \
  --attention-backend triton \
  --sampling-backend pytorch \
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
| [Status](docs/status.md) | Validation evidence, known numerical limits, and remaining work |
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

Licensed under the [Apache License 2.0](LICENSE), with MIT-licensed adaptations
listed in [third-party notices](THIRD_PARTY_NOTICES.md).

[Core-compatible execution modes](docs/core-compat.md) select fused rounding by
default for supported model configurations with TP1/EP1 BF16 serving. Explicit
opt-out, tensor controls and the slower full numerical reference remain available.

## Contact and contributions

Please use [GitHub issues](https://github.com/allenai/olmo-sglang/issues) for
public questions and bug reports. Pull requests with fixes, improvements, and
documentation updates are welcome; please contribute a pull request when you can.
For other inquiries, contact [robertb@allenai.org](mailto:robertb@allenai.org).
