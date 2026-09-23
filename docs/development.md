# Development

## Repository layout

```text
examples/                 minimal user-facing inference examples
tools/                    fixture generators and developer diagnostics
src/olmo_sglang/models/   SGLang model implementation
src/olmo_sglang/kda/      KDA backend, layer, and kernels
src/olmo_sglang/validation/ packaged validation CLIs and reference model
tests/                    portable and CUDA-gated tests
```

Files under `examples/` should demonstrate an ordinary user workflow. Synthetic
checkpoint construction and implementation diagnostics belong under `tools/`.
The source distribution includes both directories and the detailed docs through
`MANIFEST.in`; the runtime wheel contains only the importable package and CLI
entry points.

Keep generated measurement JSON, logs, and activation dumps with run artifacts
outside the source tree or under Git-ignored `runs/`. Commit the checks that
produce them and concise summaries of the findings, source/image identities,
commands, thresholds, and limitations. Numerical fixtures belong in the tree
when a regression test consumes them. Historical reports can be linked at an
immutable Git commit rather than carried forward in the maintained tree.

## Local checks

GitHub Actions runs Ruff and the portable CPU suite on pull requests, pushes to
`main`, and manual dispatches, using Ubuntu and Python 3.12. Reproduce the CI
environment in a separate virtualenv (the CPU requirements replace PyTorch):

```bash
python3.12 -m venv .venv-ci
.venv-ci/bin/python -m pip install -r requirements/cpu.txt -r requirements/lint.txt -e .
.venv-ci/bin/python -m pytest tests --cpu-only -q
.venv-ci/bin/ruff format --check src tests tools examples
.venv-ci/bin/ruff check src tests tools examples
```

Direct test and lint dependencies are pinned in `requirements/`; update those
pins together with a passing CPU run. The runtime package deliberately has no
mandatory dependencies, so installing it does not replace a serving stack.

`--cpu-only` excludes six modules that import SGLang internals during collection:
EP diagnostics, KDA backend, KDA radix cache, KDA tensor parallelism, and
speculative KDA kernels, plus model weight targets. The remaining suite covers
configuration, registration, activations, attention, routing, reference models,
compatibility metadata, serving diagnostics, and validation
harness control flow without SGLang or CUDA installed. New test modules are
included by default; runtime-dependent modules must be added to the explicit
list in `tests/conftest.py`.

Run the full suite in the compatible SGLang runtime:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests -q
```

Run formatting and lint checks:

```bash
.venv/bin/ruff format --check src tests tools examples
.venv/bin/ruff check src tests tools examples
```

Run the FLA recurrence check on a CUDA machine:

```bash
PYTHONPATH=src .venv/bin/python tools/check_kda_recurrence.py
```

CUDA tests are marked and skip automatically when CUDA is unavailable, but
collecting runtime tests still requires SGLang and its dependencies. The full
suite additionally validates KDA state shape and CPU recurrence, cache lifecycle,
and runtime integration. GitHub Actions does not yet run this suite or the
whole-engine GPU validation commands.

## Change discipline

Changes to SGLang imports, KDA state shape, model weight mapping, or worker
registration should be tested against the exact runtime described in
[compatibility](compatibility.md). A passing CPU suite does not establish GPU
kernel, whole-engine, tensor-parallel, or CUDA-graph correctness.

The `models` package path and `olmo_sglang.register()` are public integration
contracts. Preserve `olmo_sglang.kda_backend` until the OLMo-MILES preflight has
migrated to the canonical `olmo_sglang.kda.backend` path.
