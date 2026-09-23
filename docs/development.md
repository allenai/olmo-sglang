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

Run the portable suite:

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

CUDA tests are marked and skip automatically when CUDA is unavailable. The
portable suite validates configuration, registration, routing, reference-model
behavior, KDA state shape and CPU recurrence, cache lifecycle, and validation
harness control flow.

## Change discipline

Changes to SGLang imports, KDA state shape, model weight mapping, or worker
registration should be tested against the exact runtime described in
[compatibility](compatibility.md). A passing CPU suite does not establish GPU
kernel, whole-engine, tensor-parallel, or CUDA-graph correctness.

The `models` package path and `olmo_sglang.register()` are public integration
contracts. Preserve `olmo_sglang.kda_backend` until the OLMo-MILES preflight has
migrated to the canonical `olmo_sglang.kda.backend` path.
