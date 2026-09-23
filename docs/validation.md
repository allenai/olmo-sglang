# Validation

The repository has two kinds of runnable material:

- `examples/` contains the smallest ordinary user workflow;
- `tools/` and the `olmo_sglang.validation` package contain developer fixtures
  and correctness probes.

Commands below assume an editable install in `.venv`.

## Barebones embedded inference

Run a checkpoint with its tokenizer through an embedded engine:

```bash
.venv/bin/python examples/embedded_inference.py \
  /path/to/olmo-hf-checkpoint \
  --prompt "The future of language modeling is" \
  --max-new-tokens 32
```

This is an inference example, not a parity gate: it registers the external
model, creates one engine with conservative tiny-fixture-compatible backends,
generates greedily, and shuts the engine down. Its 128-token pool is an exercise
limit rather than a serving default.

## Deterministic KDA fixture

Create a two-layer tokenizer-free checkpoint with one KDA and one full-attention
layer:

```bash
PYTHONPATH=src .venv/bin/python tools/create_tiny_kda_checkpoint.py \
  /tmp/olmo-sglang-tiny-kda
```

Run the token-in/token-out inference check:

```bash
.venv/bin/olmo-sglang-smoke \
  --model /tmp/olmo-sglang-tiny-kda \
  --skip-hf \
  --input-ids 2 3 4 5 \
  --max-new-tokens 4
```

The established deterministic output on an RTX 4090 is
`[53, 15, 39, 25]`. Treat this as fixture-specific evidence, not a portable
cross-GPU bitwise guarantee.

Verify FLA prompt prefill plus cached single-token decode against one
uninterrupted recurrence:

```bash
PYTHONPATH=src .venv/bin/python tools/check_kda_recurrence.py
```

The check covers ordinary and negative-eigenvalue beta semantics.

## Radix-cache lifecycle

The packaged radix command compares a cached continuation with a fresh,
cache-disabled engine and reports `cached_tokens`:

```bash
.venv/bin/olmo-sglang-radix-smoke \
  --model /tmp/olmo-sglang-tiny-kda \
  --input-ids 2 3 4 5 \
  --max-new-tokens 4 \
  --mamba-radix-cache-strategy no_buffer
```

The short fixture uses SGLang's endpoint-only `no_buffer` baseline because it
does not cross an intermediate tracking boundary. For native OLMo KDA branch
state, create a longer fixture and use `extra_buffer`:

```bash
PYTHONPATH=src .venv/bin/python tools/create_tiny_parity_checkpoint.py \
  /tmp/olmo-sglang-radix-long \
  --profile kda-dense \
  --max-position-embeddings 512

.venv/bin/olmo-sglang-radix-smoke \
  --model /tmp/olmo-sglang-radix-long \
  --prompt-length 300 \
  --max-new-tokens 4 \
  --mamba-radix-cache-strategy extra_buffer
```

Additional flags exercise repeated simultaneous prompts, mixed chunked
prefill, cancellation, forced or organic scheduler retraction, and idle policy
refresh. Use `olmo-sglang-radix-smoke --help` for the current knobs. These are
stress gates; their low-memory settings are not serving defaults.

## Independent parity loop

The parity fixture includes a tokenizer and an independent PyTorch reference.
The standard profiles isolate attention, KDA, hybrid MoE, and production dimensions:

```bash
export PARITY_ROOT=/tmp/olmo-sglang-parity

for profile in attention-dense kda-dense hybrid-moe production-shape; do
  PYTHONPATH=src .venv/bin/python tools/create_tiny_parity_checkpoint.py \
    "$PARITY_ROOT/$profile" --profile "$profile"
  .venv/bin/olmo-sglang-parity \
    --model "$PARITY_ROOT/$profile" \
    --max-new-tokens 4 \
    --output "$PARITY_ROOT/$profile.json" \
    --require-parity
done
```

The report preserves the rendered prompt, exact input IDs, greedy outputs,
per-step top log probabilities, forced-prefix results, and shape/statistical
summaries at reference boundaries. Forced-prefix results help distinguish a
cached-decode error from a broader model or prefill mismatch.

The `production-shape` fixture uses production hidden, attention, KDA, latent,
and expert widths plus the first five-layer attention pattern. It keeps 32
rather than 512 routed experts so it remains practical on a workstation GPU.

The additional `scaled-attention-hybrid-moe` profile exercises per-head Q/K gains
and scalable softmax. `biased-sliding-hybrid-moe` adds an eight-token sliding
window and nonzero Q/K/V/output/gate projection biases. Both profiles have a CPU
reference. See [attention](attention.md) for their HF comparison, CUDA-graph,
and live weight-update checks.

## Speculative verification

Compare ordinary decoding with breadth-one NGRAM speculation:

```bash
.venv/bin/olmo-sglang-speculative-smoke \
  --model /path/to/olmo-hf-checkpoint \
  --prompt-length 32 \
  --max-new-tokens 16 \
  --context-length 64
```

To require a real two-leaf tree, supply two divergent corpus continuations and
set `--ngram-breadth 2`. SGLang's NGRAM extension also requires `ninja`. The
report includes verify passes, proposals, accepted drafts, acceptance rate, and
average accepted length. Exact ordinary/speculative greedy-token parity is the
gate; random fixture acceptance is not a performance signal.

Both engines explicitly use Triton attention and PyTorch sampling, so the
comparison also supports tiny head dimensions without inheriting FlashInfer
defaults. `--cuda-graph-backend-decode full` exercises captured decoding.

## Tensor parallelism

Compare TP=1 with a sharded engine:

```bash
.venv/bin/olmo-sglang-tp-smoke \
  --model /path/to/olmo-hf-checkpoint \
  --tp-size 2 \
  --prompt-length 32 \
  --max-new-tokens 8 \
  --context-length 512 \
  --mem-fraction-static 0.25
```

The gate requires exact greedy token IDs and reports chosen-token log-probability
differences. It disables radix reuse and CUDA graphs so tensor sharding is the
primary changed variable.

## September 23, 2026 validation audit

These checks ran on one RTX 4090 using Python 3.12.13, PyTorch 2.13.0+cu130,
Triton 3.7.1, FLA 0.5.2, Transformers 5.12.1, and SGLang source
`3145136dcd1238754e0ea2b2ffd546532119c71c`. The matching HF source was Core
`3d35ab326b72d92e06137cc310631d9187d8a2c5`. The extension was based on `8852392`
with the fixes and tests accompanying this audit. Logs and JSON reports are
stored with local run artifacts under `runs/readiness-20260923/`.

| Check | Result |
|---|---|
| Full repository suite, CUDA available | 118 passed |
| Fresh CPU CI environment | 74 passed, 9 runtime/CUDA tests skipped; runtime-only modules excluded by `--cpu-only` |
| Scaled full-attention hybrid MoE, BF16 | All eight HF greedy tokens matched in eager and chunked/graph modes; max checked logprob error 0.02604771 against 0.05 |
| Biased sliding-attention hybrid MoE, BF16 | All eight HF greedy tokens matched in both modes; max checked logprob error 0.01934958 against 0.05 |
| Live Q/K gain and scale updates, both fixtures | All control responses succeeded; outputs changed and matched fresh changed-checkpoint engines with zero logprob difference |
| KDA radix flush/reload, 300-token prompt | Cached tokens `0, 256` before reload and `0, 256` after; greedy tokens unchanged |
| Mixed-length KDA prefill, 64-token chunks | Three cached requests matched the uncached control, with a 256-token tracked prefix |
| Scaled-attention NGRAM, decode graphs | Eight greedy tokens matched ordinary decode; seven verify passes, 21 proposed extras, zero accepted extras |

The sliding-attention check caught a missing model window-size hook and a
pinned-backend convention difference: Triton decode needs the KV token count
including the current token, while each layer's prefill mask uses the exclusive
left window. Gate projection biases were also being discarded during loading.
Both paths now have regression coverage. Failed weight-update control responses
also fail the serving check, rather than merely appearing in its report.

Reproduce the engine checks in the pinned environment, with the matching HF
source in `$HF_SOURCE`. Keep `.venv/bin` on `PATH` for Ninja and compiler helpers.
The fixture directories below must not already exist.

```bash
export PATH="$PWD/.venv/bin:$PATH"
for profile in scaled-attention-hybrid-moe biased-sliding-hybrid-moe; do
  python tools/qualify_serving.py \
    --tiny-dir "/tmp/olmo-audit-$profile" --tiny-profile "$profile" \
    --hf-source "$HF_SOURCE" --recurrent-hf-prefill \
    --require-token-parity --logprob-atol 0.05 --check-live-update \
    --output "runs/readiness/$profile.json"
done
python tools/create_tiny_parity_checkpoint.py /tmp/olmo-audit-kda \
  --profile kda-dense --max-position-embeddings 512
for probe in policy-refresh-probe mixed-chunked-prefill-probe; do
  olmo-sglang-radix-smoke --model /tmp/olmo-audit-kda \
    --prompt-length 300 --max-new-tokens 4 \
    --mamba-radix-cache-strategy extra_buffer --chunked-prefill-size 64 "--$probe"
done
olmo-sglang-speculative-smoke \
  --model /tmp/olmo-audit-scaled-attention-hybrid-moe \
  --prompt-length 32 --max-new-tokens 8 --context-length 256 \
  --cuda-graph-backend-decode full
```

The HF reference explicitly uses recurrent FLA prefill. These are TP1 fixture
checks with Triton attention; they do not clear the historical full-checkpoint
probability mismatch or establish whole-engine TP2 coverage for the newer
gain/scale features. See [status](status.md) for those specific limits.

## Changed weights with radix caches and graphs

The grouped probe creates two tiny BF16 hybrid-MoE fixtures (scaled full
attention and biased sliding attention). In each engine it populates caches,
updates every tensor, checks cold/warm/mixed-length requests, and restores the
original weights. A fresh changed-checkpoint engine runs the same requests.
This exercises KDA projections/convolution/recurrent parameters, attention
gains/scales/biases, dense/shared/routed expert loading, and graph replay in one
run. All generations are capped at eight tokens.

```bash
python tools/qualify_cache_updates.py \
  --tiny-dir /tmp/olmo-cache-updates \
  --output runs/readiness/changed-cache.json
```

The fixture directories must not exist. Use `--eager` to diagnose a graph-specific
failure. On September 23 both profiles passed on the RTX 4090: 72/77 tensors
updated, zero logprob difference from matched fresh engines, exact restoration,
and cache counts `0` cold / `256` warm and for all three mixed requests. Prompts
were 300/268/332 tokens, with 64-token prefill chunks and graph batches 1/2/4.
The [compact evidence](measurements/readiness-limits-20260923.json) records these
results alongside recovered historical full-checkpoint controls. This is
standalone TP1 publication; the caller still owns multi-replica admission.
The follow-up full suite passed 127 tests with CUDA available; the portable CPU
subset passed 83 (nine runtime/CUDA skips). A text-prompt check with SDPA HF
attention matched all eight generated tokens in eager and graph modes, with maximum checked
logprob error 0.026860 against the unchanged 0.05 fixture tolerance.

For bounded real-checkpoint diagnosis, `qualify_serving.py` accepts repeated
`--prompt-text`, `--max-new-tokens` from 1 through 8, and `--hf-attention eager|sdpa`.
Reports record the reference attention backend and `OLMO_HF_MOE_CORE_REFERENCE`
setting so a controlled reference cannot be confused with ordinary HF execution.
The [full-checkpoint follow-up](numerical-findings.md#current-source-checkpoint-audit-september-23)
ran both reference variants on one B300: greedy parity and graph comparisons
passed, while strict probability checks failed at the unchanged 0.1 threshold.

## Repository tests

GitHub Actions runs the portable subset with `--cpu-only`; see
[development](development.md#local-checks) for its pinned environment and coverage.
To include the runtime integration tests, use a compatible SGLang installation:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests -q
.venv/bin/ruff format --check src tests tools examples
.venv/bin/ruff check src tests tools examples
```

Passing portable tests is necessary but does not replace whole-engine CUDA,
tensor-parallel, cache-refresh, or production-checkpoint validation.
