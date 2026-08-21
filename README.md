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
- Standard tensor parallel sharding for KDA projections, convolution windows,
  and recurrent state.
- Experimental branch-capable radix caching of OLMo KDA prefix state.
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
Its radix path uses FLA's intermediate recurrent states plus SGLang's
`extra_buffer` strategy to snapshot branch points, with copy-on-write for both
the convolution window and recurrent matrix. Tensor parallelism currently
requires matching model/attention TP groups and head counts divisible by the TP
size. Other limits are chain-only correctness-first speculative verification
and no changed-weight numerical attribution across an RL policy refresh. The
idle refresh lifecycle, production MILES refresh path, and breadth-one NGRAM
target verification are validated. The verifier is not yet a fused production
kernel and tree speculation remains unsupported. This is not yet the final
optimized serving kernel.

FLA is optional; attention-only checkpoints do not import it.

## Roadmap and TODOs

These items are intentionally explicit so a working correctness path is not
confused with complete production serving support.

### Correctness gate for MILES and Megatron Bridge

- [x] Load a real Megatron-Bridge-exported production checkpoint, with no
  synthetic or attention-only substitutions.
- [ ] Compare prompt logits, greedy tokens, and MoE router selections against
  the reference implementation at the intended serving dtype.
- [x] Exercise short multi-request rollouts through the same HTTP or embedded
  interface that MILES will use, including stop conditions and request
  cancellation.
- [x] Validate an idle policy-refresh boundary locally: establish cache reuse,
  flush, reload weights, require the first new-version request to miss, and
  require reuse to recover on the next request.
- [ ] Validate the production RL weight transfer with changed actor weights on
  every rollout replica and confirm generation changes at the new version.

### KDA serving features

- [x] **Radix state lifecycle:** copy both the convolution window and recurrent
  KDA matrix at prefix-tree branch points; cover prefix hits, divergent branches,
  eviction, request isolation, and cleared slot reuse in focused tests.
- [x] **Local radix engine parity:** run a real repeated-prefix continuation
  through the tiny hybrid KDA engine, observe nonzero `cached_tokens`, and match
  the cache-disabled greedy continuation.
- [x] **Branch-capable KDA snapshots:** expose FLA's per-chunk intermediate KDA
  states to SGLang's `extra_buffer` strategy without modifying SGLang or FLA.
  A 300-token local probe reuses the 256-token tracked prefix with overlap
  scheduling enabled and matches the uncached greedy output.
- [x] **Policy-refresh invalidation:** establish a 256-token hit, flush the idle
  engine, perform an actual checkpoint weight reload, require the first request
  to miss, and observe a new 256-token hit on the next request.
- [ ] **Production RL radix lifecycle:** validate a production checkpoint,
  changed actor weights, cancellation under load, and the refresh across every
  rollout replica before enabling the cache in MILES.
- [x] **Chain speculative correctness:** write every proposal's OLMo-semantic
  KDA state to SGLang's speculative scratch pool, leave the committed pool
  untouched during verification, and let SGLang commit only the accepted
  prefix. A breadth-one NGRAM engine performed 14 verify passes and exactly
  matched all 16 ordinary greedy tokens on a production-shaped local model.
- [ ] **Production speculative kernel:** fuse the chain verifier, remove its
  eager host synchronization/PyTorch loop, support CUDA-graph replay, and add
  tree-ancestor traversal before enabling branching algorithms.
- [x] **Tensor parallelism greater than one:** shard the KDA projections,
  unequal-width K/V heads, convolution windows, and recurrent state correctly;
  compare TP=1 and TP=2 greedy generation plus chosen-token log probabilities.
  A production checkpoint matched all eight greedy output IDs; chosen-token
  log probabilities differed by at most 0.04458 and by 0.02373 on average.
- [x] Validate mixed-length and long-prefix KDA batching, 64-token chunked
  prefill, in-flight and queued cancellation, and forced scheduler retraction.
  Organic retraction under production memory pressure remains a load-test item.

### Hardening and performance

- [x] Replace the correctness-first FLA chunk call for one-token decode with a
  packed Triton kernel that preserves OLMo's optional
  `beta = 2 * sigmoid(raw_beta)` semantics. Local model-shaped kernel
  measurements on an RTX 4090 reduced an eight-head 128-by-128 state update
  from 0.434 ms to 0.007 ms at batch one and from 0.463 ms to 0.010 ms at
  batch four. Torch recurrence parity covers both beta modes; the tiny hybrid
  engine retains its exact greedy tokens and 256-token radix-cache hits.
- [ ] Compare decode CUDA-graph replay against the matched packed eager path.
  Production-checkpoint packed-kernel parity is complete; prefill graphs stay
  disabled for a separate experiment.
- [ ] Replace or upstream the narrow FLA 0.5.2/Triton 3.6+ source shim.
- [ ] Add BF16 and production-dimension coverage, including real sparse-MoE
  layers rather than only the tiny dense smoke checkpoint.
- [ ] Optionally add a slow PyTorch CPU reference recurrence for portable unit
  tests; production KDA serving remains GPU-oriented.

Speculative decoding and TP>1 are not prerequisites for the first MILES
integration. Chain verification now has a correctness path, but ordinary
autoregressive decoding remains the production default until the verifier is
fused and benchmarked. The immediate radix gate is real-checkpoint numerical
correctness and policy-refresh invalidation in the current MILES topology of
TP=1 replicas.

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
sglang_output_ids=[53, 15, 39, 25]
```

The recurrence harness verifies that prompt prefill followed by cached
single-token decode matches one uninterrupted FLA sequence:

```bash
PYTHONPATH=src .venv/bin/python examples/check_kda_recurrence.py
```

It also confirms that enabling negative eigenvalues produces a nonzero result
delta from ordinary KDA, so the test would catch accidentally dropping the
`2 * sigmoid(beta)` behavior.

Exercise the complete radix state lifecycle through the embedded engine. The
command warms one prefix, continues it with a cache hit, recomputes that
continuation in a cache-disabled engine, and requires identical greedy output:

```bash
PYTHONPATH=src .venv/bin/python -m olmo_sglang.radix_smoke \
  --model /tmp/olmo-sglang-tiny-kda \
  --input-ids 2 3 4 5 \
  --max-new-tokens 4 \
  --mamba-radix-cache-strategy no_buffer
```

The JSON result includes each request's `cached_tokens`; the cached continuation
must be nonzero. This short fixture explicitly uses SGLang's endpoint-only
`no_buffer` baseline because it does not cross an intermediate-state tracking
boundary. For OLMo KDA, `auto` now resolves to the branch-capable `extra_buffer`
strategy.

Probe the same prompt three times on one GPU, both sequentially and as one
simultaneous scheduler batch:

```bash
PYTHONPATH=src .venv/bin/python -m olmo_sglang.radix_smoke \
  --model /tmp/olmo-sglang-tiny-kda \
  --input-ids 2 3 4 5 \
  --max-new-tokens 4 \
  --repeat-prompt-probe \
  --repeats 3 \
  --mamba-radix-cache-strategy no_buffer
```

The report contains three experiments. Plain sequential and simultaneous
requests are observational: an earlier completion may store its KDA state past
the prompt boundary, and recurrent state cannot be split out of a compressed
radix edge the way per-token attention KV can. The third experiment first seeds
`prompt[:-1]` with a throwaway one-token generation, then submits the three full
prompts simultaneously. Every seeded request must report a nonzero
`cached_tokens` count and identical greedy output. This is a small local model
of both the current grouped-RL miss pattern and an explicit prefill-seeding
strategy.

For native branch-point validation, generate a longer-context fixture and use
a prompt long enough to cross both FLA's 64-token KDA chunk boundaries and
SGLang's default 256-token state-tracking interval:

```bash
PYTHONPATH=src .venv/bin/python examples/create_tiny_parity_checkpoint.py \
  /tmp/olmo-sglang-radix-long \
  --profile hybrid-moe \
  --max-position-embeddings 512

PYTHONPATH=src .venv/bin/python -m olmo_sglang.radix_smoke \
  --model /tmp/olmo-sglang-radix-long \
  --prompt-length 300 \
  --max-new-tokens 4 \
  --repeat-prompt-probe \
  --repeats 3 \
  --mamba-radix-cache-strategy extra_buffer
```

The generated checkpoint and prompt are synthetic and intended only for a
controlled cache-mechanism test. In the validated run, the second and third
sequential requests and all three pre-seeded simultaneous requests reused 256
tokens; every path generated `[15, 23, 24, 23]`. The matched `no_buffer`
baseline could reuse the endpoint after explicit seeding, but not the repeated
prompt sequentially, and required overlap scheduling to be disabled. Production
acceptance still requires the real checkpoint, tokenizer, request distribution,
and policy-refresh lifecycle.

Exercise the policy-refresh boundary on the same long-prefix fixture:

```bash
PYTHONPATH=src .venv/bin/python -m olmo_sglang.radix_smoke \
  --model /tmp/olmo-sglang-radix-long \
  --prompt-length 300 \
  --max-new-tokens 4 \
  --policy-refresh-probe \
  --mamba-radix-cache-strategy extra_buffer
```

This performs a successful idle cache flush and an actual in-place reload of
the tiny checkpoint. The validated cache counts were `[0, 256]` before refresh
and `[0, 256]` afterward: old state did not cross the refresh, and cache reuse
recovered immediately within the refreshed policy version. Reloading identical
weights deliberately isolates cache lifecycle behavior; the production test
must additionally transfer changed actor weights to every rollout replica.

Exercise mixed prompt lengths through actual 64-token chunked prefill while
reusing the same 256-token recurrent-state snapshot:

```bash
PYTHONPATH=src .venv/bin/python -m olmo_sglang.radix_smoke \
  --model /tmp/olmo-sglang-radix-long \
  --prompt-length 300 \
  --max-new-tokens 4 \
  --mixed-chunked-prefill-probe \
  --chunked-prefill-size 64 \
  --mamba-radix-cache-strategy extra_buffer
```

This sends 268-, 300-, and 332-token prompts together after warming their
shared 256-token prefix. Every request must report at least 256 cached tokens
and match a fresh cache-disabled engine configured with the same prefill chunk
size. It exercises ragged batching, handoff between prefill chunks, radix-state
copy-on-write, and cached decode in one bounded local test.

Exercise cancellation cleanup while the KDA scheduler has both running and
queued work:

```bash
PYTHONPATH=src .venv/bin/python -m olmo_sglang.radix_smoke \
  --model /tmp/olmo-sglang-radix-long \
  --prompt-length 300 \
  --max-new-tokens 128 \
  --cancellation-probe \
  --chunked-prefill-size 64 \
  --mamba-radix-cache-strategy extra_buffer
```

The probe submits four ignore-EOS requests with a two-request admission cap,
cancels the first and last request IDs, and requires both abort responses. It
then sends a recovery batch, compares it with a fresh cache-disabled engine,
and requires an idle cache flush to succeed. This catches leaked recurrent
slots and state contamination after both running and queued cancellation.

Exercise SGLang's actual decode retraction/resume path deterministically:

```bash
PYTHONPATH=src .venv/bin/python -m olmo_sglang.radix_smoke \
  --model /tmp/olmo-sglang-radix-long \
  --prompt-length 300 \
  --max-new-tokens 64 \
  --retraction-probe \
  --retraction-interval 7 \
  --chunked-prefill-size 64 \
  --mamba-radix-cache-strategy extra_buffer
```

This uses SGLang's scheduler test hook to force decode retractions every seven
forwards. The response metadata must prove that retraction occurred, every
resumed 64-token continuation must exactly match a normal scheduler control,
and an idle cache flush must succeed. The hook makes the correctness test
deterministic; a separate production load test should still observe organic
retraction under real memory pressure.

## Exercise chain speculative decoding

The packaged speculative smoke test launches matched ordinary and breadth-one
NGRAM engines, generates greedily from the same token IDs, requires identical
output, and requires SGLang's response metadata to prove that target
verification actually ran:

```bash
uv pip install --python .venv/bin/python ninja

PYTHONPATH=src uv run --no-sync python -m olmo_sglang.speculative_smoke \
  --model /path/to/olmo-hf-checkpoint \
  --prompt-length 32 \
  --max-new-tokens 16 \
  --context-length 64
```

Ninja is needed for SGLang's bundled NGRAM corpus JIT extension; it is a runtime
environment prerequisite rather than an `olmo-sglang` package dependency. The
JSON report includes `spec_verify_ct`, proposed and accepted draft counts,
acceptance rate, and average accepted length. A validated production-shaped
synthetic run performed 14 verify passes, proposed 42 drafts, accepted one, and
exactly reproduced the ordinary output IDs. Low acceptance is unsurprising for
random weights; this probe is a correctness gate, not a speed benchmark.

This first implementation intentionally supports only a linear draft chain. It
runs eager PyTorch recurrence and writes every post-token state to SGLang's
intermediate state pool, without mutating the committed pool. SGLang's central
commit then selects the accepted state, so rejection and rollback use its
normal lifecycle. CUDA graphs are disabled for the smoke test. Branching NGRAM
or EAGLE trees, a fused verifier, and speculative performance validation remain
TODOs. The repository's 8-wide tiny full-attention fixture is below
FlashInfer's supported production head sizes, so use a production-shaped local
fixture or the real checkpoint for this command.

## Exercise tensor parallelism

The packaged TP smoke test starts a TP=1 control and then a standard TP engine,
runs the same greedy token-ID request through both, and requires exact output
token parity. It also reports the max and mean absolute difference between the
chosen-token log probabilities:

```bash
PYTHONPATH=src uv run --no-sync python -m olmo_sglang.tp_smoke \
  --model /path/to/olmo-hf-checkpoint \
  --tp-size 2 \
  --prompt-length 32 \
  --max-new-tokens 8 \
  --context-length 512 \
  --mem-fraction-static 0.25
```

The harness disables radix reuse and CUDA graphs, and pins Triton attention
with one-token pages, so only tensor sharding changes. The standard TP path
requires the model and attention TP groups to match and both KDA key/value head
counts to divide evenly by `tp-size`. Distinct attention TP, DCP, and a TP-aware
MILES replica topology remain separate work. The current MILES launcher still
uses eight independent TP=1 inference replicas.

On the production SFT checkpoint, TP=1 and TP=2 returned the same greedy token
IDs, `[3505, 198, 5, 6, 7, 8, 5, 6]`. The chosen-token log-probability max and
mean absolute differences were 0.04458 and 0.02373, respectively. Treat this as
a functional greedy-parity gate, not a claim of bitwise numerical equivalence.

## Local production-shaped parity loop

Use the independent tiny reference before spending a full-checkpoint cycle on
Beaker. Four deterministic BF16 profiles isolate the base attention model, KDA
prefill/cached decode, latent sparse-MoE execution, and the production model's
critical dimensions:

```bash
export PARITY_ROOT=/tmp/olmo-sglang-parity

for profile in attention-dense kda-dense hybrid-moe production-shape; do
  PYTHONPATH=src .venv/bin/python examples/create_tiny_parity_checkpoint.py \
    "$PARITY_ROOT/$profile" --profile "$profile"
  PYTHONPATH=src .venv/bin/python -m olmo_sglang.parity \
    --model "$PARITY_ROOT/$profile" \
    --max-new-tokens 4 \
    --output "$PARITY_ROOT/$profile.json" \
    --require-parity
done
```

The checkpoints include a tiny tokenizer, ChatML-style chat template, peri-LN,
NoPE attention, non-unit embedding scale, headwise Q/K normalization, elementwise
attention gating, and unequal K/V widths. The `hybrid-moe` profile adds
factorized KDA gates, negative eigenvalues, a latent four-expert top-2 MoE,
unaligned expert widths, and a shared expert. The `production-shape` profile
uses the real 1280 hidden width, 2048 full-attention width, 4096 KDA value
width, 640 latent width, 952 expert width, 8568 dense width, top-16 routing,
and the first production block's four-KDA-then-full-attention pattern. It keeps
32 rather than 512 routed experts so it remains practical on a 24 GB GPU.

`olmo-sglang.parity` tokenizes and renders the prompt once, passes the exact
token IDs to both implementations, and compares an independent PyTorch
reference with embedded SGLang. Its JSON report preserves the rendered prompt,
input IDs, greedy outputs, per-step top log probabilities, forced-prefix
full-prefill results, and shape/finite/statistical summaries for every reference
boundary. The forced-prefix results distinguish cached-decode drift from a
prefill/model mismatch. Exact local token agreement is the gate before running
the same check against a production checkpoint.

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
