# Hero attention support

This implementation targets the hero HF schema introduced in OLMo-core
`1a1acc8b437d4a4920d68dfcbea85a3024246979`, on the existing olmo-sglang
`81a312ee8326e279a4641e03542ef971a5ffb863` / SGLang
`3145136dcd1238754e0ea2b2ffd546532119c71c` runtime.

With `qk_norm_per_head_gains=true`, each full-attention layer loads separate
`q_norm.weight[query_heads, head_dim]` and
`k_norm.weight[kv_heads, head_dim]`. RMS statistics and multiplication by the
gains use FP32, followed by one cast to the activation dtype. This requires
`use_head_qk_norm=true`. Configurations without the new flag retain the existing
shared-gain implementation.

With `scalable_softmax=true`, `ssmax_scale[query_heads]` is a learned parameter.
After normalization and optional RoPE, queries are multiplied by
`log(absolute_position + 1) * ssmax_scale`, forming the combined scale in the
activation dtype before multiplying the query. SGLang supplies absolute positions
for packed prefill, chunked continuation, and decode; positions never come from
the flattened batch offset. The attention backend still applies its ordinary
`1/sqrt(head_dim)` factor.

The normal HF loader and MILES' ordinary tensor-update path use the same parameter
loaders. They accept full HF head tensors or correctly shaped local shards, shard
by attention TP rank, and replicate KV head gains when TP exceeds the KV head
count. Copies preserve parameter storage for captured graphs. Missing hero tensors
are rejected before the first attention forward, after any partial loading buckets
have completed. Direct loading paths that bypass model parameter loaders are not
qualified.

The tiny `hero-hybrid-moe` checkpoint profile exercises KDA, latent MoE, distinct
per-head gains and nonuniform learned scales:

```bash
python tools/create_tiny_parity_checkpoint.py /tmp/tiny-hero \
  --profile hero-hybrid-moe --max-position-embeddings 256
python -m pytest tests/test_hero_attention.py tests/test_config.py tests/test_toy_reference.py
```

Unit tests cover BF16 rounding, packed absolute positions, head sharding and KV
replication, missing weights, separate live loading buckets, and CUDA graph replay
after in-place gain and position updates. CUDA tests require the pinned native
runtime. TP sharding unit tests are not a distributed TP qualification. Full hero
checkpoint inference, TP greater than one, speculative decoding, and full MILES
publication/restart remain separate integration gates.

The reproducible live check accepts an existing HF checkpoint without modifying
it, or creates a new tiny fixture using the hero HF Python files:

```bash
python tools/qualify_hero_serving.py \
  --tiny-dir /tmp/hero-serving-check \
  --hf-source /path/to/OLMo-core/src/olmo_core/nn/moe/v2/hf \
  --recurrent-hf-prefill --require-token-parity --logprob-atol 0.05 \
  --check-live-update \
  --output /tmp/hero-serving-check.json
```

Use `--model /path/to/hf` instead of `--tiny-dir` for an existing checkpoint. The
check compares mixed prompt lengths 16/81, four generated tokens each, forced
reference prefixes, unchunked prefill, 32-token chunked prefill, and decode graphs
for batch sizes 1/2/4. It fails on the configured logprob tolerance, a difference
between the two SGLang modes' greedy outputs, or HF token disagreement when
`--require-token-parity` is supplied. It records observed differences; default
BF16 numerical equality is not assumed.

`--recurrent-hf-prefill` explicitly selects the export's existing recurrent FLA
reference helper with native BF16 linear/attention. The pinned FLA chunk-prefill
reference failed to compile on the local RTX 4090 (`next_power_of_2`); the serving
KDA implementation is unchanged. The tiny live check passed with exact agreement
on all eight greedy tokens in both modes and maximum full-vocabulary conditional
logprob error 0.02998. The first successful run used Core hero source `3a643bc` and
the compiled base image `01M24E7MSDGN2QFW1T8Z31BCKS`. This is a small-model
qualification, not evidence of full hero numerical parity or performance.

The final reproducible run also passed three separate live tensor-update buckets
(Q gains, K gains, scales), followed by cache flush and captured-graph inference.
Its outputs exactly matched a fresh engine loading the changed HF checkpoint:
identical tokens and zero logprob difference, with verified changed logprobs from
the original weights. This is the ordinary local tensor update path, not a
qualification of distributed MILES publication or restart. `--check-live-update`
is intentionally limited to newly created tiny fixtures, so it can copy the
checkpoint into a separate changed artifact without copying a production model.
Omit it when using `--model`.

Measured evidence is retained in
[`measurements/hero-tiny-serving-20260910.json`](measurements/hero-tiny-serving-20260910.json).
The final regression suite passed all 83 tests in the pinned CUDA runtime;
Ruff import/error checks and formatting checks passed.
