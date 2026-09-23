# Per-head Q/K gains and scalable softmax

These optional features use the HF schema introduced in OLMo-core
`1a1acc8b437d4a4920d68dfcbea85a3024246979`. Configurations without either flag
retain the existing attention computation.

With `qk_norm_per_head_gains=true`, each full-attention layer loads separate
`q_norm.weight[query_heads, head_dim]` and `k_norm.weight[kv_heads, head_dim]`.
RMS statistics and gain multiplication use FP32, followed by one cast to the
activation dtype. This requires `use_head_qk_norm=true`.

With `scalable_softmax=true`, `ssmax_scale[query_heads]` is a learned parameter.
After normalization and optional RoPE, queries are multiplied by
`log(absolute_position + 1) * ssmax_scale`. The combined scale is formed in the
activation dtype before multiplying the query. SGLang supplies absolute token
positions for packed prefill, chunked continuation, and decode. The attention
backend still applies its ordinary `1/sqrt(head_dim)` factor.

HF loading and MILES tensor updates use the same parameter loaders. They accept
full HF tensors or correctly shaped local shards, shard by attention TP rank,
and replicate KV head gains when TP exceeds the KV head count. Copies preserve
parameter storage for captured graphs. Missing gain or scale tensors are
rejected before the first attention forward, after partial loading buckets have
completed. Loading paths that bypass these loaders are not qualified.

## Serving checks

The `scaled-attention-hybrid-moe` fixture includes KDA, latent MoE, distinct
per-head gains, and nonuniform softmax scales. Its attention head dimension is
16 and expert intermediate width is 32.

```bash
python tools/create_tiny_parity_checkpoint.py /tmp/tiny-scaled-attention \
  --profile scaled-attention-hybrid-moe --max-position-embeddings 256
python -m pytest tests/test_attention.py tests/test_config.py tests/test_toy_reference.py
```

Run the HF-versus-SGLang check with the matching HF Python implementation:

```bash
python tools/qualify_serving.py \
  --tiny-dir /tmp/scaled-attention-serving-check \
  --hf-source /path/to/OLMo-core/src/olmo_core/nn/moe/v2/hf \
  --recurrent-hf-prefill --require-token-parity --logprob-atol 0.05 \
  --check-live-update --output /tmp/scaled-attention-serving-check.json
```

Use `--model /path/to/hf` instead of `--tiny-dir` for an existing checkpoint.
`--check-live-update` requires a new tiny fixture: it creates a separate changed
checkpoint, updates Q gains, K gains, and scales in separate buckets, then
compares captured-graph inference against a fresh engine loading that checkpoint.
Omit that flag for an existing model.

The check uses prompt lengths 16 and 81, four generated tokens each, forced HF
prefixes, unchunked prefill, 32-token chunked prefill, and decode graphs for batch
sizes 1/2/4. It checks top-token and cached-decode log probabilities against the
requested tolerance and requires identical greedy outputs between serving modes.
`--require-token-parity` also requires agreement with HF.

`--recurrent-hf-prefill` explicitly selects the HF export's recurrent FLA helper;
the pinned chunk-prefill reference failed to compile on the RTX 4090. Serving
KDA execution is unchanged by this reference setting.

To separate chunking from graph execution, run `--diagnostic-mode-matrix`
without `--check-live-update`. It crosses graphs off/on with prefill chunks
128/32 and compares identical forced prefixes. Comparisons report common and
missing top-token IDs; disjoint sets do not produce an invented zero error.

## Measured scope

On September 22, 2026, source `02ccb5dcf641cbabc9b78a5bc65dacf8690707a7`
passed the tiny serving check on one RTX 4090 with base image
`01M24E7MSDGN2QFW1T8Z31BCKS`, SGLang `3145136dcd1238754e0ea2b2ffd546532119c71c`,
and HF source from Core `3d35ab326b72d92e06137cc310631d9187d8a2c5`.
Both serving modes matched all eight HF tokens. Maximum checked logprob error
was 0.04114890 against a 0.05 threshold. Live gain/scale updates changed outputs
and matched a fresh changed-checkpoint engine with zero logprob difference.
The command above retains those settings; the fixture and tool names have since
been updated. Save new reports with run artifacts as described in
[development](development.md).

These are tiny BF16, TP1 checks. Full-checkpoint probability parity, TP greater
than one, speculative decoding with these features, and distributed publication
remain separate gates. See [numerical findings](numerical-findings.md) for the
recorded full-checkpoint failure and arithmetic differences.
