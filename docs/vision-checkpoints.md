# Vision checkpoint loading (experimental)

The `Olmo3MoeVisionForConditionalGeneration` entry point loads the language
model, SigLIP tower and pooling/MLP connector from a **complete** Olmo HF vision
export. It reuses Core's vision modules and the existing SGLang KDA/MoE decoder.
The source checkpoint's bundled HF configuration/`auto_map` files must remain
present; loading uses those files with `trust_remote_code=True`, just as for the
text model. No new tokenizer IDs are guessed at model load.

This is checkpoint-loading groundwork for [planning #63](https://github.com/allenai/oe-planning/issues/63).
Image request preprocessing, post-normalization feature injection, generated
answers and serving-runtime qualification are tracked in
[#65](https://github.com/allenai/oe-planning/issues/65). The entry point rejects
image batches and externally supplied embeddings until that integration exists.
Do not use this branch as a production image server or infer task accuracy from
the CPU feature check.

## Dependencies and CPU checks

Vision modules require OLMo-core's vision implementation; the original text model
does not import Core vision during registry discovery. The CPU checks below used
Core commit `8f8ae5df42de590268b87b3701679f8c21aecfcf`, Transformers 5.14.1,
PyTorch 2.13.0+cu130 (CPU execution), and safetensors. Core and SGLang runtime
updates must be qualified together before deploying an image server. The
existing SGLang source pin in `pyproject.toml` is unchanged.

With those sources installed or on `PYTHONPATH`:

```bash
python -m pytest tests/test_vision_loading.py tests/test_vision_weight_coverage.py -q
python -m olmo_sglang.vision.checkpoint audit /path/to/full-hf-vision-export
```

The vision test module skips if the Core vision dependencies are unavailable;
such a skip is **not** a passing vision qualification. The portable configuration
helpers in `olmo_sglang.vision.config` do not import Core, CUDA or SGLang.

## Language-only exports need their trained image rows restored

The two inspected Jason exports (`...merged-mt-lc-v6-lm1e4/hf-step5000-alt` and
`...hero-mt-lc-256k-lm1e4/hf-step5000-alt`) retained the vision modules but used
100,278-row LM input/output tables. Their image patch ID is 100280; the original
Core exports contain 100,352 rows. These exports were deliberately used for
language-only evaluation. Their bundled HF and vLLM wrappers explicitly refuse
image inference.

The audit rejects that mismatch before model allocation. To produce a **new**
artifact from the matching original Core export:

```bash
python -m olmo_sglang.vision.checkpoint restore-vocabulary \
  --hf /path/to/hf-step5000-alt \
  --core /path/to/step5000 \
  --output /path/to/new-vision-export
```

The tool checks the component configs, vocabulary shapes and exact equality of
the existing HF embedding/head rows with the prefix of the original Core tables.
It restores **trained** input and output rows, applies Core's ordered image-token
preparation, verifies the patch ID, and preserves all other weight names. It does
not invent embeddings or modify either input checkpoint. Select a matching pair
from the same training checkpoint; these checks do not establish equality of
every language-model expert tensor across two unrelated checkpoints.

The output deliberately uses the **full Core output vocabulary**, including its
normalization denominator. Thus text logprobs need not match the old truncated
language-only export. Compare against the original Core model under this policy.

Affected shards are streamed to new files without the old embedding/head tensors;
untouched shards are symlinked. The source must remain accessible. Plan disk space
for the affected shards, potentially nearly the entire model for a monolithic
checkpoint. A JSON index alone is insufficient: SGLang visits every tensor in
every shard, so leaving old vocabulary tensors in a referenced shard would load
duplicates. `vision-restoration.json` records restored tensor hashes, the
normalization policy, image IDs and shard dependencies. Incomplete output after
an I/O failure has no successful receipt and must not be served.

The native loader also checks every packed Q/K/V and expert slice: touching a
packed destination once does not count as loading all of its source tensors.

The audit checks the artifact contract and vocabulary; it is not a substitute
for the model loader's strict parameter-shape and completeness checks.

## Local evidence

A [read-only extraction](https://beaker.org/ex/01M5TEK43CRM08STX0W43Q3KJK)
from the 64K `step5000` checkpoint saved the real vision/connector and vocabulary
tensors (about 1.26 GB; no expert weights) in dataset
`01M5TEK43MCFKKZB4ZAGBYV3KJ`. Extraction ran on one Saturn CPU with zero GPUs,
high priority, unallocated, and a five-minute timeout. Execution took 4.6 s;
upload/finalization took longer. This dataset is a one-off validation fixture,
not a complete model or routine source-distribution mechanism.

With eight CPU threads and FP32 computation on those BF16 source weights:

| Check | Result |
| --- | --- |
| Vision/connector tensors loaded | 414, all exactly equal after FP32 widening |
| Restored input/output table shapes | 100352 × 1024 |
| Resolved patch token | 100280 |
| Load and tensor checks | 2.08 s |
| One 378×378 crop's patchified feature forward | 0.75 s |
| Output (two selected pooled groups) | 1 × 2 × 1024 |
| Difference from direct Core feature-layer concatenation + connector | exactly zero |
| Replacing synthetic patches with zero patches | changes features |

The check uses synthetic patch inputs and real weights. It tests loading and
feature computation, not image preprocessing or semantic accuracy. It restores
the tokenizer/config in memory; a complete restored expert checkpoint has not
yet been produced or loaded in an SGLang GPU process. The real SGLang CUDA-only
decoder remains to be qualified; CPU wrapper tests replace that decoder alone.

The 256K checkpoint's metadata was inspected, but its full model has not been
run. [#63](https://github.com/allenai/oe-planning/issues/63) stays open pending
complete-artifact/runtime validation.

Reproduce the real-weight CPU check after downloading that fixture:

```bash
python tools/check_vision_cpu_fixture.py /path/to/fixture --output report.json
```
