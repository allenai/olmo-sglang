# Vision checkpoint loading (experimental)

The `Olmo3MoeVisionForConditionalGeneration` entry point loads the language
model, SigLIP tower and pooling/MLP connector from a **complete** Olmo HF vision
export. It reuses Core's vision modules and the existing SGLang KDA/MoE decoder.
The source checkpoint's bundled HF configuration/`auto_map` files must remain
present; loading uses those files with `trust_remote_code=True`, just as for the
text model. No new tokenizer IDs are guessed at model load.

This is checkpoint-loading groundwork for [planning #63](https://github.com/allenai/oe-planning/issues/63).
Image request preprocessing, post-normalization feature injection, generated
answers and image-serving qualification are tracked in
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
feature computation, not image preprocessing or semantic accuracy. That fixture probe restores the tokenizer/config in memory. A separate full-artifact
check below exercises the converter against every retained checkpoint tensor.
CPU wrapper tests replace the CUDA-only decoder alone; the native H100 check
below separately exercises the real decoder and full model loader.

The 256K export has the same 23,855 tensor names, shapes and dtypes as the
64K export; its vision and connector configs are identical. The only nested
text-config difference is `max_position_embeddings` (65,536 versus 262,144).
Its config passes the same validation once the original vocabulary size is
restored in memory. This is metadata compatibility, not a 256K weight-content
or runtime qualification. [#63](https://github.com/allenai/oe-planning/issues/63) remains open for implementation
review; image-conditioned generation remains #65.

Reproduce the real-weight CPU check after downloading that fixture:

```bash
python tools/check_vision_cpu_fixture.py /path/to/fixture --output report.json
```

For a bounded native SGLang check on a compatible GPU runtime:

```bash
python tools/check_vision_runtime.py /path/to/full-restored-export --output runtime.json
```

This loads all components but disables image request processing and CUDA graphs,
uses a 256-token context limit, and generates only four text tokens. It does not
qualify image serving, full-context operation, or throughput.

## Complete restored 64K artifact

The [full-artifact CPU check](https://beaker.org/ex/01M5TFWR1KDEFTB7VCRCWJCT5G)
created `/weka/oe-training-default/robertb/vision63/64k-7c54bb9` without changing
Jason's original exports. All **23,853 non-vocabulary tensors** matched the
source HF tensors exactly (name, shape, dtype and value); the two vocabulary
tables were restored from Core. The output audit found 23,855 tensors and all
six image token IDs at 100278–100283 within a 100352-row vocabulary.

Restoration, full tensor comparison and source/output file hashing took **124.8 s**;
job execution including environment setup took about 141 s. This ran on four CPU
cores with a 16 GiB request, high/unallocated, on Jupiter after a Saturn attempt
was preempted during image startup. It used PyTorch 2.13.0+cu130 and Transformers
5.12.1 with GPU execution unused. The first setup attempt selected the image's
old source through inherited `PYTHONPATH` and exited before checkpoint writes;
the successful run explicitly selected the pinned sources.

Result dataset `01M5TFWR1V0XG8SAZFT45FP33S` contains `complete-artifact.json`,
including hashes for all output files and the two source weight files, source
commits, image IDs and the full-vocabulary policy. The same receipt resides in
the output checkpoint. Image processing configuration remains in the bundled
Core config; executable preprocessing and request integration are still #65.
This artifact audit does not establish image-conditioned correctness.

## Native H100 loading check

The [four-token native smoke](https://beaker.org/ex/01M5TG5Z080YF0PDJCTFS0WMVM)
passed on one H100 80 GB, loading the complete restored 64K artifact through
SGLang's native model discovery, strict full-component loader and KDA cache
setup. It generated four text tokens with finite logprobs. Result dataset
`01M5TG5Z0VTJSM1PTNGQQFX937` retains the generation, logprobs, hardware and runtime
versions. Source: `080bcca`; SGLang `0.5.19.dev49+g3145136`, FLA 0.5.2,
PyTorch 2.13.0+cu130, Transformers 5.12.1.

Engine initialization took **46.50 s**. The first cold request took **83.20 s**;
no warm-throughput measurement was attempted, and this must not be reported as
steady-state inference performance. Job execution was **154 s**, with about
three minutes of preceding node/image startup. The job was high/unallocated on
Jupiter with a 12-minute timeout and exited successfully.

This covers TP=1, a 256-token runtime context cap, graphs disabled, image request
processing disabled, and four generated text tokens. It does not qualify image
answers, long-context operation, tensor parallelism or production serving.
