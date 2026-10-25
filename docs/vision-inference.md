# Experimental image inference

This follow-on to checkpoint-loading PR #28 implements the image path for
planning [#65](https://github.com/allenai/oe-planning/issues/65). It requires a
complete restored vision checkpoint; language-only exports still fail validation.

The processor reuses pinned Core's 378/14 SigLIP multi-crop recipe, with the
checkpoint's `recipe.max_crops`, evaluation-time crop selection, mean/std 0.5,
2×2 pooling and 4-patch overlap margins. It resolves all structural token IDs
from the trained tokenizer. There must be exactly one `<|image|>` placeholder
per image. Pretokenized input is expanded directly without decode/re-tokenize.

The native wrapper runs the resident SigLIP tower and connector. At each patch
position it places **normalized learned patch-token embedding + image feature**;
all text/structural embeddings undergo the same scale and norm. The language
model explicitly bypasses a second embedding normalization for this path.
Attention remains causal, as required by the KDA model. No bidirectional Molmo
image-attention mask is introduced.

SGLang's existing multimodal interfaces handle image-dependent padded IDs,
feature caching and chunk slicing. Hashes cover pixel tensors, pooling indices
and image-token layout. Only patch-token runs are replaced: start/end/column
markers remain ordinary learned embeddings. The first supported deployment is
TP=1 without speculation. Audio/video, quantized models and caller-supplied
precomputed embeddings are rejected. Image accuracy, production serving and
long-context behavior must not be inferred from a short smoke run.

## Usage

Install Core `8f8ae5df42de590268b87b3701679f8c21aecfcf` alongside the pinned
SGLang runtime. For embedded inference, call `olmo_sglang.register()` before
creating an engine with `enable_multimodal=True`. This registers the external
model and processor packages. For CLI serving set both environment variables:

```bash
export SGLANG_EXTERNAL_MODEL_PACKAGE=olmo_sglang.models
export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=olmo_sglang.processors
```

For the inspected stage-1 checkpoint use document prompting, not its inherited
text chat template:

```python
from olmo_sglang.vision.processing import OlmoVisionProcessor
processor = OlmoVisionProcessor.from_pretrained(checkpoint)
ids = processor.document_prompt_ids("olmocr:")
result = engine.generate(
    input_ids=ids,
    image_data=["page.png"],
    sampling_params={"temperature": 0, "max_new_tokens": 128},
)
```

For two images pass `num_images=2` when constructing the prompt and supply two
image inputs. Generic applications may instead place the explicit placeholders
in their own tokenized prompts; this does not silently add a chat template.

## CPU evidence

`tools/check_vision_toy_cpu.py` combines the extracted real SigLIP/connector and
trained input/output embeddings with a synthetic two-layer CPU KDA/MoE decoder.
It uses the actual wrapper's image and forward methods and pinned SGLang's
multimodal placement/cache implementation; only the CUDA text executor is
substituted. The probe ran in 9.97 s on eight threads, loaded 414 visual tensors,
and tested two image controls. Full and 97-token chunked embeddings were
**bitwise identical** to the direct Core ordering. Swapping images changed the
toy decoder's logits. This is a machinery check, not a semantic evaluation.

```bash
PYTHONPATH=src:. python tools/check_vision_toy_cpu.py /path/to/vision63-fixture \
  --output cpu-result.json
```

Local portable suite: 196 passed, 172 CUDA-dependent skips. Runtime dependencies
are required for the SGLang placement probe even though execution is on CPU.

## H100 smoke

`tools/check_vision_inference.py` is the bounded GPU validation harness. It runs
image A/B/text/A isolation, multiple images, mixed text/image requests, an
eight-page frozen olmOCR slice, response-prefix scoring and a CPU/GPU profile.
It uses 256-token chunked prefill and keeps generations, token logprobs, exact
inputs and timings. The OCR generations are capped; resulting task checks are
smoke diagnostics, not benchmark accuracy.

Native H100 run on source `384c971`:
[Beaker](https://beaker.org/ex/01M5TJH45F728RA8WDSPXD8M2W),
result dataset `01M5TJH45RJHVPN2TWXMSN2NP9`. It exited successfully in 252 s
(including process startup and cold compilation). Engine initialization was
50.34 s; the first image request took 85.70 s. Warm control requests took
0.25–0.33 s, but this tiny, eager, batch-1 smoke is **not** a throughput benchmark.

- Image A read `ALPHA`, B read `BETA`, and the two-image request read both in
  order. A/B/text/A repeated A's greedy tokens exactly. Mixed text/image requests
  completed, and the text-only request answered Paris for France's capital.
- All eight OCR pages completed; five hit the 512-token cap, three reached EOS.
  The official olmOCR 0.4.27 checks passed 24/52, including all eight baseline
  nonempty/nonrepeating-output checks. Failures on capped pages do not establish
  an accuracy regression. Category counts: multi-column 3/10, old scans 5/12,
  tables 3/17, headers/footers 5/5, baseline 8/8.
- Every response and its token logprobs is retained, along with prefill scoring
  of the first 32 response tokens. Inputs are the frozen PNGs in payload dataset
  `01M5TJ4FECF2Q4J1J30BC44NG2`; hashes and upstream dataset revision are in the
  saved manifest. No renderer differences enter the paired comparison.
- CPU/GPU trace: `inference/profile/1792967217.3940604-TP-0.trace.json.gz`
  within the result dataset. It captures a fresh image with 541 prompt tokens,
  256-token prefill chunks and two generated tokens. The encoder/connector is
  called once; subsequent chunks reuse its features. Profiling perturbs timings.

Runtime: SGLang `0.5.19.dev49+g3145136`, torch `2.13.0+cu130`, Transformers
`5.12.1`, FLA `0.5.2`, NVIDIA H100 80GB HBM3. CUDA graphs were disabled. TP>1,
speculation, production concurrency, long contexts and uncapped benchmark
accuracy remain unqualified.

`tools/score_vision_ocr_smoke.py` applies the official diagnostic checks.
`tools/score_vision_core.py` scores the identical saved response prefixes with
the original full-vocabulary Core checkpoint and the pinned olmo-eval MoE
fallback source. Its report measures disagreement; successful completion alone
is not a numerical acceptance gate.


## Paired Core arithmetic check

[Core scoring run](https://beaker.org/ex/01M5TK5Q2VHS1H2XT7AFDS4ANP)
completed in 84 s on one H100; result dataset `01M5TK5Q32542JMFMQFYFX3CZX`.
The first page included 64.3 s of cold compilation; subsequent forward/scoring
calls took 63–71 ms. These are teacher-forced scoring calls, not generation
throughput. Core peak allocated tensor memory was 26.36 GB.

The original Core checkpoint scores the **same first 32 response tokens on each
of the eight pages** (256 total), with the same pixels, image preprocessing and
full trained vocabulary. Core revision is `8f8ae5df42de590268b87b3701679f8c21aecfcf`;
MoE fallback helpers come unchanged from olmo-eval
`2f383bc7d26aa413a99ad2a97b7b8c7e099ddc0c`. Attention uses Torch SDPA.
This probe feeds one unpadded causal document per call: `document_mode=False`
avoids Core Torch attention's unsupported packed-document metadata. It does not
add bidirectional image attention. No generation/KV-cache patches are installed
in the reference scoring model.

| Serving score vs Core | Mean absolute error (nats) | Signed mean (serving − Core) | Maximum absolute error | Trainer/serving ratio outside [0.8, 1.2] |
|---|---:|---:|---:|---:|
| Decode | 0.005734 | +0.000359 | 0.204707 | 0 / 256 |
| Prefill | 0.005466 | +0.000601 | 0.228574 | 2 / 256 |

Decode versus SGLang prefill on those tokens had MAE 0.005563 nats. These are
small, descriptive checks on mostly confident OCR prefixes; no statistical
non-regression claim, universal BF16 error floor, or full accuracy claim follows
from them. They complement the CPU embedding-order parity and the native image
swap/cache controls. Broader numerical and uncapped evaluation qualification
belongs to the next validation pass.

Combined CPU checks, official OCR diagnostics, per-token scores and provenance
are preserved in evidence dataset `01M5TKC07G4Z1G2X0D4TQW92MF`.
