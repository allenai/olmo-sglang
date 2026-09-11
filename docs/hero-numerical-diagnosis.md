# Hero numerical diagnosis

The full step-75500 qualification failed its existing 0.1 absolute log-probability threshold. Eight greedy predictions matched in both serving modes, but the worst checked SGLang top-token error was 0.3118 with unchunked prefill and 0.5619 with chunked prefill plus decode graphs. Core/HF full-vocabulary error reached 0.5473 (mean 0.08255). Matching greedy tokens does not resolve these errors. Keep the failed report and threshold unchanged.

First isolate Core/HF drift using native BF16 weights, HF eager attention, Core torch SDPA, native Core experts, and the same explicitly selected recurrent KDA reference helper used for the failed gate:

```bash
python tools/diagnose_hero_core.py --model /path/to/step75500/hf \
  --recurrent-hf-prefill --reference-report /path/to/failed-serving.json \
  --logprob-atol 0.1 --output /path/to/core-layerwise.json \
  --save-activations /path/to/hf-layerwise.pt
```

The tool compares block inputs/outputs, attention input/output, all four peri-norm inputs/outputs, and combined MLP output (input to the post-FFN norm). It repeats each Core block on the exact captured HF block input. Compare accumulated errors against these same-input errors: this distinguishes divergence entering a layer from differences produced locally. It captures both original prompt prefixes, not all eight generated continuations. No kernel, accumulation dtype, router, weight, or gate is changed. Optional HF activation tensors support subsequent SGLang layer comparisons. This is a diagnostic and does not qualify training or serving.

Next isolate SGLang chunking from graph execution, which the original two-mode gate varied together. Cross graph execution off/on with prefill chunks 128/32, retain exact forced-prefix token IDs and checked top-token log-probabilities, and keep the same threshold for every mode. Use the optional diagnostic matrix below; the normal two-mode qualification defaults are unchanged. Compare identical prefix/token pairs; differing top-token sets require explicit intersection reporting rather than assuming missing entries match. The original 16-token prompt had identical forced-prefix error summaries in both modes; the longer prompt differed, making chunking the first serving variable to isolate. This observation is not proof that graphs are irrelevant.

```bash
python tools/qualify_hero_serving.py --model /path/to/step75500/hf \
  --recurrent-hf-prefill --diagnostic-mode-matrix \
  --require-token-parity --logprob-atol 0.1 \
  --output /path/to/serving-mode-matrix.json
```

The JSON includes direct comparisons varying one setting at a time. Those comparisons explicitly report the number of common top-token IDs and IDs missing from either set; disjoint sets produce no invented zero error. All mode outputs must still pass the original absolute threshold against HF. Diagnostic matrix and live-update testing require separate invocations.

If a particular Core/HF boundary first diverges, compare that operation on identical inputs before making changes. For routing, record selected expert IDs and weights before attributing later discrepancies to GEMM order. For full attention, compare Q/K normalization, absolute-position scaling, and attention output separately. For KDA, compare projection/convolution, recurrent state, and output normalization. Only after Core/HF drift is understood should corresponding SGLang layer outputs on the saved HF activations be instrumented; that worker-level instrumentation remains follow-up work.

Validation: four capture-helper tests pass (math unchanged, hook cleanup on success/error, exact boundary coverage, shape/non-finite rejection). The actual two-layer tiny hero GPU diagnostic passed with max full-vocabulary log-probability error 0.033332, matching the earlier semantic reference result. The tiny four-mode GPU matrix also passed at the existing tiny tolerance 0.05, with all eight predictions matching in every mode; exact errors and direct mode comparisons are recorded in `measurements/hero-tiny-mode-matrix-20260910.json`. Full checkpoint layerwise diagnosis and the full checkpoint four-mode matrix remain pending.
