"""Bounded real-image native SGLang smoke with cache/isolation controls.

Inputs are a frozen eight-page olmOCR slice. Prompts follow the inspected Core
provider's document layout, not a chat template. Capped outputs are integration
smoke evidence, not a leaderboard score or a throughput benchmark.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import sglang
from PIL import Image, ImageDraw

import olmo_sglang
from olmo_sglang.vision.processing import OlmoVisionProcessor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("inputs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    processor = OlmoVisionProcessor.from_pretrained(args.checkpoint)
    tokenizer = processor.tokenizer
    prompt = [
        tokenizer.eos_token_id,
        processor.token_ids.image_placeholder_id,
        *tokenizer.encode("olmocr:", add_special_tokens=False),
    ]
    olmo_sglang.register()
    start = time.perf_counter()
    engine = sglang.Engine(
        model_path=args.checkpoint,
        trust_remote_code=True,
        enable_multimodal=True,
        context_length=8192,
        max_total_tokens=8192,
        max_running_requests=2,
        mem_fraction_static=0.65,
        disable_cuda_graph=True,
        chunked_prefill_size=256,
        attention_backend="triton",
        dtype="bfloat16",
        mm_feature_transport="cpu",
    )
    records = []
    report = {
        "engine_init_seconds": time.perf_counter() - start,
        "records": records,
        "checkpoint": args.checkpoint,
        "prompt_ids_unexpanded": prompt,
        "sampling": {"temperature": 0.0, "max_new_tokens": args.max_new_tokens},
        "scope": "Image integration smoke; bounded/truncated OCR outputs, no accuracy or throughput qualification",
    }

    def save():
        (out / "inference.json").write_text(json.dumps(report, indent=2) + "\n")

    def run(name, images, request_prompt=prompt, max_tokens=16):
        t = time.perf_counter()
        result = engine.generate(
            input_ids=request_prompt,
            image_data=[str(p) for p in images] or None,
            sampling_params={"temperature": 0.0, "max_new_tokens": max_tokens},
            return_logprob=True,
        )
        record = {
            "name": name,
            "images": [str(p) for p in images],
            "input_ids": request_prompt,
            "seconds": time.perf_counter() - t,
            "result": result,
        }
        records.append(record)
        save()
        return record

    try:
        for letter, word in (("a", "ALPHA"), ("b", "BETA")):
            image = Image.new("RGB", (500, 240), "white")
            ImageDraw.Draw(image).text((25, 60), word, fill="black", font_size=64)
            image.save(out / f"control-{letter}.png")
        a, b = out / "control-a.png", out / "control-b.png"
        first = run("image-a", [a])
        swapped = run("image-b", [b])
        run(
            "text-only",
            [],
            [
                tokenizer.eos_token_id,
                *tokenizer.encode("The capital of France is", add_special_tokens=False),
            ],
        )
        repeated = run("image-a-repeat", [a])

        def token_ids(record):
            return [
                row[1] for row in record["result"]["meta_info"]["output_token_logprobs"]
            ]

        report["repeat_tokens_equal"] = token_ids(first) == token_ids(repeated)
        if not report["repeat_tokens_equal"]:
            raise AssertionError("A/B/text/A request isolation changed greedy A tokens")
        report["image_swap_changes_logprobs"] = (
            first["result"]["meta_info"]["output_token_logprobs"]
            != swapped["result"]["meta_info"]["output_token_logprobs"]
        )
        if not report["image_swap_changes_logprobs"]:
            raise AssertionError("Image swap did not change output token logprobs")
        run(
            "two-images",
            [a, b],
            [
                tokenizer.eos_token_id,
                processor.token_ids.image_placeholder_id,
                processor.token_ids.image_placeholder_id,
                *tokenizer.encode("olmocr:", add_special_tokens=False),
            ],
        )
        mixed = engine.generate(
            input_ids=[
                prompt,
                [
                    tokenizer.eos_token_id,
                    *tokenizer.encode("Hello", add_special_tokens=False),
                ],
            ],
            image_data=[[str(a)], None],
            sampling_params={"temperature": 0.0, "max_new_tokens": 8},
            return_logprob=True,
        )
        report["mixed_batch"] = mixed
        save()
        source = Path(args.inputs)
        manifest = json.loads((source / "manifest.json").read_text())
        report["input_manifest"] = manifest
        for i, page in enumerate(manifest["pages"]):
            path = source / page["image"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == page["image_sha256"]
            t = time.perf_counter()
            expanded, images = processor.prepare(
                prompt, [Image.open(path).convert("RGB")]
            )
            preprocessing = time.perf_counter() - t
            record = run(f"ocr-{i:02d}", [path], max_tokens=args.max_new_tokens)
            record.update(
                page=page,
                preprocess_seconds=preprocessing,
                expanded_prompt_ids=expanded,
                image_crops=images[0].patches.shape[1],
            )
            # Re-score a short identical response prefix through prefill for the
            # Core comparison; response token positions follow image expansion.
            response_ids = token_ids(record)[:32]
            score = engine.generate(
                input_ids=prompt + response_ids,
                image_data=[str(path)],
                sampling_params={"temperature": 0.0, "max_new_tokens": 1},
                return_logprob=True,
                logprob_start_len=len(expanded) - 1,
            )
            record["response_prefix_ids"] = response_ids
            record["prefill_score"] = score
            save()
        engine.start_profile(
            output_dir=str(out / "profile"), activities=["CPU", "GPU"], with_stack=False
        )
        # A fresh control image avoids measuring only an embedding-cache hit.
        image = Image.new("RGB", (500, 240), "white")
        ImageDraw.Draw(image).text((25, 60), "PROFILE", fill="black", font_size=64)
        image.save(out / "profile.png")
        run("profile", [out / "profile.png"], max_tokens=8)
        engine.stop_profile()
        report["passed"] = True
        save()
        print(
            json.dumps(
                {
                    k: v
                    for k, v in report.items()
                    if k not in ("records", "mixed_batch", "input_manifest")
                },
                indent=2,
            )
        )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
