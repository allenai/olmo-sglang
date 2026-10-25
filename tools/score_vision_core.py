"""Score identical saved image-response tokens with the pinned Core VLM.

Run after the native SGLang smoke, in a fresh process so it owns the GPU. Uses
Jason's pinned eval MoE fallback source, supplied explicitly, and the same full
Core vocabulary as the restored HF artifact. This does not generate new text.
"""

import argparse
import hashlib
import importlib.util
import json
import time
from importlib import metadata
from pathlib import Path

import torch
from olmo_core.nn.vision.multimodal import MultimodalLMConfig
from PIL import Image
from safetensors.torch import load_file

from olmo_sglang.vision.processing import OlmoVisionProcessor


def torch_attention(config):
    if isinstance(config, dict):
        return {
            k: "torch" if k == "backend" and isinstance(v, str) else torch_attention(v)
            for k, v in config.items()
        }
    if isinstance(config, list):
        return [torch_attention(v) for v in config]
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("core_checkpoint")
    parser.add_argument("hf_checkpoint")
    parser.add_argument("inference_json")
    parser.add_argument("--reference-moe-fallback", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    spec = importlib.util.spec_from_file_location(
        "pinned_eval_moe_fallback", args.reference_moe_fallback
    )
    fallback = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fallback)
    fallback.install_torch_moe_permutation()
    fallback.install_grouped_mm_fallback(torch.device("cuda"))
    fallback.install_swiglu_row_buckets()
    fallback.install_conv1d_autotune_buckets()
    root = Path(args.core_checkpoint)
    cfg = MultimodalLMConfig.from_dict(
        torch_attention(
            json.loads((root / "olmo_core_config.json").read_text())["model"]
        )
    )
    # This probe feeds one unpadded document at a time. Core's plain Torch
    # backend rejects packed-document metadata; no boundary is needed here.
    cfg.document_mode = False
    started = time.perf_counter()
    model = cfg.build(init_device="meta")
    state = load_file(root / "model.safetensors")
    model.load_state_dict(state, strict=True, assign=True)
    del state
    model = model.to(dtype=torch.bfloat16, device="cuda").eval().requires_grad_(False)
    processor = OlmoVisionProcessor.from_pretrained(args.hf_checkpoint)
    inference = json.loads(Path(args.inference_json).read_text())
    if not inference.get("passed"):
        raise ValueError("The native SGLang smoke must complete before Core comparison")
    report = {
        "core_checkpoint": str(root),
        "versions": {
            p: metadata.version(p) for p in ("torch", "flash-linear-attention")
        },
        "fallback_sha256": hashlib.sha256(
            Path(args.reference_moe_fallback).read_bytes()
        ).hexdigest(),
        "core_init_seconds": time.perf_counter() - started,
        "records": [],
        "normalizer": "full_core_vocabulary",
        "attention_backend": "torch",
        "document_mode": "single_unpadded_causal_sequence",
        "reference_moe_fallback": args.reference_moe_fallback,
    }
    for record in inference["records"]:
        if not record["name"].startswith("ocr-"):
            continue
        image = Image.open(record["images"][0]).convert("RGB")
        prompt, images = processor.prepare(record["input_ids"], [image])
        assert prompt == record["expanded_prompt_ids"]
        response = record["response_prefix_ids"]
        tokens = torch.tensor([prompt + response], dtype=torch.long)
        mask = torch.zeros_like(tokens, dtype=torch.float32)
        mask[:, len(prompt) - 1 : len(prompt) + len(response) - 1] = 1
        started = time.perf_counter()
        with torch.inference_mode():
            logits = model(
                tokens,
                images=images[0].patches,
                pooled_patches_idx=images[0].pooling,
                loss_masks=mask,
                response_logits_only=True,
            )
            logprobs = logits.reshape(-1, logits.shape[-1]).float().log_softmax(-1)
            selected = (
                logprobs.gather(1, torch.tensor(response, device="cuda")[:, None])
                .flatten()
                .cpu()
            )
        assert len(selected) == len(response) and torch.isfinite(selected).all()
        decode = record["result"]["meta_info"]["output_token_logprobs"][: len(response)]
        prefill = record["prefill_score"]["meta_info"]["input_token_logprobs"][
            -len(response) :
        ]
        assert [row[1] for row in decode] == response
        assert [row[1] for row in prefill] == response
        row = {
            "name": record["name"],
            "token_ids": response,
            "core_logprobs": selected.tolist(),
            "seconds": time.perf_counter() - started,
        }
        for name, values in [("decode", decode), ("prefill", prefill)]:
            candidate = torch.tensor([item[0] for item in values])
            assert torch.isfinite(candidate).all()
            error = candidate - selected
            row[name] = {
                "logprobs": candidate.tolist(),
                "mean_abs_error": error.abs().mean().item(),
                "signed_mean": error.mean().item(),
                "max_abs_error": error.abs().max().item(),
            }
        report["records"].append(row)
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    k: v
                    for k, v in row.items()
                    if k not in ("token_ids", "core_logprobs")
                },
                default=str,
            ),
            flush=True,
        )
    if len(report["records"]) != len(inference["input_manifest"]["pages"]):
        raise ValueError("Missing OCR page comparisons")
    report["completed"] = True
    report["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
