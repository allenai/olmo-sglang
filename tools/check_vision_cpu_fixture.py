"""Validate the extracted 64K Jason checkpoint visual stack on CPU."""

import argparse
import json
import time
from pathlib import Path

import torch
from olmo_core.nn.vision.molmo2_tokens import prepare_molmo2_tokenizer
from safetensors import safe_open
from transformers import AutoTokenizer, PretrainedConfig

from olmo_sglang.vision.config import validate_image_tokenizer, validate_vision_config
from olmo_sglang.vision.model import OlmoVisionStack


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "fixture",
        help="Extracted fixture directory described in docs/vision-checkpoints.md",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    root = Path(args.fixture)
    config = json.loads((root / "hf-step5000-alt/config.json").read_text())
    try:
        validate_vision_config(config)
    except ValueError as error:
        rejected = str(error)
    else:
        raise AssertionError("Real truncated export was not rejected")
    original = json.loads((root / "step5000/olmo_core_config.json").read_text())
    config["text_config"]["vocab_size"] = original["model"]["lm"]["vocab_size"]
    tokenizer = AutoTokenizer.from_pretrained(
        root / "hf-step5000-alt",
        config=PretrainedConfig(),
        local_files_only=True,
        trust_remote_code=False,
    )
    prepare_molmo2_tokenizer(
        tokenizer, model_vocab_size=config["text_config"]["vocab_size"]
    )
    token_ids = validate_image_tokenizer(config, tokenizer)
    start = time.perf_counter()
    model = OlmoVisionStack(config, dtype=torch.float32).eval()
    with safe_open(
        root / "vision_and_embeddings.safetensors", framework="pt", device="cpu"
    ) as handle:
        names = [
            name for name in handle.keys() if name.startswith(("vision.", "connector."))
        ]
        loaded = model.load_weights((name, handle.get_tensor(name)) for name in names)
        checks = []
        for name, parameter in model.named_parameters():
            source_name = name.replace("vision_tower.", "vision.", 1)
            checks.append(
                torch.equal(parameter, handle.get_tensor(source_name).float())
            )
        embedding = handle.get_tensor("lm.embeddings.weight")
        head = handle.get_tensor("lm.lm_head.w_out.weight")
        shapes = {"embedding": list(embedding.shape), "head": list(head.shape)}
        image_row_norm = float(embedding[token_ids["<im_patch>"]].float().norm())
    assert all(checks)
    load_seconds = time.perf_counter() - start
    torch.manual_seed(63)
    images = torch.randn(1, 1, 729, 588)
    pools = torch.tensor([[[0, 1, 27, 28], [726, 727, 728, -1]]])
    with torch.inference_mode():
        start = time.perf_counter()
        actual = model.encode_images(images, pools)
        forward_seconds = time.perf_counter() - start
        hidden = model.vision_tower(images.reshape(1, 729, 588))
        expected = model.connector(torch.cat([hidden[24], hidden[18]], dim=-1), pools)
        changed = model.encode_images(torch.zeros_like(images), pools)
    assert torch.equal(actual, expected)
    assert torch.isfinite(actual).all()
    assert not torch.equal(actual, changed)
    report = {
        "fixture_source": str(root),
        "torch": torch.__version__,
        "dtype": "float32 from BF16 checkpoint",
        "threads": torch.get_num_threads(),
        "rejected_original_export": rejected,
        "restored_config_vocab_size": config["text_config"]["vocab_size"],
        "resolved_image_token_ids": token_ids,
        "vocabulary_shapes": shapes,
        "image_patch_embedding_norm": image_row_norm,
        "loaded_visual_tensors": len(loaded),
        "all_visual_tensors_equal_source": all(checks),
        "load_seconds": load_seconds,
        "feature_forward_seconds": forward_seconds,
        "output_shape": list(actual.shape),
        "max_feature_error": float((actual - expected).abs().max()),
        "zero_image_changes_features": True,
        "scope": "Actual vision+connector weights; synthetic patch inputs; no LM generation or image accuracy claim",
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
