"""Inspect or restore a language-only vision export without loading its experts."""

import argparse
import hashlib
import json
import shutil
import struct
from pathlib import Path

import torch
from olmo_core.nn.vision.molmo2_tokens import prepare_molmo2_tokenizer
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer, PretrainedConfig

from olmo_sglang.vision.config import validate_image_tokenizer, validate_vision_config

VOCAB_TENSORS = {
    "language_model.model.embed_tokens.weight": "lm.embeddings.weight",
    "language_model.lm_head.weight": "lm.lm_head.w_out.weight",
}


def tensor_files(directory: Path) -> dict[str, Path]:
    """Resolve a single-file or indexed HF checkpoint, validating its index."""
    index = directory / "model.safetensors.index.json"
    if index.exists():
        mapping = json.loads(index.read_text())["weight_map"]
        result = {}
        for key, name in mapping.items():
            if Path(name).name != name:
                raise ValueError("Checkpoint shard names must be filenames")
            result[key] = directory / name
        by_file = {}
        for key, path in result.items():
            by_file.setdefault(path, set()).add(key)
        for path, keys in by_file.items():
            with safe_open(path, framework="pt", device="cpu") as handle:
                if keys != set(handle.keys()):
                    raise ValueError(
                        f"Index and actual tensor names differ in {path.name}"
                    )
        return result
    path = directory / "model.safetensors"
    with safe_open(path, framework="pt", device="cpu") as handle:
        return {key: path for key in handle.keys()}


def _copy_without(source: Path, destination: Path, excluded: set[str]) -> None:
    """Rewrite an affected shard in bounded memory, excluding replaced tensors.

    SGLang's iterator visits every key in every shard, not just keys assigned to
    that shard by the index. Keeping stale vocabulary tensors would double-load
    them, so index-only overlays are not sufficient.
    """
    with source.open("rb") as src:
        header_size = struct.unpack("<Q", src.read(8))[0]
        header = json.loads(src.read(header_size))
        data_start = 8 + header_size
        selected = {
            k: v for k, v in header.items() if k != "__metadata__" and k not in excluded
        }
        rewritten, offset = {}, 0
        if "__metadata__" in header:
            rewritten["__metadata__"] = header["__metadata__"]
        for name, tensor in selected.items():
            length = tensor["data_offsets"][1] - tensor["data_offsets"][0]
            rewritten[name] = {**tensor, "data_offsets": [offset, offset + length]}
            offset += length
        raw = json.dumps(rewritten).encode()
        raw += b" " * (-len(raw) % 8)
        with destination.open("xb") as dst:
            dst.write(struct.pack("<Q", len(raw)))
            dst.write(raw)
            for tensor in selected.values():
                start, end = tensor["data_offsets"]
                src.seek(data_start + start)
                remaining = end - start
                while remaining:
                    chunk = src.read(min(remaining, 8 * 1024 * 1024))
                    if not chunk:
                        raise ValueError(
                            "Short safetensors read while rewriting a shard"
                        )
                    dst.write(chunk)
                    remaining -= len(chunk)


def audit_checkpoint(directory: str | Path) -> dict:
    """Validate architecture, tokenizer, vocabulary shapes and component presence."""
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    validate_vision_config(config)
    tokenizer = AutoTokenizer.from_pretrained(
        directory,
        config=PretrainedConfig(),
        local_files_only=True,
        trust_remote_code=False,
    )
    token_ids = validate_image_tokenizer(config, tokenizer)
    files = tensor_files(directory)
    expected_shape = [
        config["text_config"]["vocab_size"],
        config["text_config"]["hidden_size"],
    ]
    for name in VOCAB_TENSORS:
        if name not in files:
            raise ValueError(f"Missing vocabulary tensor {name}")
        with safe_open(files[name], framework="pt", device="cpu") as handle:
            if handle.get_slice(name).get_shape() != expected_shape:
                raise ValueError(f"Vocabulary tensor shape differs from config: {name}")
    prefixes = {key.split(".")[0] for key in files}
    if prefixes != {"language_model", "vision_tower", "connector"}:
        raise ValueError(
            f"Expected LM, vision and connector tensors; found {sorted(prefixes)}"
        )
    return {
        "tensor_count": len(files),
        "image_tokens": token_ids,
        "vocab_size": expected_shape[0],
    }


def restore_vocabulary(hf_directory, core_directory, output_directory) -> dict:
    """Restore trained rows into a new local overlay, never modify the inputs.

    Affected shards are streamed to new files without their old embeddings/head;
    untouched shards are referenced with symlinks. Keep the source accessible.
    The full Core output vocabulary is retained, making normalization explicit.
    """
    source, core, output = map(Path, (hf_directory, core_directory, output_directory))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    config = json.loads((source / "config.json").read_text())
    original = json.loads((core / "olmo_core_config.json").read_text())["model"]
    if original["image_patch_token_id"] != config["image_patch_token_id"]:
        raise ValueError("Source checkpoints disagree on image_patch_token_id")
    for outer, inner in (
        ("vision_config", "vision"),
        ("connector_config", "connector"),
    ):
        if config[outer] != original[inner]:
            raise ValueError(f"Source checkpoints disagree on {outer}")
    if config["vit_layers"] != original["vit_layers"]:
        raise ValueError("Source checkpoints disagree on vit_layers")
    files = tensor_files(source)
    restored = {}
    hashes = {}
    with safe_open(core / "model.safetensors", framework="pt", device="cpu") as handle:
        for hf_name, core_name in VOCAB_TENSORS.items():
            complete = handle.get_tensor(core_name)
            with safe_open(files[hf_name], framework="pt", device="cpu") as hf_handle:
                current = hf_handle.get_tensor(hf_name)
            if (
                complete.dtype != current.dtype
                or complete.ndim != 2
                or current.shape[1:] != complete.shape[1:]
                or current.shape[0] > complete.shape[0]
                or not torch.equal(current, complete[: current.shape[0]])
            ):
                raise ValueError(
                    f"HF tensor is not an exact prefix of the Core tensor: {hf_name}"
                )
            restored[hf_name] = complete
            hashes[hf_name] = hashlib.sha256(
                complete.view(torch.uint8).numpy().tobytes()
            ).hexdigest()
        if len({tuple(t.shape) for t in restored.values()}) != 1:
            raise ValueError(
                "Core input/output vocabulary tables must have matching shapes"
            )
        vocab_size, hidden_size = next(iter(restored.values())).shape
        if (
            original["lm"]["vocab_size"] != vocab_size
            or config["text_config"]["hidden_size"] != hidden_size
        ):
            raise ValueError("Vocabulary tensor dimensions disagree with source config")
        config["text_config"]["vocab_size"] = vocab_size
        validate_vision_config(config)
        tokenizer = AutoTokenizer.from_pretrained(
            source,
            config=PretrainedConfig(),
            local_files_only=True,
            trust_remote_code=False,
        )
        prepare_molmo2_tokenizer(tokenizer, model_vocab_size=vocab_size)
        token_ids = validate_image_tokenizer(config, tokenizer)
        # All source/shape/token checks precede output creation.
        output.mkdir(parents=True)
        for file in source.iterdir():
            if file.is_file() and file.suffix in (".py", ".jinja"):
                shutil.copyfile(file, output / file.name)
        if (source / "generation_config.json").exists():
            shutil.copyfile(
                source / "generation_config.json", output / "generation_config.json"
            )
        tokenizer.save_pretrained(output)
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        save_file(restored, output / "restored-vocabulary.safetensors")
    weight_map = {}
    shard_names = {}
    rewritten_shards = []
    symlink_shards = []
    affected = {files[name] for name in restored}
    for name, path in files.items():
        if name in restored:
            weight_map[name] = "restored-vocabulary.safetensors"
        else:
            if path not in shard_names:
                target = f"source-{len(shard_names):05d}.safetensors"
                if path in affected:
                    _copy_without(path, output / target, set(restored))
                    rewritten_shards.append(target)
                else:
                    (output / target).symlink_to(path.resolve())
                    symlink_shards.append(target)
                shard_names[path] = target
            weight_map[name] = shard_names[path]
    (output / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}, indent=2) + "\n"
    )
    receipt = {
        "source_hf": str(source.resolve()),
        "source_core": str(core.resolve()),
        "output_vocab_policy": "full_core_vocabulary",
        "image_tokens": token_ids,
        "restored_tensor_sha256": hashes,
        "rewritten_shards": rewritten_shards,
        "symlink_shards": symlink_shards,
    }
    receipt["audit"] = audit_checkpoint(output)
    (output / "vision-restoration.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("checkpoint")
    restore = commands.add_parser("restore-vocabulary")
    restore.add_argument("--hf", required=True)
    restore.add_argument("--core", required=True)
    restore.add_argument("--output", required=True)
    args = parser.parse_args()
    result = (
        audit_checkpoint(args.checkpoint)
        if args.command == "audit"
        else restore_vocabulary(args.hf, args.core, args.output)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
