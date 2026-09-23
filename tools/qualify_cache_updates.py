"""Compare populated TP1 caches after full weight updates with a fresh engine."""

import argparse
import json
from pathlib import Path

import sglang as sgl
from create_tiny_parity_checkpoint import build_checkpoint
from safetensors.torch import load_file, save_file

from olmo_sglang import register
from olmo_sglang.validation.cache_updates import compare_cache_outputs
from olmo_sglang.validation.weight_updates import update_weights_in_buckets


def create_engine(root, *, graphs):
    return sgl.Engine(
        model_path=str(root),
        trust_remote_code=True,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        tp_size=1,
        disable_radix_cache=False,
        mamba_radix_cache_strategy="extra_buffer",
        cuda_graph_backend_decode="full" if graphs else "disabled",
        cuda_graph_backend_prefill="disabled",
        cuda_graph_bs_decode=[1, 2, 4],
        attention_backend="triton",
        sampling_backend="pytorch",
        context_length=512,
        max_total_tokens=2048,
        max_running_requests=4,
        max_mamba_cache_size=32,
        chunked_prefill_size=64,
        mem_fraction_static=0.15,
        random_seed=42,
        skip_server_warmup=True,
        log_level="error",
    )


def sample(engine, prompts):
    outputs = engine.generate(
        input_ids=prompts,
        sampling_params={"temperature": 0, "max_new_tokens": 8, "ignore_eos": True},
        return_logprob=True,
    )
    return [
        {
            "tokens": out["output_ids"],
            "logprobs": [item[0] for item in out["meta_info"]["output_token_logprobs"]],
            "cached_tokens": out["meta_info"]["cached_tokens"],
        }
        for out in outputs
    ]


def exercise(engine, prompts):
    # The first cold request proves update invalidation before other requests
    # can repopulate the common prefix. Then exercise hits and mixed batching.
    return {
        "cold": sample(engine, prompts[:1]),
        "warm": sample(engine, prompts[:1]),
        "mixed": sample(engine, prompts),
    }


def run(root, profile, graphs):
    original = root / profile
    changed = root / (profile + "-updated")
    for path in (original, changed):
        if path.exists():
            raise ValueError(f"Fixture directory must not already exist: {path}")
        build_checkpoint(path, profile=profile, max_position_embeddings=512)
    weights = load_file(str(original / "model.safetensors"))
    updated = {
        name: (value.float() * 0.9 + 0.01).to(value.dtype)
        for name, value in weights.items()
    }
    save_file(updated, str(changed / "model.safetensors"))
    common = [5 + index % 30 for index in range(256)]
    prompts = [
        common + [5 + (index + branch * 7) % 30 for index in range(length - 256)]
        for branch, length in enumerate((300, 268, 332))
    ]
    report = {"profile": profile, "graphs": graphs, "tensor_count": len(weights)}
    engine = create_engine(original, graphs=graphs)
    try:
        report["before"] = exercise(engine, prompts)
        report["update_controls"] = update_weights_in_buckets(
            engine, list(updated.items())
        )
        report["updated"] = exercise(engine, prompts)
        report["restore_controls"] = update_weights_in_buckets(
            engine, list(weights.items())
        )
        report["restored"] = exercise(engine, prompts)
    finally:
        engine.shutdown()
    engine = create_engine(changed, graphs=graphs)
    try:
        report["fresh"] = exercise(engine, prompts)
    finally:
        engine.shutdown()
    report["updated_vs_fresh"] = compare_cache_outputs(
        report["updated"], report["fresh"]
    )
    report["restored_vs_original"] = compare_cache_outputs(
        report["restored"], report["before"]
    )
    report["changed_outputs"] = report["updated"]["cold"] != report["before"]["cold"]
    report["cache_lifecycle"] = all(
        report[phase]["cold"][0]["cached_tokens"] == 0
        and report[phase]["warm"][0]["cached_tokens"] >= 256
        and all(out["cached_tokens"] >= 256 for out in report[phase]["mixed"])
        for phase in ("before", "updated", "restored", "fresh")
    )
    report["passed"] = (
        report["changed_outputs"]
        and report["cache_lifecycle"]
        and all(
            report[comparison]["token_parity"]
            and report[comparison]["max_logprob_error"] <= 1e-6
            for comparison in ("updated_vs_fresh", "restored_vs_original")
        )
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiny-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eager", action="store_true")
    args = parser.parse_args()
    register()
    reports = []
    for profile in ("scaled-attention-hybrid-moe", "biased-sliding-hybrid-moe"):
        reports.append(run(args.tiny_dir, profile, graphs=not args.eager))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(reports, indent=2))
        print("CACHE_UPDATE_COMPLETE", profile, reports[-1]["passed"], flush=True)
    if not all(report["passed"] for report in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
