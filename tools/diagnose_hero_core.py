"""Locate native Core/HF drift before investigating serving cache/chunk differences."""

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from contextlib import ExitStack
from importlib import import_module
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from olmo_sglang.validation.layerwise import (
    capture_block,
    compare_captures,
    tensor_error,
)


def release():
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recurrent-hf-prefill", action="store_true")
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--logprob-atol", type=float, default=0.1)
    parser.add_argument("--save-activations", type=Path)
    args = parser.parse_args()
    if args.logprob_atol <= 0:
        parser.error("Log-probability threshold must be positive")
    if args.recurrent_hf_prefill:
        spec = importlib.util.spec_from_file_location(
            "hero_reference_settings", args.model / "inference_settings.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.install(recurrent=True)
    previous = (
        json.loads(args.reference_report.read_text()) if args.reference_report else None
    )
    prompts = (
        previous["prompt_ids"]
        if previous
        else [[5 + index % 30 for index in range(length)] for length in (16, 81)]
    )
    model, load_info = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        output_loading_info=True,
    )
    if load_info["missing_keys"] or load_info["unexpected_keys"]:
        raise ValueError(f"Incomplete HF state: {load_info}")
    model = model.cuda().eval()
    config = model.config
    if not config.use_peri_ln or config.use_rope:
        raise ValueError("This bounded diagnostic requires hero peri-norm and no RoPE")
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    hf_runs, hf_logits = [], []
    with torch.no_grad():
        for prompt in prompts:
            with ExitStack() as stack:
                captures = [
                    stack.enter_context(capture_block(block, backend="hf"))
                    for block in model.model.layers
                ]
                logits = model(
                    torch.tensor([prompt], device="cuda"), use_cache=False
                ).logits
                hf_logits.append(logits[0, -1].detach().cpu())
                hf_runs.append(captures)
    del model, logits
    release()
    core_config = import_module("olmo_core.config")
    attention = import_module("olmo_core.nn.attention")
    factory = import_module("olmo_core.nn.moe.v2.olmo3")
    experts = import_module("olmo_core.nn.moe.v2.routed_experts")
    model_config = factory.build_olmo3_moe_config_from_hf_config(
        config,
        dtype=core_config.DType.bfloat16,
        attention_backend=attention.AttentionBackendName.torch,
        router_aux_loss_weight=0.0,
        router_z_loss_weight=0.0,
    )
    model = model_config.build(init_device="meta")
    longest = max(map(len, prompts))
    model.init_weights(
        max_seq_len=longest,
        max_local_microbatch_size=longest,
        device=torch.device("cuda"),
    )
    factory.load_olmo3_moe_hf_state(model, config, state)
    del state
    model.eval()
    results = []
    with torch.no_grad():
        for prompt_index, prompt in enumerate(prompts):
            with ExitStack() as stack:
                captures = [
                    stack.enter_context(capture_block(block, backend="core"))
                    for block in model.blocks.values()
                ]
                logits = model(torch.tensor([prompt], device="cuda"))[0, -1].cpu()
            layers = []
            for index, block in enumerate(model.blocks.values()):
                reference = hf_runs[prompt_index][index]
                with capture_block(block, backend="core") as isolated:
                    block(reference["block.input"].cuda())
                layers.append(
                    {
                        "layer": index,
                        "type": config.layer_types[index],
                        "accumulated": compare_captures(captures[index], reference),
                        "same_hf_block_input": compare_captures(isolated, reference),
                    }
                )
            reference_logits = hf_logits[prompt_index].float()
            lp = tensor_error(
                logits.float().log_softmax(-1), reference_logits.log_softmax(-1)
            )
            results.append(
                {
                    "prompt_ids": prompt,
                    "layers": layers,
                    "last_logits": tensor_error(logits, hf_logits[prompt_index]),
                    "last_logprobs": lp,
                    "within_existing_threshold": lp["max_abs"] <= args.logprob_atol,
                }
            )
    report = {
        "diagnosis_only": True,
        "model": str(args.model),
        "dtype": "bfloat16",
        "attention": {"hf": "eager", "core": "torch_sdpa"},
        "kda": "shared_explicit_hf_recurrent_helper"
        if args.recurrent_hf_prefill
        else "native_chunk",
        "core_experts": "torch_grouped_mm"
        if experts.use_torch_grouped_mm()
        else "grouped_gemm",
        "environment": {
            name: os.getenv(name)
            for name in (
                "OLMO_USE_TORCH_GROUPED_MM",
                "OLMO_HF_MOE_REFERENCE_LOOP",
                "OLMO_HF_MOE_CORE_REFERENCE",
            )
        },
        "core_factory_sha256": hashlib.sha256(
            Path(factory.__file__).read_bytes()
        ).hexdigest(),
        "hf_source_sha256": {
            name: hashlib.sha256((args.model / name).read_bytes()).hexdigest()
            for name in ("configuration_olmo3moe.py", "modeling_olmo3moe.py")
        },
        "reference_report_sha256": hashlib.sha256(
            args.reference_report.read_bytes()
        ).hexdigest()
        if previous
        else None,
        "logprob_atol": args.logprob_atol,
        "results": results,
        "limitations": (
            "Initial forced prefixes only; same-input blocks isolate local drift, "
            "not individual kernels. No serving, EP, backward, optimizer, "
            "or threshold promotion."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.save_activations:
        torch.save(
            {"prompts": prompts, "hf": hf_runs, "hf_logits": hf_logits},
            args.save_activations,
        )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "max_logprob_error": max(
                    r["last_logprobs"]["max_abs"] for r in results
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
