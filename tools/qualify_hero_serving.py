import argparse
import gc
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import sglang as sgl
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM

from olmo_sglang import register
from olmo_sglang.validation.core_reference import evaluate_core_reference


def main():
    parser = argparse.ArgumentParser(
        description="Bounded BF16 hero HF / TP1 SGLang serving qualification"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--model", type=Path, help="Existing HF checkpoint; never modified"
    )
    source.add_argument(
        "--tiny-dir", type=Path, help="Create a NEW tiny hero fixture here"
    )
    parser.add_argument(
        "--hf-source",
        type=Path,
        help="Hero HF Python source directory, required for a new tiny fixture",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--core-reference",
        action="store_true",
        help="Also compare native Core factory/import using torch attention and no EP",
    )
    parser.add_argument("--core-logprob-atol", type=float, default=0.05)
    parser.add_argument(
        "--check-live-update",
        action="store_true",
        help="For --tiny-dir: compare live updates with a fresh changed checkpoint",
    )
    parser.add_argument("--logprob-atol", type=float, default=0.1)
    parser.add_argument("--require-token-parity", action="store_true")
    parser.add_argument(
        "--recurrent-hf-prefill",
        action="store_true",
        help="Use HF recurrent FLA prefill; keeps native BF16 linear/attention",
    )
    args = parser.parse_args()
    if args.tiny_dir is not None:
        if args.tiny_dir.exists():
            parser.error("--tiny-dir must not already exist")
        if args.hf_source is None:
            parser.error("--tiny-dir requires --hf-source")
    if args.check_live_update and args.tiny_dir is None:
        parser.error(
            "--check-live-update requires --tiny-dir to bound checkpoint copying"
        )
    ROOT = args.model or args.tiny_dir
    HF = args.hf_source or ROOT
    if args.tiny_dir is not None:
        spec = importlib.util.spec_from_file_location(
            "generator", Path(__file__).with_name("create_tiny_parity_checkpoint.py")
        )
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        generator.build_checkpoint(
            ROOT, profile="hero-hybrid-moe", max_position_embeddings=256
        )
        for name in (
            "configuration_olmo3moe.py",
            "modeling_olmo3moe.py",
            "inference_settings.py",
        ):
            shutil.copyfile(HF / name, ROOT / name)
        config = json.loads((ROOT / "config.json").read_text())
        config.update(
            model_type="olmo3moe",
            auto_map={
                "AutoConfig": "configuration_olmo3moe.Olmo3MoeConfig",
                "AutoModelForCausalLM": "modeling_olmo3moe.Olmo3MoeForCausalLM",
            },
        )
        (ROOT / "config.json").write_text(json.dumps(config))
    config = json.loads((ROOT / "config.json").read_text())
    if not config.get("qk_norm_per_head_gains") or not config.get("scalable_softmax"):
        parser.error("Both hero attention flags must be enabled")
    if args.core_reference and args.core_logprob_atol <= 0:
        parser.error("--core-logprob-atol must be positive")
    if args.logprob_atol <= 0:
        parser.error("--logprob-atol must be positive")
    if args.recurrent_hf_prefill:
        spec = importlib.util.spec_from_file_location(
            "hf_inference_settings", HF / "inference_settings.py"
        )
        settings = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(settings)
        settings.install(recurrent=True)
    prompts = [[5 + i % 30 for i in range(n)] for n in (16, 81)]
    model, load_info = AutoModelForCausalLM.from_pretrained(
        ROOT,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        output_loading_info=True,
    )
    assert not load_info["missing_keys"] and not load_info["unexpected_keys"], load_info
    model = model.cuda().eval()
    references = []
    with torch.no_grad():
        for prompt in prompts:
            seq = torch.tensor([prompt], device="cuda")
            tokens = []
            distributions = []
            for _ in range(4):
                logits = model(seq, use_cache=False).logits[0, -1].float()
                distributions.append(torch.log_softmax(logits, -1).cpu().tolist())
                token = int(logits.argmax())
                tokens.append(token)
                seq = torch.cat([seq, torch.tensor([[token]], device="cuda")], dim=-1)
            references.append({"tokens": tokens, "logprobs": distributions})
    core_comparison = None
    if args.core_reference:
        native_hf_config = model.config
        native_hf_state = {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    if args.core_reference:
        core_comparison = evaluate_core_reference(
            hf_config=native_hf_config,
            hf_state=native_hf_state,
            prompts=prompts,
            references=references,
            logprob_atol=args.core_logprob_atol,
            recurrent_kda=args.recurrent_hf_prefill,
        )
        del native_hf_state
    register()
    report = {
        "model_path": str(ROOT),
        "model_config": config,
        "prompt_ids": prompts,
        "core_reference": core_comparison,
        "hf_source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("configuration_olmo3moe.py", "modeling_olmo3moe.py")
        },
        "hf_load_info": load_info,
        "reference": references,
        "modes": [],
        "logprob_atol": args.logprob_atol,
        "recurrent_hf_prefill": args.recurrent_hf_prefill,
    }
    changed_root = None
    updated_outputs = None
    before_update_outputs = None
    update_tensors = []
    if args.check_live_update:
        changed_root = ROOT.with_name(ROOT.name + "-updated")
        shutil.copytree(ROOT, changed_root)
        weights = load_file(str(changed_root / "model.safetensors"))
        for name, value in weights.items():
            if name.endswith(".q_norm.weight"):
                weights[name] = value * 0.4
            elif name.endswith(".k_norm.weight"):
                weights[name] = value * 1.3
            elif name.endswith(".ssmax_scale"):
                weights[name] = value * -1.7
            else:
                continue
            update_tensors.append((name, weights[name]))
        save_file(weights, str(changed_root / "model.safetensors"))
    modes = [(False, 128, ROOT), (True, 32, ROOT)]
    if changed_root is not None:
        modes.append((True, 32, changed_root))
    for graphs, chunk, serving_root in modes:
        engine = sgl.Engine(
            model_path=str(serving_root),
            trust_remote_code=True,
            skip_tokenizer_init=True,
            dtype="bfloat16",
            tp_size=1,
            disable_radix_cache=True,
            disable_overlap_schedule=True,
            cuda_graph_backend_decode="full" if graphs else "disabled",
            cuda_graph_backend_prefill="disabled",
            cuda_graph_bs_decode=[1, 2, 4],
            attention_backend="triton",
            sampling_backend="pytorch",
            context_length=256,
            max_total_tokens=512,
            max_running_requests=4,
            max_mamba_cache_size=8,
            chunked_prefill_size=chunk,
            mem_fraction_static=0.15,
            skip_server_warmup=True,
        )
        try:
            outputs = engine.generate(
                input_ids=prompts,
                sampling_params={
                    "temperature": 0,
                    "max_new_tokens": 4,
                    "ignore_eos": True,
                },
                return_logprob=True,
                top_logprobs_num=min(20, config["vocab_size"]),
            )
            if serving_root == changed_root:
                actual_logprobs = [
                    item[0]
                    for out in outputs
                    for item in out["meta_info"]["output_token_logprobs"]
                ]
                updated_logprobs = [
                    item[0]
                    for out in updated_outputs
                    for item in out["meta_info"]["output_token_logprobs"]
                ]
                before_logprobs = [
                    item[0]
                    for out in before_update_outputs
                    for item in out["meta_info"]["output_token_logprobs"]
                ]
                report["live_update"] = {
                    "parameter_names": [name for name, _ in update_tensors],
                    "fresh_outputs": outputs,
                    "updated_outputs": updated_outputs,
                    "token_parity": [out["output_ids"] for out in outputs]
                    == [out["output_ids"] for out in updated_outputs],
                    "max_logprob_error": max(
                        abs(a - b) for a, b in zip(actual_logprobs, updated_logprobs)
                    ),
                    "changed_logprobs": before_logprobs != updated_logprobs,
                }
                continue
            forced = []
            for prompt, ref in zip(prompts, references):
                samples = []
                for step in range(4):
                    out = engine.generate(
                        input_ids=prompt + ref["tokens"][:step],
                        sampling_params={
                            "temperature": 0,
                            "max_new_tokens": 1,
                            "ignore_eos": True,
                        },
                        return_logprob=True,
                        top_logprobs_num=min(20, config["vocab_size"]),
                    )
                    actual = {
                        int(t[1]): t[0]
                        for t in out["meta_info"]["output_top_logprobs"][0]
                    }
                    max_error = max(
                        abs(value - ref["logprobs"][step][token])
                        for token, value in actual.items()
                    )
                    samples.append(
                        {"token": out["output_ids"][0], "max_logprob_error": max_error}
                    )
                forced.append(samples)
            report["modes"].append(
                {"graphs": graphs, "chunk": chunk, "outputs": outputs, "forced": forced}
            )
            if args.check_live_update and graphs:
                before_update_outputs = outputs
                responses = [engine.begin_weight_update()]
                try:
                    for index, tensor in enumerate(update_tensors):
                        responses.append(
                            engine.update_weights_from_tensor(
                                [tensor], flush_cache=index == len(update_tensors) - 1
                            )
                        )
                finally:
                    responses.append(engine.end_weight_update())
                report["update_responses"] = responses
                updated_outputs = engine.generate(
                    input_ids=prompts,
                    sampling_params={
                        "temperature": 0,
                        "max_new_tokens": 4,
                        "ignore_eos": True,
                    },
                    return_logprob=True,
                )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, default=list))
            print(
                "HERO_MODE",
                graphs,
                chunk,
                [(o["output_ids"], r["tokens"]) for o, r in zip(outputs, references)],
                forced,
                flush=True,
            )
        finally:
            engine.shutdown()
    decode_errors = []
    for mode in report["modes"]:
        for output, reference in zip(mode["outputs"], references):
            for step, value in enumerate(output["meta_info"]["output_token_logprobs"]):
                if output["output_ids"][:step] == reference["tokens"][:step]:
                    decode_errors.append(
                        abs(value[0] - reference["logprobs"][step][value[1]])
                    )
    report["max_cached_decode_logprob_error"] = max(decode_errors)
    errors = decode_errors + [
        step["max_logprob_error"]
        for mode in report["modes"]
        for prompt in mode["forced"]
        for step in prompt
    ]
    token_parity = all(
        output["output_ids"] == reference["tokens"]
        for mode in report["modes"]
        for output, reference in zip(mode["outputs"], references)
    )
    mode_parity = [out["output_ids"] for out in report["modes"][0]["outputs"]] == [
        out["output_ids"] for out in report["modes"][1]["outputs"]
    ]
    report.update(
        max_logprob_error=max(errors),
        token_parity=token_parity,
        prefill_decode_graph_token_parity=mode_parity,
    )
    report["passed"] = (
        max(errors) <= args.logprob_atol
        and mode_parity
        and (token_parity or not args.require_token_parity)
    )
    if args.core_reference:
        report["passed"] &= core_comparison["passed"] and (
            core_comparison["token_parity"] or not args.require_token_parity
        )
    if args.check_live_update:
        update = report["live_update"]
        report["passed"] &= (
            update["token_parity"]
            and update["max_logprob_error"] <= 1e-6
            and update["changed_logprobs"]
        )
    args.output.write_text(json.dumps(report, indent=2, default=list))
    print("HERO_COMPLETE", args.output, "passed=", report["passed"], flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
