# SPDX-License-Identifier: Apache-2.0

"""Run local tokenizer, reference-model, and SGLang parity diagnostics."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from olmo_sglang import register
from olmo_sglang.validation.reference import ToyReferenceForCausalLM, trace_summary

LOGGER = logging.getLogger(__name__)
DEFAULT_PROMPT = "What is the capital of France?"


def _render_prompt(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _run_sglang(
    *,
    model_path: Path,
    input_ids: list[int],
    max_new_tokens: int,
    dtype: str,
    top_logprobs_num: int,
    forced_prefix_tokens: list[int],
) -> tuple[list[int], dict[str, Any]]:
    executable_dir = str(Path(sys.executable).parent)
    os.environ["PATH"] = f"{executable_dir}:{os.environ.get('PATH', '')}"
    register()
    import sglang as sgl

    engine = sgl.Engine(
        model_path=str(model_path),
        trust_remote_code=True,
        skip_tokenizer_init=True,
        dtype=dtype,
        tp_size=1,
        disable_radix_cache=True,
        cuda_graph_backend_decode="disabled",
        cuda_graph_backend_prefill="disabled",
        attention_backend="torch_native",
        context_length=64,
        max_total_tokens=128,
        mem_fraction_static=0.15,
        skip_server_warmup=True,
    )
    try:
        output = engine.generate(
            input_ids=input_ids,
            sampling_params={"temperature": 0, "max_new_tokens": max_new_tokens},
            return_logprob=True,
            top_logprobs_num=top_logprobs_num,
        )
        meta_info = output.get("meta_info", {})
        logprob_diagnostics = {
            "output_token_logprobs": meta_info.get("output_token_logprobs"),
            "output_top_logprobs": meta_info.get("output_top_logprobs"),
        }
        forced_prefill_steps = []
        for step in range(len(forced_prefix_tokens)):
            forced_output = engine.generate(
                input_ids=input_ids + forced_prefix_tokens[:step],
                sampling_params={"temperature": 0, "max_new_tokens": 1},
                return_logprob=True,
                top_logprobs_num=top_logprobs_num,
            )
            forced_meta = forced_output.get("meta_info", {})
            forced_prefill_steps.append(
                {
                    "output_id": forced_output["output_ids"][0],
                    "output_token_logprob": forced_meta.get(
                        "output_token_logprobs", [None]
                    )[0],
                    "output_top_logprobs": forced_meta.get(
                        "output_top_logprobs", [None]
                    )[0],
                }
            )
        logprob_diagnostics["forced_prefill_steps"] = forced_prefill_steps
        return output["output_ids"], logprob_diagnostics
    finally:
        engine.shutdown()


def evaluate_local_parity(
    *,
    model_path: Path,
    prompt: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Compare the independent local reference against embedded SGLang."""

    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("local SGLang parity requires a CUDA GPU")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    rendered_prompt = _render_prompt(tokenizer, prompt)
    input_ids = tokenizer(
        rendered_prompt,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to("cuda")

    reference = ToyReferenceForCausalLM.from_pretrained(
        model_path,
        device=torch.device("cuda"),
    )
    with torch.no_grad():
        logits, trace = reference(input_ids, return_trace=True)
        reference_tokens: list[int] = []
        reference_top_logprobs: list[list[list[float | int]]] = []
        sequence = input_ids
        top_logprobs_num = min(config["vocab_size"], 20)
        for _ in range(max_new_tokens):
            step_logits, _ = reference(sequence)
            step_logprobs = torch.log_softmax(step_logits[0, -1].float(), dim=-1)
            values, indices = step_logprobs.topk(top_logprobs_num)
            reference_top_logprobs.append(
                [
                    [float(logprob), int(token_id)]
                    for logprob, token_id in zip(values.tolist(), indices.tolist())
                ]
            )
            next_token = int(indices[0].item())
            reference_tokens.append(next_token)
            sequence = torch.cat(
                (
                    sequence,
                    torch.tensor([[next_token]], device=sequence.device),
                ),
                dim=1,
            )
    if trace is None:
        raise RuntimeError("reference trace was not produced")
    reference_next_token = int(logits[0, -1].argmax().item())
    summary = trace_summary(trace)
    del reference, logits, trace
    gc.collect()
    torch.cuda.empty_cache()

    sglang_tokens, sglang_logprobs = _run_sglang(
        model_path=model_path,
        input_ids=input_ids[0].cpu().tolist(),
        max_new_tokens=max_new_tokens,
        dtype=config["dtype"],
        top_logprobs_num=top_logprobs_num,
        forced_prefix_tokens=reference_tokens,
    )
    return {
        "model_path": str(model_path),
        "prompt": prompt,
        "rendered_prompt": rendered_prompt,
        "input_ids": input_ids[0].cpu().tolist(),
        "tokenizer_class": type(tokenizer).__name__,
        "reference_next_token": reference_next_token,
        "reference_output_ids": reference_tokens,
        "reference_top_logprobs": reference_top_logprobs,
        "reference_text": tokenizer.decode(reference_tokens, skip_special_tokens=False),
        "sglang_output_ids": sglang_tokens,
        "sglang_logprobs": sglang_logprobs,
        "sglang_text": tokenizer.decode(sglang_tokens, skip_special_tokens=False),
        "first_token_agreement": reference_tokens[0] == sglang_tokens[0],
        "token_agreement": reference_tokens == sglang_tokens,
        "reference_trace": summary,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-parity", action="store_true")
    return parser


def main() -> None:
    """Run local parity diagnostics and optionally require exact greedy tokens."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parser().parse_args()
    report = evaluate_local_parity(
        model_path=args.model,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
    )
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    if args.require_parity and not report["token_agreement"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
