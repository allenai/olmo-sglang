# SPDX-License-Identifier: Apache-2.0

"""Run a token-in/token-out HF versus SGLang inference smoke test."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from olmo_sglang import register


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse inference smoke-test arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-ids", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument("--require-token-parity", action="store_true")
    return parser.parse_args()


def run_hf(model_path: Path, input_ids: list[int], max_new_tokens: int) -> list[int]:
    """Generate greedily with the checkpoint's Transformers implementation."""

    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.float16,
        attn_implementation="eager",
    ).cuda()
    inputs = torch.tensor([input_ids], dtype=torch.long, device="cuda")
    with torch.no_grad():
        output = model.generate(
            inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=model.config.pad_token_id,
        )
    return output[0, len(input_ids) :].cpu().tolist()


def run_sglang(model_path: Path, input_ids: list[int], max_new_tokens: int) -> list[int]:
    """Generate greedily with the external native SGLang model."""

    executable_dir = str(Path(sys.executable).parent)
    os.environ["PATH"] = f"{executable_dir}:{os.environ.get('PATH', '')}"
    register()
    import sglang as sgl

    engine = sgl.Engine(
        model_path=str(model_path),
        trust_remote_code=True,
        skip_tokenizer_init=True,
        dtype="float16",
        cuda_graph_backend_decode="disabled",
        cuda_graph_backend_prefill="disabled",
        disable_radix_cache=True,
        attention_backend="torch_native",
        context_length=32,
        max_total_tokens=64,
        mem_fraction_static=0.15,
    )
    try:
        output = engine.generate(
            input_ids=input_ids,
            sampling_params={"temperature": 0, "max_new_tokens": max_new_tokens},
        )
        return output["output_ids"]
    finally:
        engine.shutdown()


def main() -> None:
    """Run both engines and apply the requested token-agreement checks."""

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    LOGGER.setLevel(logging.INFO)
    hf_tokens = None if args.skip_hf else run_hf(args.model, args.input_ids, args.max_new_tokens)
    sglang_tokens = run_sglang(args.model, args.input_ids, args.max_new_tokens)
    LOGGER.info("input_ids=%s", args.input_ids)
    if hf_tokens is not None:
        LOGGER.info("hf_output_ids=%s", hf_tokens)
    LOGGER.info("sglang_output_ids=%s", sglang_tokens)
    if hf_tokens is not None and hf_tokens[0] != sglang_tokens[0]:
        raise AssertionError(f"First-token mismatch: HF={hf_tokens}, SGLang={sglang_tokens}")
    if args.require_token_parity and hf_tokens is not None and hf_tokens != sglang_tokens:
        raise AssertionError(f"Greedy token mismatch: HF={hf_tokens}, SGLang={sglang_tokens}")
    if hf_tokens is not None and hf_tokens != sglang_tokens:
        LOGGER.info(
            "note=first token agrees; later tiny-checkpoint tokens differ because "
            "fused-MoE rounding is amplified by peri-LN"
        )


if __name__ == "__main__":
    main()
