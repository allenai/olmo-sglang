# SPDX-License-Identifier: Apache-2.0

"""Compare exact greedy OLMo KDA output between TP=1 and TP>1."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from olmo_sglang import register

LOGGER = logging.getLogger(__name__)


def _create_engine(
    model_path: Path,
    *,
    tp_size: int,
    context_length: int,
    mem_fraction_static: float,
) -> Any:
    import sglang as sgl

    return sgl.Engine(
        model_path=str(model_path),
        trust_remote_code=True,
        skip_tokenizer_init=True,
        dtype="auto",
        tp_size=tp_size,
        disable_radix_cache=True,
        cuda_graph_backend_decode="disabled",
        cuda_graph_backend_prefill="disabled",
        attention_backend="triton",
        page_size=1,
        context_length=context_length,
        max_total_tokens=max(256, context_length * 2),
        max_mamba_cache_size=4,
        max_running_requests=2,
        mem_fraction_static=mem_fraction_static,
    )


def _generate(engine: Any, input_ids: list[int], max_new_tokens: int) -> dict[str, Any]:
    return engine.generate(
        input_ids=input_ids,
        sampling_params={
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        },
        return_logprob=True,
    )


def _chosen_token_logprobs(result: dict[str, Any]) -> list[float]:
    """Extract chosen-token log probabilities from an SGLang response."""

    try:
        values = result["meta_info"]["output_token_logprobs"]
        return [float(value[0]) for value in values]
    except (KeyError, TypeError, ValueError) as error:
        raise AssertionError(
            "SGLang response did not contain output token log probabilities"
        ) from error


def run_tp_smoke(
    model_path: Path,
    input_ids: list[int],
    max_new_tokens: int,
    *,
    tp_size: int,
    context_length: int,
    mem_fraction_static: float = 0.25,
) -> dict[str, Any]:
    """Require exact greedy token parity between TP=1 and ``tp_size``."""

    if tp_size <= 1:
        raise ValueError("tp_size must be greater than one")
    register()
    baseline_engine = _create_engine(
        model_path,
        tp_size=1,
        context_length=context_length,
        mem_fraction_static=mem_fraction_static,
    )
    try:
        baseline = _generate(baseline_engine, input_ids, max_new_tokens)
    finally:
        baseline_engine.shutdown()

    sharded_engine = _create_engine(
        model_path,
        tp_size=tp_size,
        context_length=context_length,
        mem_fraction_static=mem_fraction_static,
    )
    try:
        sharded = _generate(sharded_engine, input_ids, max_new_tokens)
    finally:
        sharded_engine.shutdown()

    if sharded["output_ids"] != baseline["output_ids"]:
        raise AssertionError(
            "tensor parallelism changed greedy output: "
            f"tp1={baseline['output_ids']}, tp{tp_size}={sharded['output_ids']}"
        )
    baseline_logprobs = _chosen_token_logprobs(baseline)
    sharded_logprobs = _chosen_token_logprobs(sharded)
    if len(sharded_logprobs) != len(baseline_logprobs):
        raise AssertionError(
            "tensor parallelism changed the number of output token log probabilities: "
            f"tp1={len(baseline_logprobs)}, tp{tp_size}={len(sharded_logprobs)}"
        )
    logprob_abs_diffs = [
        abs(tp1 - sharded)
        for tp1, sharded in zip(baseline_logprobs, sharded_logprobs, strict=True)
    ]
    return {
        "input_ids": input_ids,
        "output_ids": baseline["output_ids"],
        "baseline_tp_size": 1,
        "comparison_tp_size": tp_size,
        "token_parity": True,
        "chosen_token_logprob_max_abs_diff": max(logprob_abs_diffs, default=0.0),
        "chosen_token_logprob_mean_abs_diff": (
            sum(logprob_abs_diffs) / len(logprob_abs_diffs)
            if logprob_abs_diffs
            else 0.0
        ),
    }


def parse_args() -> argparse.Namespace:
    """Parse tensor-parallel smoke-test arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--input-ids", type=int, nargs="+")
    parser.add_argument("--prompt-length", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--mem-fraction-static", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    """Run matched TP=1 and sharded engines and emit a JSON report."""

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    LOGGER.setLevel(logging.INFO)
    args = parse_args()
    input_ids = args.input_ids
    if input_ids is None:
        if args.prompt_length < 1:
            raise ValueError("prompt-length must be positive")
        input_ids = [5 + index % 4 for index in range(args.prompt_length)]
    if len(input_ids) + args.max_new_tokens > args.context_length:
        raise ValueError("prompt plus requested output exceeds context-length")
    report = run_tp_smoke(
        args.model,
        input_ids,
        args.max_new_tokens,
        tp_size=args.tp_size,
        context_length=args.context_length,
        mem_fraction_static=args.mem_fraction_static,
    )
    LOGGER.info("%s", json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
