# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare greedy OLMo KDA inference with breadth-one NGRAM speculation."""

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
    context_length: int,
    speculative: bool,
    mem_fraction_static: float,
) -> Any:
    """Create one eager embedded engine for the matched greedy A/B."""

    import sglang as sgl

    kwargs: dict[str, Any] = {}
    if speculative:
        kwargs.update(
            speculative_algorithm="NGRAM",
            speculative_num_draft_tokens=4,
            speculative_ngram_min_bfs_breadth=1,
            speculative_ngram_max_bfs_breadth=1,
        )
    return sgl.Engine(
        model_path=str(model_path),
        trust_remote_code=True,
        skip_tokenizer_init=True,
        dtype="auto",
        cuda_graph_backend_decode="disabled",
        cuda_graph_backend_prefill="disabled",
        context_length=context_length,
        max_total_tokens=max(256, context_length * 2),
        max_mamba_cache_size=8,
        max_running_requests=2,
        mem_fraction_static=mem_fraction_static,
        **kwargs,
    )


def _generate(engine: Any, input_ids: list[int], max_new_tokens: int) -> dict[str, Any]:
    return engine.generate(
        input_ids=input_ids,
        sampling_params={
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        },
    )


def run_speculative_smoke(
    model_path: Path,
    input_ids: list[int],
    max_new_tokens: int,
    *,
    context_length: int,
    mem_fraction_static: float = 0.25,
) -> dict[str, Any]:
    """Require exact greedy parity and evidence of target verification.

    The speculative engine uses a breadth-one NGRAM corpus, which produces a
    linear draft chain. Tree branching is deliberately outside this overlay's
    current verifier contract.
    """

    register()
    baseline_engine = _create_engine(
        model_path,
        context_length=context_length,
        speculative=False,
        mem_fraction_static=mem_fraction_static,
    )
    try:
        baseline = _generate(baseline_engine, input_ids, max_new_tokens)
    finally:
        baseline_engine.shutdown()

    speculative_engine = _create_engine(
        model_path,
        context_length=context_length,
        speculative=True,
        mem_fraction_static=mem_fraction_static,
    )
    try:
        speculative = _generate(speculative_engine, input_ids, max_new_tokens)
    finally:
        speculative_engine.shutdown()

    if speculative["output_ids"] != baseline["output_ids"]:
        raise AssertionError(
            "NGRAM target verification changed greedy output: "
            f"baseline={baseline['output_ids']}, speculative={speculative['output_ids']}"
        )
    metadata = speculative["meta_info"]
    verify_count = int(metadata.get("spec_verify_ct", 0))
    proposed_drafts = int(metadata.get("spec_num_proposed_drafts", 0))
    if verify_count < 1 or proposed_drafts < 1:
        raise AssertionError(
            "SGLang returned no evidence of speculative target verification: "
            f"spec_verify_ct={verify_count}, spec_num_proposed_drafts={proposed_drafts}"
        )

    return {
        "input_ids": input_ids,
        "output_ids": baseline["output_ids"],
        "spec_verify_ct": verify_count,
        "spec_num_proposed_drafts": proposed_drafts,
        "spec_num_correct_drafts": int(metadata.get("spec_num_correct_drafts", 0)),
        "spec_accept_rate": float(metadata["spec_accept_rate"]),
        "spec_accept_length": float(metadata["spec_accept_length"]),
    }


def parse_args() -> argparse.Namespace:
    """Parse speculative smoke-test arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-ids", type=int, nargs="+")
    parser.add_argument("--prompt-length", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--mem-fraction-static", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    """Run matched ordinary and NGRAM inference and emit a JSON report."""

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
    report = run_speculative_smoke(
        args.model,
        input_ids,
        args.max_new_tokens,
        context_length=args.context_length,
        mem_fraction_static=args.mem_fraction_static,
    )
    LOGGER.info("%s", json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
