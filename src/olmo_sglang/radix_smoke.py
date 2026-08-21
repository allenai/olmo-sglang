# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise cached OLMo KDA continuation against a cache-disabled engine."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from olmo_sglang import register

LOGGER = logging.getLogger(__name__)


def _create_engine(model_path: Path, *, disable_radix_cache: bool) -> Any:
    """Create the correctness-first tiny-checkpoint SGLang engine."""

    import sglang as sgl

    return sgl.Engine(
        model_path=str(model_path),
        trust_remote_code=True,
        skip_tokenizer_init=True,
        dtype="float16",
        cuda_graph_backend_decode="disabled",
        cuda_graph_backend_prefill="disabled",
        disable_radix_cache=disable_radix_cache,
        attention_backend="torch_native",
        context_length=32,
        max_total_tokens=64,
        max_mamba_cache_size=4,
        mem_fraction_static=0.15,
    )


def _generate(engine: Any, input_ids: list[int], max_new_tokens: int) -> dict[str, Any]:
    output = engine.generate(
        input_ids=input_ids,
        sampling_params={"temperature": 0, "max_new_tokens": max_new_tokens},
    )
    return {
        "input_ids": input_ids,
        "output_ids": output["output_ids"],
        "cached_tokens": int(output["meta_info"]["cached_tokens"]),
    }


def run_radix_smoke(
    model_path: Path, input_ids: list[int], max_new_tokens: int
) -> dict[str, Any]:
    """Compare a cache-hit continuation with an uncached continuation.

    The first cache-enabled generation creates a complete recurrent-state
    checkpoint. The second request continues from that exact history, requiring
    SGLang to copy both the convolution window and KDA matrix from the radix
    node. A fresh cache-disabled engine then recomputes the same continuation.

    Args:
        model_path: Tiny OLMo KDA checkpoint directory.
        input_ids: Initial prompt token IDs.
        max_new_tokens: Greedy tokens generated in each turn.

    Returns:
        JSON-serializable warm, cached, and uncached generation details.

    Raises:
        AssertionError: If no prefix was reused or cached output changes.
    """

    register()
    cached_engine = _create_engine(model_path, disable_radix_cache=False)
    try:
        warm = _generate(cached_engine, input_ids, max_new_tokens)
        continuation_input = input_ids + warm["output_ids"]
        cached = _generate(cached_engine, continuation_input, max_new_tokens)
    finally:
        cached_engine.shutdown()

    uncached_engine = _create_engine(model_path, disable_radix_cache=True)
    try:
        uncached = _generate(uncached_engine, continuation_input, max_new_tokens)
    finally:
        uncached_engine.shutdown()

    if cached["cached_tokens"] <= 0:
        raise AssertionError(f"Expected a KDA radix-cache hit, got {cached!r}")
    if cached["output_ids"] != uncached["output_ids"]:
        raise AssertionError(
            "KDA radix state changed greedy continuation: "
            f"cached={cached['output_ids']}, uncached={uncached['output_ids']}"
        )
    return {"warm": warm, "cached": cached, "uncached": uncached}


def parse_args() -> argparse.Namespace:
    """Parse tiny radix-cache smoke-test arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-ids", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--max-new-tokens", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    """Run the smoke test and emit its machine-readable result."""

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    LOGGER.setLevel(logging.INFO)
    args = parse_args()
    report = run_radix_smoke(args.model, args.input_ids, args.max_new_tokens)
    LOGGER.info("%s", json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
