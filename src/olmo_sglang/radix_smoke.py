# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise OLMo KDA radix-cache reuse and cached-continuation parity."""

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
    disable_radix_cache: bool,
    mamba_radix_cache_strategy: str,
    context_length: int | None,
    chunked_prefill_size: int | None = None,
) -> Any:
    """Create the correctness-first tiny-checkpoint SGLang engine."""

    import sglang as sgl

    if context_length is None:
        config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        context_length = int(config["max_position_embeddings"])
    engine_kwargs = {}
    if chunked_prefill_size is not None:
        engine_kwargs["chunked_prefill_size"] = chunked_prefill_size
    return sgl.Engine(
        model_path=str(model_path),
        trust_remote_code=True,
        skip_tokenizer_init=True,
        dtype="auto",
        cuda_graph_backend_decode="disabled",
        cuda_graph_backend_prefill="disabled",
        disable_radix_cache=disable_radix_cache,
        mamba_radix_cache_strategy=mamba_radix_cache_strategy,
        disable_overlap_schedule=mamba_radix_cache_strategy == "no_buffer",
        attention_backend="torch_native",
        context_length=context_length,
        max_total_tokens=max(64, context_length * 4),
        max_mamba_cache_size=32,
        mem_fraction_static=0.15,
        **engine_kwargs,
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


def _generate_batch(
    engine: Any, input_ids: list[int], max_new_tokens: int, repeats: int
) -> list[dict[str, Any]]:
    """Submit identical prompts together as one scheduler batch."""

    outputs = engine.generate(
        input_ids=[input_ids for _ in range(repeats)],
        sampling_params={"temperature": 0, "max_new_tokens": max_new_tokens},
    )
    if not isinstance(outputs, list):
        raise TypeError(f"Expected a batch of SGLang outputs, got {type(outputs)!r}")
    return [
        {
            "input_ids": input_ids,
            "output_ids": output["output_ids"],
            "cached_tokens": int(output["meta_info"]["cached_tokens"]),
        }
        for output in outputs
    ]


def _generate_mixed_batch(
    engine: Any, input_ids: list[list[int]], max_new_tokens: int
) -> list[dict[str, Any]]:
    """Submit different prompt lengths together as one scheduler batch."""

    outputs = engine.generate(
        input_ids=input_ids,
        sampling_params={"temperature": 0, "max_new_tokens": max_new_tokens},
    )
    if not isinstance(outputs, list):
        raise TypeError(f"Expected a batch of SGLang outputs, got {type(outputs)!r}")
    return [
        {
            "input_ids": prompt,
            "output_ids": output["output_ids"],
            "cached_tokens": int(output["meta_info"]["cached_tokens"]),
        }
        for prompt, output in zip(input_ids, outputs, strict=True)
    ]


def _mixed_prompts(prompt_length: int, tracked_prefix_length: int) -> list[list[int]]:
    """Build three different-length prompts sharing one tracked prefix."""

    if prompt_length <= tracked_prefix_length + 32:
        raise ValueError(
            "prompt_length must exceed tracked_prefix_length by more than 32 tokens"
        )
    common = [5 + index % 30 for index in range(tracked_prefix_length)]
    lengths = (prompt_length - 32, prompt_length, prompt_length + 32)
    return [
        common
        + [
            5 + (index + branch * 7) % 30
            for index in range(length - tracked_prefix_length)
        ]
        for branch, length in enumerate(lengths)
    ]


def run_mixed_chunked_prefill_probe(
    model_path: Path,
    prompt_length: int,
    max_new_tokens: int,
    *,
    chunked_prefill_size: int = 64,
    tracked_prefix_length: int = 256,
    mamba_radix_cache_strategy: str = "extra_buffer",
    context_length: int | None = None,
) -> dict[str, Any]:
    """Validate mixed-length chunked prefill with a shared KDA radix snapshot.

    A warm request creates a recurrent-state snapshot at the tracked prefix.
    Three different prompt lengths then reuse that state while SGLang splits
    their remaining prefill into bounded chunks. Their greedy continuations
    must match a fresh cache-disabled engine using the same chunk size.

    Args:
        model_path: Tiny OLMo KDA checkpoint directory.
        prompt_length: Center length of the three synthetic prompts.
        max_new_tokens: Greedy tokens generated for each request.
        chunked_prefill_size: Maximum tokens in one SGLang prefill chunk.
        tracked_prefix_length: Expected recurrent-state checkpoint boundary.
        mamba_radix_cache_strategy: SGLang recurrent radix-cache strategy.
        context_length: Optional checkpoint context-length override.

    Returns:
        JSON-serializable warm, cached, and uncached request details.

    Raises:
        AssertionError: If the shared snapshot is not reused or changes output.
        ValueError: If the requested shapes cannot exercise chunked prefill.
    """

    if chunked_prefill_size < 1:
        raise ValueError("chunked_prefill_size must be positive")
    if tracked_prefix_length % chunked_prefill_size != 0:
        raise ValueError(
            "tracked_prefix_length must be divisible by chunked_prefill_size"
        )
    prompts = _mixed_prompts(prompt_length, tracked_prefix_length)
    resolved_context_length = context_length
    if resolved_context_length is None:
        config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        resolved_context_length = int(config["max_position_embeddings"])
    required_context_length = len(prompts[-1]) + max_new_tokens
    if required_context_length > resolved_context_length:
        raise ValueError(
            f"probe requires context length {required_context_length}, "
            f"got {resolved_context_length}"
        )

    register()
    cached_engine = _create_engine(
        model_path,
        disable_radix_cache=False,
        mamba_radix_cache_strategy=mamba_radix_cache_strategy,
        context_length=resolved_context_length,
        chunked_prefill_size=chunked_prefill_size,
    )
    try:
        warm = _generate(cached_engine, prompts[-1], 1)
        cached = _generate_mixed_batch(cached_engine, prompts, max_new_tokens)
    finally:
        cached_engine.shutdown()

    uncached_engine = _create_engine(
        model_path,
        disable_radix_cache=True,
        mamba_radix_cache_strategy=mamba_radix_cache_strategy,
        context_length=resolved_context_length,
        chunked_prefill_size=chunked_prefill_size,
    )
    try:
        uncached = _generate_mixed_batch(uncached_engine, prompts, max_new_tokens)
    finally:
        uncached_engine.shutdown()

    cached_counts = [request["cached_tokens"] for request in cached]
    if any(count < tracked_prefix_length for count in cached_counts):
        raise AssertionError(
            f"Expected every mixed request to reuse at least {tracked_prefix_length} "
            f"tokens, got {cached_counts}"
        )
    mismatches = [
        index
        for index, (cached_request, uncached_request) in enumerate(
            zip(cached, uncached, strict=True)
        )
        if cached_request["output_ids"] != uncached_request["output_ids"]
    ]
    if mismatches:
        raise AssertionError(
            "Mixed-length chunked prefill changed greedy output for request "
            f"indices {mismatches}: cached={cached!r}, uncached={uncached!r}"
        )
    return {
        "chunked_prefill_size": chunked_prefill_size,
        "tracked_prefix_length": tracked_prefix_length,
        "warm": warm,
        "cached": cached,
        "uncached": uncached,
    }


def run_radix_smoke(
    model_path: Path,
    input_ids: list[int],
    max_new_tokens: int,
    *,
    mamba_radix_cache_strategy: str = "auto",
    context_length: int | None = None,
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
    cached_engine = _create_engine(
        model_path,
        disable_radix_cache=False,
        mamba_radix_cache_strategy=mamba_radix_cache_strategy,
        context_length=context_length,
    )
    try:
        warm = _generate(cached_engine, input_ids, max_new_tokens)
        continuation_input = input_ids + warm["output_ids"]
        cached = _generate(cached_engine, continuation_input, max_new_tokens)
    finally:
        cached_engine.shutdown()

    uncached_engine = _create_engine(
        model_path,
        disable_radix_cache=True,
        mamba_radix_cache_strategy=mamba_radix_cache_strategy,
        context_length=context_length,
    )
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


def run_repeated_prompt_probe(
    model_path: Path,
    input_ids: list[int],
    max_new_tokens: int,
    repeats: int,
    *,
    mamba_radix_cache_strategy: str = "auto",
    context_length: int | None = None,
) -> dict[str, Any]:
    """Compare sequential and simultaneous submissions of one prompt.

    The sequential and simultaneous halves reveal whether ordinary completed
    requests leave a reusable state at the prompt boundary. The seeded half
    first completes ``input_ids[:-1]`` with one throwaway token, creating the
    exact recurrent-state boundary needed by a batch of full-prompt requests.

    Args:
        model_path: Tiny OLMo KDA checkpoint directory.
        input_ids: Prompt token IDs repeated in every request.
        max_new_tokens: Greedy tokens generated by each request.
        repeats: Number of identical requests per experiment.

    Returns:
        JSON-serializable sequential and simultaneous request details.

    Raises:
        AssertionError: If seeded reuse does not occur or greedy outputs differ
            across identical requests.
        ValueError: If fewer than two requests or input tokens are specified.
    """

    if repeats < 2:
        raise ValueError(f"repeats must be at least 2, got {repeats}")
    if len(input_ids) < 2:
        raise ValueError("input_ids must contain at least two tokens")

    register()
    sequential_engine = _create_engine(
        model_path,
        disable_radix_cache=False,
        mamba_radix_cache_strategy=mamba_radix_cache_strategy,
        context_length=context_length,
    )
    try:
        sequential = [
            _generate(sequential_engine, input_ids, max_new_tokens)
            for _ in range(repeats)
        ]
    finally:
        sequential_engine.shutdown()

    simultaneous_engine = _create_engine(
        model_path,
        disable_radix_cache=False,
        mamba_radix_cache_strategy=mamba_radix_cache_strategy,
        context_length=context_length,
    )
    try:
        simultaneous = _generate_batch(
            simultaneous_engine, input_ids, max_new_tokens, repeats
        )
    finally:
        simultaneous_engine.shutdown()

    seeded_engine = _create_engine(
        model_path,
        disable_radix_cache=False,
        mamba_radix_cache_strategy=mamba_radix_cache_strategy,
        context_length=context_length,
    )
    try:
        seed = _generate(seeded_engine, input_ids[:-1], 1)
        seeded_simultaneous = _generate_batch(
            seeded_engine, input_ids, max_new_tokens, repeats
        )
    finally:
        seeded_engine.shutdown()

    if any(item["cached_tokens"] <= 0 for item in seeded_simultaneous):
        raise AssertionError(
            "Expected the explicit prompt-boundary seed to produce cache hits, "
            f"got seed={seed!r}, requests={seeded_simultaneous!r}"
        )
    expected_output = sequential[0]["output_ids"]
    all_outputs = sequential[1:] + simultaneous + seeded_simultaneous
    if any(item["output_ids"] != expected_output for item in all_outputs):
        raise AssertionError(
            "Identical greedy requests produced different outputs: "
            f"sequential={sequential!r}, simultaneous={simultaneous!r}, "
            f"seeded_simultaneous={seeded_simultaneous!r}"
        )
    return {
        "mamba_radix_cache_strategy": mamba_radix_cache_strategy,
        "sequential": sequential,
        "simultaneous": simultaneous,
        "seeded_simultaneous": {"seed": seed, "requests": seeded_simultaneous},
    }


def _control_result(operation: str, result: Any) -> dict[str, Any]:
    """Require one SGLang control operation to report success."""

    if isinstance(result, tuple):
        success = bool(result[0]) if result else False
        message = str(result[1]) if len(result) > 1 else ""
    else:
        success = bool(getattr(result, "success", False))
        message = str(getattr(result, "message", ""))
    if not success:
        detail = message if message else repr(result)
        raise AssertionError(f"SGLang {operation} failed: {detail}")
    return {"success": success, "message": message}


def run_policy_refresh_probe(
    model_path: Path,
    input_ids: list[int],
    max_new_tokens: int,
    *,
    mamba_radix_cache_strategy: str = "extra_buffer",
    context_length: int | None = None,
) -> dict[str, Any]:
    """Verify cache reuse is bounded by an idle policy-weight refresh.

    This mirrors MILES' correctness-first refresh boundary on one embedded
    engine: establish cache reuse, flush while idle, reload weights, then prove
    that the first request misses and the next identical request reuses only
    state created after the refresh. Reloading the same tiny checkpoint keeps
    greedy tokens stable so cache lifecycle behavior is isolated from model
    quality or optimization effects.

    Args:
        model_path: Tiny OLMo KDA checkpoint directory.
        input_ids: Prompt token IDs repeated before and after refresh.
        max_new_tokens: Greedy tokens generated by each request.
        mamba_radix_cache_strategy: SGLang recurrent radix-cache strategy.
        context_length: Optional checkpoint context-length override.

    Returns:
        JSON-serializable generations and control-operation results.

    Raises:
        AssertionError: If cache reuse crosses the refresh boundary, does not
            recover afterward, or changes greedy output.
    """

    register()
    engine = _create_engine(
        model_path,
        disable_radix_cache=False,
        mamba_radix_cache_strategy=mamba_radix_cache_strategy,
        context_length=context_length,
    )
    try:
        before_refresh = [
            _generate(engine, input_ids, max_new_tokens),
            _generate(engine, input_ids, max_new_tokens),
        ]
        flush = _control_result("cache flush", engine.flush_cache())
        weight_update = _control_result(
            "weight update", engine.update_weights_from_disk(str(model_path))
        )
        after_refresh = [
            _generate(engine, input_ids, max_new_tokens),
            _generate(engine, input_ids, max_new_tokens),
        ]
    finally:
        engine.shutdown()

    if before_refresh[1]["cached_tokens"] <= 0:
        raise AssertionError(
            f"Expected a cache hit before policy refresh, got {before_refresh[1]!r}"
        )
    if after_refresh[0]["cached_tokens"] != 0:
        raise AssertionError(
            "Old recurrent state crossed the policy refresh boundary: "
            f"{after_refresh[0]!r}"
        )
    if after_refresh[1]["cached_tokens"] <= 0:
        raise AssertionError(
            "Cache reuse did not recover within the refreshed policy version: "
            f"{after_refresh[1]!r}"
        )
    expected_output = before_refresh[0]["output_ids"]
    all_requests = before_refresh[1:] + after_refresh
    if any(request["output_ids"] != expected_output for request in all_requests):
        raise AssertionError(
            "Reloading the same policy changed greedy output: "
            f"before={before_refresh!r}, after={after_refresh!r}"
        )
    return {
        "mamba_radix_cache_strategy": mamba_radix_cache_strategy,
        "before_refresh": before_refresh,
        "refresh": {"flush": flush, "weight_update": weight_update},
        "after_refresh": after_refresh,
    }


def parse_args() -> argparse.Namespace:
    """Parse tiny radix-cache smoke-test arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-ids", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument(
        "--prompt-length",
        type=int,
        help="Generate deterministic synthetic input IDs of this length.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument(
        "--context-length",
        type=int,
        help="Override the checkpoint's max_position_embeddings.",
    )
    parser.add_argument(
        "--mamba-radix-cache-strategy",
        choices=("auto", "no_buffer", "extra_buffer", "extra_buffer_lazy"),
        default="auto",
    )
    probe = parser.add_mutually_exclusive_group()
    probe.add_argument(
        "--repeat-prompt-probe",
        action="store_true",
        help="Compare repeated sequential requests with one simultaneous batch.",
    )
    probe.add_argument(
        "--policy-refresh-probe",
        action="store_true",
        help="Require cache invalidation and recovery across a weight reload.",
    )
    probe.add_argument(
        "--mixed-chunked-prefill-probe",
        action="store_true",
        help="Validate shared-prefix reuse across mixed-length chunked prefill.",
    )
    parser.add_argument(
        "--chunked-prefill-size",
        type=int,
        default=64,
        help="Maximum tokens in one prefill chunk for the mixed-length probe.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    """Run the smoke test and emit its machine-readable result."""

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    LOGGER.setLevel(logging.INFO)
    args = parse_args()
    input_ids = args.input_ids
    if args.prompt_length is not None:
        if args.prompt_length < 2:
            raise ValueError("prompt_length must be at least 2")
        input_ids = [5 + index % 30 for index in range(args.prompt_length)]
    if args.mixed_chunked_prefill_probe:
        prompt_length = args.prompt_length if args.prompt_length is not None else 300
        report = run_mixed_chunked_prefill_probe(
            args.model,
            prompt_length,
            args.max_new_tokens,
            chunked_prefill_size=args.chunked_prefill_size,
            mamba_radix_cache_strategy=args.mamba_radix_cache_strategy,
            context_length=args.context_length,
        )
    elif args.policy_refresh_probe:
        report = run_policy_refresh_probe(
            args.model,
            input_ids,
            args.max_new_tokens,
            mamba_radix_cache_strategy=args.mamba_radix_cache_strategy,
            context_length=args.context_length,
        )
    elif args.repeat_prompt_probe:
        report = run_repeated_prompt_probe(
            args.model,
            input_ids,
            args.max_new_tokens,
            args.repeats,
            mamba_radix_cache_strategy=args.mamba_radix_cache_strategy,
            context_length=args.context_length,
        )
    else:
        report = run_radix_smoke(
            args.model,
            input_ids,
            args.max_new_tokens,
            mamba_radix_cache_strategy=args.mamba_radix_cache_strategy,
            context_length=args.context_length,
        )
    LOGGER.info("%s", json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
