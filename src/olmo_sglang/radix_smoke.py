# SPDX-License-Identifier: Apache-2.0

"""Exercise OLMo KDA radix-cache reuse and cached-continuation parity."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
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
    max_running_requests: int | None = None,
    max_mamba_cache_size: int = 32,
    max_total_tokens: int | None = None,
    schedule_conservativeness: float = 1.0,
    skip_tokenizer_init: bool = True,
    attention_backend: str = "torch_native",
) -> Any:
    """Create the correctness-first tiny-checkpoint SGLang engine."""

    import sglang as sgl

    if context_length is None:
        config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        context_length = int(config["max_position_embeddings"])
    engine_kwargs = {}
    if chunked_prefill_size is not None:
        engine_kwargs["chunked_prefill_size"] = chunked_prefill_size
    if max_running_requests is not None:
        engine_kwargs["max_running_requests"] = max_running_requests
    return sgl.Engine(
        model_path=str(model_path),
        trust_remote_code=True,
        skip_tokenizer_init=skip_tokenizer_init,
        dtype="auto",
        cuda_graph_backend_decode="disabled",
        cuda_graph_backend_prefill="disabled",
        disable_radix_cache=disable_radix_cache,
        mamba_radix_cache_strategy=mamba_radix_cache_strategy,
        disable_overlap_schedule=mamba_radix_cache_strategy == "no_buffer",
        attention_backend=attention_backend,
        context_length=context_length,
        max_total_tokens=(
            max(64, context_length * 4)
            if max_total_tokens is None
            else max_total_tokens
        ),
        max_mamba_cache_size=max_mamba_cache_size,
        mem_fraction_static=0.15,
        schedule_conservativeness=schedule_conservativeness,
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


def _finish_reason(output: dict[str, Any]) -> str | None:
    """Extract SGLang's structured finish-reason type."""

    finish_reason = output.get("meta_info", {}).get("finish_reason")
    return finish_reason.get("type") if isinstance(finish_reason, dict) else None


async def _run_cancellation_batch(
    engine: Any,
    prompts: list[list[int]],
    max_new_tokens: int,
    aborted_indices: tuple[int, ...],
    abort_delay_seconds: float,
) -> list[dict[str, Any]]:
    """Submit one overloaded batch and cancel selected request IDs."""

    request_ids = [f"olmo-kda-cancel-{index}" for index in range(len(prompts))]
    tasks = [
        asyncio.create_task(
            engine.async_generate(
                input_ids=prompt,
                sampling_params={
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
                rid=request_id,
            )
        )
        for request_id, prompt in zip(request_ids, prompts, strict=True)
    ]
    await asyncio.sleep(abort_delay_seconds)
    for index in aborted_indices:
        engine.tokenizer_manager.abort_request(rid=request_ids[index])
    return await asyncio.wait_for(asyncio.gather(*tasks), timeout=120)


def run_cancellation_probe(
    model_path: Path,
    prompt_length: int,
    max_new_tokens: int,
    *,
    abort_delay_seconds: float = 0.1,
    chunked_prefill_size: int = 64,
    max_running_requests: int = 2,
    max_mamba_cache_size: int = 22,
    mamba_radix_cache_strategy: str = "extra_buffer",
    context_length: int | None = None,
) -> dict[str, Any]:
    """Cancel running and queued work, then prove KDA state remains usable.

    Four requests are submitted to an engine that admits two at a time. The
    first and last request IDs are cancelled after scheduling begins, covering
    one expected running request and one expected queued request. All responses
    must terminate, both cancellations must be acknowledged, and a subsequent
    mixed batch must match a fresh cache-disabled engine before an idle cache
    flush succeeds.

    Args:
        model_path: Tiny OLMo KDA checkpoint directory.
        prompt_length: Center length of the synthetic request batch.
        max_new_tokens: Ignore-EOS decode length used to keep work cancellable.
        abort_delay_seconds: Time allowed for scheduler admission before abort.
        chunked_prefill_size: Maximum tokens in one SGLang prefill chunk.
        max_running_requests: Admission ceiling for the overloaded batch.
        max_mamba_cache_size: Recurrent-state slots available to the engine.
        mamba_radix_cache_strategy: SGLang recurrent radix-cache strategy.
        context_length: Optional checkpoint context-length override.

    Returns:
        JSON-serializable cancellation and post-cancellation parity details.

    Raises:
        AssertionError: If cancellation, recovery parity, or cleanup fails.
        ValueError: If the requested workload cannot create queue pressure.
    """

    if max_running_requests < 1:
        raise ValueError("max_running_requests must be positive")
    if max_running_requests >= 4:
        raise ValueError("max_running_requests must be less than four")
    if max_new_tokens < 16:
        raise ValueError("max_new_tokens must be at least 16 for cancellation")
    if abort_delay_seconds <= 0:
        raise ValueError("abort_delay_seconds must be positive")

    base_prompts = _mixed_prompts(prompt_length, tracked_prefix_length=256)
    prompts = [base_prompts[0], base_prompts[1], base_prompts[2], base_prompts[1]]
    resolved_context_length = context_length
    if resolved_context_length is None:
        config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        resolved_context_length = int(config["max_position_embeddings"])
    required_context_length = max(map(len, prompts)) + max_new_tokens
    if required_context_length > resolved_context_length:
        raise ValueError(
            f"probe requires context length {required_context_length}, "
            f"got {resolved_context_length}"
        )

    register()
    engine = _create_engine(
        model_path,
        disable_radix_cache=False,
        mamba_radix_cache_strategy=mamba_radix_cache_strategy,
        context_length=resolved_context_length,
        chunked_prefill_size=chunked_prefill_size,
        max_running_requests=max_running_requests,
        max_mamba_cache_size=max_mamba_cache_size,
    )
    aborted_indices = (0, 3)
    try:
        outputs = engine.loop.run_until_complete(
            _run_cancellation_batch(
                engine,
                prompts,
                max_new_tokens,
                aborted_indices,
                abort_delay_seconds,
            )
        )
        recovery_prompts = [prompts[1], prompts[2]]
        recovered = _generate_mixed_batch(engine, recovery_prompts, 4)
        flush = _control_result("post-cancellation cache flush", engine.flush_cache())
    finally:
        engine.shutdown()

    uncached_engine = _create_engine(
        model_path,
        disable_radix_cache=True,
        mamba_radix_cache_strategy=mamba_radix_cache_strategy,
        context_length=resolved_context_length,
        chunked_prefill_size=chunked_prefill_size,
        max_running_requests=max_running_requests,
        max_mamba_cache_size=max_mamba_cache_size,
    )
    try:
        uncached = _generate_mixed_batch(uncached_engine, recovery_prompts, 4)
    finally:
        uncached_engine.shutdown()

    reasons = [_finish_reason(output) for output in outputs]
    if any(reasons[index] != "abort" for index in aborted_indices):
        raise AssertionError(
            f"Expected request indices {aborted_indices} to abort, got {reasons}"
        )
    surviving_indices = set(range(len(outputs))) - set(aborted_indices)
    if any(reasons[index] == "abort" for index in surviving_indices):
        raise AssertionError(f"Unexpected survivor cancellation: {reasons}")
    mismatches = [
        index
        for index, (recovered_request, uncached_request) in enumerate(
            zip(recovered, uncached, strict=True)
        )
        if recovered_request["output_ids"] != uncached_request["output_ids"]
    ]
    if mismatches:
        raise AssertionError(
            "Post-cancellation KDA state changed greedy output for request "
            f"indices {mismatches}: recovered={recovered!r}, uncached={uncached!r}"
        )
    return {
        "max_running_requests": max_running_requests,
        "max_mamba_cache_size": max_mamba_cache_size,
        "aborted_indices": aborted_indices,
        "finish_reasons": reasons,
        "output_token_counts": [
            len(output.get("output_ids", [])) for output in outputs
        ],
        "recovered": recovered,
        "uncached": uncached,
        "flush": flush,
    }


def _generate_retraction_batch(
    engine: Any,
    prompts: list[list[int]],
    max_new_tokens: int,
    *,
    ignore_eos: bool = True,
    min_new_tokens: int = 0,
) -> list[dict[str, Any]]:
    """Generate a fixed-length batch and retain scheduler retraction counts."""

    outputs = engine.generate(
        input_ids=prompts,
        sampling_params={
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            "ignore_eos": ignore_eos,
        },
    )
    if not isinstance(outputs, list):
        raise TypeError(f"Expected a batch of SGLang outputs, got {type(outputs)!r}")
    return [
        {
            "input_length": len(prompt),
            "output_ids": output["output_ids"],
            "cached_tokens": int(output["meta_info"]["cached_tokens"]),
            "num_retractions": int(output["meta_info"].get("num_retractions", 0)),
        }
        for prompt, output in zip(prompts, outputs, strict=True)
    ]


def run_retraction_probe(
    model_path: Path,
    prompt_length: int,
    max_new_tokens: int,
    *,
    retraction_interval: int = 7,
    chunked_prefill_size: int = 64,
    max_running_requests: int = 4,
    max_mamba_cache_size: int = 32,
    mamba_radix_cache_strategy: str = "extra_buffer",
    context_length: int | None = None,
) -> dict[str, Any]:
    """Force decode retraction and compare resumed KDA output with eager control.

    The baseline runs first with SGLang's ordinary scheduler. A second engine
    enables SGLang's own deterministic retraction test hook before its worker
    process starts. At least one request must report a retraction, every resumed
    greedy continuation must match the baseline, and the engine must drain far
    enough for an idle cache flush.

    Args:
        model_path: Tiny OLMo KDA checkpoint directory.
        prompt_length: Center length of the synthetic request batch.
        max_new_tokens: Ignore-EOS decode length for each request.
        retraction_interval: Scheduler forward interval between forced retracts.
        chunked_prefill_size: Maximum tokens in one SGLang prefill chunk.
        max_running_requests: Admission ceiling for the concurrent batch.
        max_mamba_cache_size: Recurrent-state slots available to the engine.
        mamba_radix_cache_strategy: SGLang recurrent radix-cache strategy.
        context_length: Optional checkpoint context-length override.

    Returns:
        JSON-serializable control, retracted, and cleanup details.

    Raises:
        AssertionError: If no retraction occurs, output changes, or cleanup fails.
        ValueError: If the requested workload is too small or exceeds context.
    """

    if retraction_interval < 1:
        raise ValueError("retraction_interval must be positive")
    if max_new_tokens < retraction_interval * 2:
        raise ValueError("max_new_tokens must span at least two retraction intervals")

    base_prompts = _mixed_prompts(prompt_length, tracked_prefix_length=256)
    prompts = [base_prompts[0], base_prompts[1], base_prompts[2], base_prompts[1]]
    resolved_context_length = context_length
    if resolved_context_length is None:
        config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        resolved_context_length = int(config["max_position_embeddings"])
    required_context_length = max(map(len, prompts)) + max_new_tokens
    if required_context_length > resolved_context_length:
        raise ValueError(
            f"probe requires context length {required_context_length}, "
            f"got {resolved_context_length}"
        )

    register()
    control_engine = _create_engine(
        model_path,
        disable_radix_cache=False,
        mamba_radix_cache_strategy=mamba_radix_cache_strategy,
        context_length=resolved_context_length,
        chunked_prefill_size=chunked_prefill_size,
        max_running_requests=max_running_requests,
        max_mamba_cache_size=max_mamba_cache_size,
    )
    try:
        control = _generate_retraction_batch(control_engine, prompts, max_new_tokens)
    finally:
        control_engine.shutdown()

    test_environment = {
        "SGLANG_TEST_RETRACT": "True",
        "SGLANG_TEST_RETRACT_INTERVAL": str(retraction_interval),
    }
    previous_environment = {name: os.environ.get(name) for name in test_environment}
    os.environ.update(test_environment)
    retraction_engine = None
    try:
        retraction_engine = _create_engine(
            model_path,
            disable_radix_cache=False,
            mamba_radix_cache_strategy=mamba_radix_cache_strategy,
            context_length=resolved_context_length,
            chunked_prefill_size=chunked_prefill_size,
            max_running_requests=max_running_requests,
            max_mamba_cache_size=max_mamba_cache_size,
        )
        retracted = _generate_retraction_batch(
            retraction_engine, prompts, max_new_tokens
        )
        flush = _control_result(
            "post-retraction cache flush", retraction_engine.flush_cache()
        )
    finally:
        if retraction_engine is not None:
            retraction_engine.shutdown()
        for name, value in previous_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    retraction_counts = [request["num_retractions"] for request in retracted]
    if not any(count > 0 for count in retraction_counts):
        raise AssertionError(
            f"SGLang forced no request retractions: {retraction_counts}"
        )
    mismatches = [
        index
        for index, (control_request, retracted_request) in enumerate(
            zip(control, retracted, strict=True)
        )
        if control_request["output_ids"] != retracted_request["output_ids"]
    ]
    if mismatches:
        raise AssertionError(
            "KDA retraction/resume changed greedy output for request indices "
            f"{mismatches}: control={control!r}, retracted={retracted!r}"
        )
    return {
        "retraction_interval": retraction_interval,
        "retraction_counts": retraction_counts,
        "control": control,
        "retracted": retracted,
        "flush": flush,
    }


def _pressure_prompts(prompt_length: int, request_count: int) -> list[list[int]]:
    """Build different-length prompts with no shared initial token."""

    if request_count < 2:
        raise ValueError("request_count must be at least two")
    offsets = [4 * index - 2 * (request_count - 1) for index in range(request_count)]
    lengths = [prompt_length + offset for offset in offsets]
    if min(lengths) < 2:
        raise ValueError("prompt_length is too short for the requested pressure batch")
    return [
        [5 + (index + branch * 7) % 30 for index in range(length)]
        for branch, length in enumerate(lengths)
    ]


def run_organic_retraction_probe(
    model_path: Path,
    prompt_length: int,
    max_new_tokens: int,
    *,
    request_count: int = 8,
    pressure_max_total_tokens: int = 1100,
    schedule_conservativeness: float = 0.1,
    chunked_prefill_size: int = 64,
    max_mamba_cache_size: int = 52,
    mamba_radix_cache_strategy: str = "extra_buffer",
    attention_backend: str = "torch_native",
    context_length: int | None = None,
) -> dict[str, Any]:
    """Cause real KV pressure and compare resumed KDA output with a control.

    Unlike :func:`run_retraction_probe`, this path does not enable SGLang's
    deterministic retraction hook. It admits a bounded batch with an explicitly
    under-conservative schedule, then relies on normal decode memory checks to
    retract and resume requests as the full-attention KV pool fills.

    Args:
        model_path: Tiny hybrid OLMo KDA checkpoint with tokenizer assets.
        prompt_length: Center length of the unique synthetic prompts.
        max_new_tokens: Exact greedy decode length for every request.
        request_count: Number of requests submitted in one batch.
        pressure_max_total_tokens: KV-pool token cap for the pressure engine.
        schedule_conservativeness: Admission reserve multiplier under pressure.
        chunked_prefill_size: Maximum tokens in one prefill chunk.
        max_mamba_cache_size: Recurrent-state slots available to each engine.
        mamba_radix_cache_strategy: SGLang recurrent radix-cache strategy.
        attention_backend: Full-attention backend used by both engines.
        context_length: Optional checkpoint context-length override.

    Returns:
        JSON-serializable control, pressure, retraction, and cleanup details.

    Raises:
        AssertionError: If pressure causes no retraction, changes output, or leaks.
        ValueError: If the requested workload cannot fit the declared context.
    """

    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if pressure_max_total_tokens < 1:
        raise ValueError("pressure_max_total_tokens must be positive")
    if not 0 < schedule_conservativeness < 1:
        raise ValueError("schedule_conservativeness must be between zero and one")
    prompts = _pressure_prompts(prompt_length, request_count)
    resolved_context_length = context_length
    if resolved_context_length is None:
        config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        resolved_context_length = int(config["max_position_embeddings"])
    required_context_length = max(map(len, prompts)) + max_new_tokens
    if required_context_length > resolved_context_length:
        raise ValueError(
            f"probe requires context length {required_context_length}, "
            f"got {resolved_context_length}"
        )
    control_max_total_tokens = max(
        4096,
        2 * (sum(map(len, prompts)) + request_count * max_new_tokens),
    )

    register()
    common_engine_args = {
        "disable_radix_cache": False,
        "mamba_radix_cache_strategy": mamba_radix_cache_strategy,
        "context_length": resolved_context_length,
        "chunked_prefill_size": chunked_prefill_size,
        "max_running_requests": request_count,
        "max_mamba_cache_size": max_mamba_cache_size,
        "skip_tokenizer_init": False,
        "attention_backend": attention_backend,
    }
    control_engine = _create_engine(
        model_path,
        max_total_tokens=control_max_total_tokens,
        **common_engine_args,
    )
    try:
        control = _generate_retraction_batch(
            control_engine,
            prompts,
            max_new_tokens,
            ignore_eos=False,
            min_new_tokens=max_new_tokens,
        )
    finally:
        control_engine.shutdown()

    pressure_engine = _create_engine(
        model_path,
        max_total_tokens=pressure_max_total_tokens,
        schedule_conservativeness=schedule_conservativeness,
        **common_engine_args,
    )
    try:
        pressure = _generate_retraction_batch(
            pressure_engine,
            prompts,
            max_new_tokens,
            ignore_eos=False,
            min_new_tokens=max_new_tokens,
        )
        flush = _control_result(
            "post-organic-retraction cache flush", pressure_engine.flush_cache()
        )
    finally:
        pressure_engine.shutdown()

    retraction_counts = [request["num_retractions"] for request in pressure]
    if not any(count > 0 for count in retraction_counts):
        raise AssertionError(
            "SGLang produced no organic request retractions; increase pressure: "
            f"{retraction_counts}"
        )
    mismatches = [
        index
        for index, (control_request, pressure_request) in enumerate(
            zip(control, pressure, strict=True)
        )
        if control_request["output_ids"] != pressure_request["output_ids"]
    ]
    retracted_indices = [
        index for index, count in enumerate(retraction_counts) if count > 0
    ]
    retracted_mismatches = sorted(set(mismatches) & set(retracted_indices))
    non_retracted_mismatches = sorted(set(mismatches) - set(retracted_indices))
    if retracted_mismatches:
        raise AssertionError(
            "Organic KDA retraction/resume changed greedy output for retracted "
            f"request indices {retracted_mismatches}: "
            f"control={control!r}, pressure={pressure!r}"
        )
    return {
        "request_count": request_count,
        "prompt_lengths": list(map(len, prompts)),
        "max_new_tokens": max_new_tokens,
        "pressure_max_total_tokens": pressure_max_total_tokens,
        "control_max_total_tokens": control_max_total_tokens,
        "schedule_conservativeness": schedule_conservativeness,
        "retraction_counts": retraction_counts,
        "retracted_indices": retracted_indices,
        "non_retracted_mismatch_indices": non_retracted_mismatches,
        "control": control,
        "pressure": pressure,
        "flush": flush,
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
    probe.add_argument(
        "--cancellation-probe",
        action="store_true",
        help="Cancel running and queued requests, then validate state recovery.",
    )
    probe.add_argument(
        "--retraction-probe",
        action="store_true",
        help="Force scheduler retraction and validate resumed KDA output.",
    )
    probe.add_argument(
        "--organic-retraction-probe",
        action="store_true",
        help="Cause real KV pressure and validate resumed KDA output.",
    )
    parser.add_argument(
        "--retraction-interval",
        type=int,
        default=7,
        help="Scheduler forwards between forced retractions in the retraction probe.",
    )
    parser.add_argument(
        "--pressure-max-total-tokens",
        type=int,
        default=1100,
        help="KV-pool token cap for the organic retraction probe.",
    )
    parser.add_argument(
        "--pressure-request-count",
        type=int,
        default=8,
        help="Batch width for the organic retraction probe.",
    )
    parser.add_argument(
        "--schedule-conservativeness",
        type=float,
        default=0.1,
        help="Admission reserve multiplier for the organic retraction probe.",
    )
    parser.add_argument(
        "--organic-attention-backend",
        default="torch_native",
        help="Full-attention backend for the organic retraction probe.",
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
    if args.organic_retraction_probe:
        prompt_length = args.prompt_length if args.prompt_length is not None else 100
        report = run_organic_retraction_probe(
            args.model,
            prompt_length,
            args.max_new_tokens,
            request_count=args.pressure_request_count,
            pressure_max_total_tokens=args.pressure_max_total_tokens,
            schedule_conservativeness=args.schedule_conservativeness,
            chunked_prefill_size=args.chunked_prefill_size,
            mamba_radix_cache_strategy=args.mamba_radix_cache_strategy,
            attention_backend=args.organic_attention_backend,
            context_length=args.context_length,
        )
    elif args.retraction_probe:
        prompt_length = args.prompt_length if args.prompt_length is not None else 300
        report = run_retraction_probe(
            args.model,
            prompt_length,
            args.max_new_tokens,
            retraction_interval=args.retraction_interval,
            chunked_prefill_size=args.chunked_prefill_size,
            mamba_radix_cache_strategy=args.mamba_radix_cache_strategy,
            context_length=args.context_length,
        )
    elif args.cancellation_probe:
        prompt_length = args.prompt_length if args.prompt_length is not None else 300
        report = run_cancellation_probe(
            args.model,
            prompt_length,
            args.max_new_tokens,
            chunked_prefill_size=args.chunked_prefill_size,
            mamba_radix_cache_strategy=args.mamba_radix_cache_strategy,
            context_length=args.context_length,
        )
    elif args.mixed_chunked_prefill_probe:
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
