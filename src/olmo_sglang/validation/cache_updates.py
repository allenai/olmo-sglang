"""Compare matched cache/update probe phases without hiding invalid results."""

import math


def compare_cache_outputs(actual, expected):
    if actual.keys() != expected.keys():
        raise ValueError("Cache probe phases differ")
    pairs = [
        (a, b)
        for phase in actual
        for a, b in zip(actual[phase], expected[phase], strict=True)
    ]
    errors = []
    for a, b in pairs:
        for output in (a, b):
            if not output["tokens"] or len(output["tokens"]) != len(output["logprobs"]):
                raise ValueError("Expected one log probability per generated token")
            if not all(math.isfinite(value) for value in output["logprobs"]):
                raise ValueError("Non-finite output log probability")
        errors.extend(
            abs(x - y) for x, y in zip(a["logprobs"], b["logprobs"], strict=True)
        )
    if not errors:
        raise ValueError("Cache probe produced no outputs")
    return {
        "token_parity": all(a["tokens"] == b["tokens"] for a, b in pairs),
        "max_logprob_error": max(errors),
    }
