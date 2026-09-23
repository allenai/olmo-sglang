from copy import deepcopy

import pytest

from olmo_sglang.validation.cache_updates import compare_cache_outputs


def outputs():
    return {"cold": [{"tokens": [1, 2], "logprobs": [-0.2, -0.3]}]}


def test_cache_comparison_checks_tokens_and_every_logprob():
    expected = outputs()
    assert compare_cache_outputs(expected, expected) == {
        "token_parity": True,
        "max_logprob_error": 0,
    }
    actual = deepcopy(expected)
    actual["cold"][0]["tokens"][1] = 3
    actual["cold"][0]["logprobs"][1] -= 0.1
    result = compare_cache_outputs(actual, expected)
    assert not result["token_parity"]
    assert result["max_logprob_error"] == pytest.approx(0.1)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_cache_comparison_rejects_nonfinite_values(value):
    actual = outputs()
    actual["cold"][0]["logprobs"][1] = value
    with pytest.raises(ValueError, match="Non-finite"):
        compare_cache_outputs(actual, outputs())


@pytest.mark.parametrize(
    "actual",
    [
        {},
        {"cold": []},
        {"cold": [{"tokens": [], "logprobs": []}]},
        {"cold": [{"tokens": [1, 2], "logprobs": [-0.2]}]},
        {"cold": [{"tokens": [1], "logprobs": [-0.2]}]},
    ],
)
def test_cache_comparison_rejects_missing_results(actual):
    with pytest.raises(ValueError):
        compare_cache_outputs(actual, outputs())
