import pytest
import torch

from olmo_sglang.validation.core_reference import compare_logprobs, forced_prefix_cases


def test_core_prefixes_follow_hf_tokens_without_retokenization_or_padding():
    references = [
        {"tokens": [2, 0], "logprobs": [[-3.0, -2.0, -1.0], [-1.0, -2.0, -3.0]]},
        {"tokens": [1], "logprobs": [[-3.0, -1.0, -2.0]]},
    ]
    cases = forced_prefix_cases([[9, 8], [7, 6, 5]], references)
    assert [case["input_ids"] for case in cases] == [[9, 8], [9, 8, 2], [7, 6, 5]]
    assert [case["hf_next_token"] for case in cases] == [2, 0, 1]
    assert len({case["input_ids_sha256"] for case in cases}) == 3
    assert references[0]["tokens"] == [2, 0]
    with pytest.raises(ValueError, match="align"):
        forced_prefix_cases([[9]], [{"tokens": [0], "logprobs": []}])
    with pytest.raises(ValueError, match="one reference"):
        forced_prefix_cases([[9]], [])


def test_core_error_includes_low_probability_tokens_and_rejects_shape_or_nan():
    metrics = compare_logprobs(torch.tensor([-1.0, -2.0, -9.0]), [-1.0, -2.0, -10.0])
    assert metrics["max_abs_logprob_error"] == 1.0
    assert metrics["mean_abs_logprob_error"] == pytest.approx(1 / 3)
    assert metrics["core_next_token"] == 0
    with pytest.raises(ValueError, match="dimensions"):
        compare_logprobs(torch.tensor([-1.0]), [-1.0, -2.0])
    with pytest.raises(ValueError, match="finite"):
        compare_logprobs(torch.tensor([float("nan")]), [-1.0])
