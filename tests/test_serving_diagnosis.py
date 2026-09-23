import copy

import pytest

from olmo_sglang.validation.serving_diagnosis import compare_modes


def modes():
    return [
        {
            "graphs": graphs,
            "chunk": chunk,
            "forced": [
                [
                    {
                        "input_ids": [1, 2],
                        "checked_top_logprobs": [
                            {
                                "token": 4,
                                "sglang": -0.5 + graphs * 0.125 + (chunk == 32) * 0.25,
                            },
                            {"token": 5 if graphs else 6, "sglang": -3},
                        ],
                    }
                ]
            ],
        }
        for graphs in (False, True)
        for chunk in (128, 32)
    ]


def test_isolates_single_variable_and_reports_missing_tokens():
    result = compare_modes(modes())
    assert len(result) == 4
    for comparison in result:
        row = comparison["records"][0]
        if comparison["variable"] == "chunk":
            assert row["compared_tokens"] == 2
            assert row["max_abs_common_token_logprob_difference"] == 0.25
            assert row["first_only_tokens"] == []
        else:
            assert row["compared_tokens"] == 1
            assert row["first_only_tokens"] == [6]
            assert row["second_only_tokens"] == [5]
            assert row["max_abs_common_token_logprob_difference"] == 0.125


def test_rejects_misaligned_prefixes():
    data = modes()
    data[1] = copy.deepcopy(data[1])
    data[1]["forced"][0][0]["input_ids"] = [3, 4]
    with pytest.raises(ValueError, match="prefixes"):
        compare_modes(data)


def test_disjoint_top_tokens_have_no_invented_zero_error():
    data = modes()[:2]
    data[1]["forced"][0][0]["checked_top_logprobs"] = [{"token": 8, "sglang": -1}]
    result = compare_modes(data)[0]["records"][0]
    assert result["compared_tokens"] == 0
    assert result["max_abs_common_token_logprob_difference"] is None
