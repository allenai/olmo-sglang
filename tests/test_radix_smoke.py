from types import SimpleNamespace

import pytest

from olmo_sglang.validation.radix import (
    _control_result,
    _finish_reason,
    _mixed_prompts,
    _pressure_prompts,
)


def test_control_result_accepts_structured_and_tuple_responses():
    assert _control_result("flush", SimpleNamespace(success=True, message="ok")) == {
        "success": True,
        "message": "ok",
    }
    assert _control_result("update", (True, "updated", 0)) == {
        "success": True,
        "message": "updated",
    }


def test_control_result_rejects_failed_responses():
    with pytest.raises(AssertionError, match="SGLang flush failed: busy"):
        _control_result("flush", SimpleNamespace(success=False, message="busy"))
    with pytest.raises(AssertionError, match="SGLang update failed: rejected"):
        _control_result("update", (False, "rejected", 1))


def test_mixed_prompts_share_prefix_and_vary_lengths():
    prompts = _mixed_prompts(prompt_length=300, tracked_prefix_length=256)

    assert [len(prompt) for prompt in prompts] == [268, 300, 332]
    assert prompts[0][:256] == prompts[1][:256] == prompts[2][:256]
    assert len({tuple(prompt[256:]) for prompt in prompts}) == 3


def test_mixed_prompts_reject_too_short_center():
    with pytest.raises(ValueError, match="must exceed tracked_prefix_length"):
        _mixed_prompts(prompt_length=288, tracked_prefix_length=256)


def test_pressure_prompts_are_unique_and_centered():
    prompts = _pressure_prompts(prompt_length=100, request_count=8)

    assert [len(prompt) for prompt in prompts] == [86, 90, 94, 98, 102, 106, 110, 114]
    assert len({prompt[0] for prompt in prompts}) == 8


@pytest.mark.parametrize("request_count", [0, 1])
def test_pressure_prompts_require_multiple_requests(request_count):
    with pytest.raises(ValueError, match="at least two"):
        _pressure_prompts(prompt_length=100, request_count=request_count)


def test_pressure_prompts_reject_too_short_center():
    with pytest.raises(ValueError, match="too short"):
        _pressure_prompts(prompt_length=2, request_count=8)


def test_finish_reason_extracts_structured_type():
    assert (
        _finish_reason({"meta_info": {"finish_reason": {"type": "abort"}}}) == "abort"
    )
    assert _finish_reason({"meta_info": {"finish_reason": None}}) is None
    assert _finish_reason({}) is None
