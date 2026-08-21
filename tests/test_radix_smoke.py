from types import SimpleNamespace

import pytest

from olmo_sglang.radix_smoke import _control_result, _finish_reason, _mixed_prompts


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


def test_finish_reason_extracts_structured_type():
    assert (
        _finish_reason({"meta_info": {"finish_reason": {"type": "abort"}}}) == "abort"
    )
    assert _finish_reason({"meta_info": {"finish_reason": None}}) is None
    assert _finish_reason({}) is None
