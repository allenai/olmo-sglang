from types import SimpleNamespace

import pytest

from olmo_sglang.radix_smoke import _control_result


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
