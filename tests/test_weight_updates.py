from unittest.mock import Mock

import pytest

from olmo_sglang.validation.weight_updates import update_weights_in_buckets


def _engine():
    engine = Mock()
    engine.begin_weight_update.return_value = (True, "opened")
    engine.update_weights_from_tensor.return_value = (True, "updated")
    engine.end_weight_update.return_value = (True, "closed")
    return engine


def test_successful_update_flushes_only_the_final_bucket():
    engine = _engine()
    tensors = [("q_gain", object()), ("k_gain", object()), ("scale", object())]
    responses = update_weights_in_buckets(engine, tensors)
    assert len(responses) == 5
    assert all(response["success"] for response in responses)
    assert [
        call.kwargs["flush_cache"]
        for call in engine.update_weights_from_tensor.call_args_list
    ] == [False, False, True]
    assert [
        call.args[0] for call in engine.update_weights_from_tensor.call_args_list
    ] == [[tensor] for tensor in tensors]
    engine.end_weight_update.assert_called_once_with()


@pytest.mark.parametrize("failure", ["begin", "bucket", "end"])
def test_failed_control_response_cannot_pass_qualification(failure):
    engine = _engine()
    method = {
        "begin": engine.begin_weight_update,
        "bucket": engine.update_weights_from_tensor,
        "end": engine.end_weight_update,
    }[failure]
    method.return_value = (False, "rejected")
    with pytest.raises(AssertionError, match="rejected"):
        update_weights_in_buckets(engine, [("q_gain", object())])
    if failure == "begin":
        engine.update_weights_from_tensor.assert_not_called()
        engine.end_weight_update.assert_not_called()
    else:
        engine.end_weight_update.assert_called_once_with()


def test_transport_error_closes_session_without_publishing_later_buckets():
    engine = _engine()
    engine.update_weights_from_tensor.side_effect = RuntimeError("transport failed")
    with pytest.raises(RuntimeError, match="transport failed"):
        update_weights_in_buckets(engine, [("q_gain", object()), ("k_gain", object())])
    engine.update_weights_from_tensor.assert_called_once()
    engine.end_weight_update.assert_called_once_with()


def test_empty_update_does_not_open_a_session():
    engine = _engine()
    with pytest.raises(ValueError, match="changed tensor"):
        update_weights_in_buckets(engine, [])
    engine.begin_weight_update.assert_not_called()
