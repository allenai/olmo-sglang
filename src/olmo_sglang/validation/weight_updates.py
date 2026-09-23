"""Checked, bucketed weight publication for the standalone serving probes."""

from olmo_sglang.validation.radix import _control_result


def update_weights_in_buckets(engine, named_tensors):
    """Require success at each boundary and close every opened update session."""
    if not named_tensors:
        raise ValueError("At least one changed tensor is required")
    responses = [_control_result("begin weight update", engine.begin_weight_update())]
    try:
        for index, tensor in enumerate(named_tensors):
            responses.append(
                _control_result(
                    f"weight update bucket {index} ({tensor[0]})",
                    engine.update_weights_from_tensor(
                        [tensor], flush_cache=index == len(named_tensors) - 1
                    ),
                )
            )
    finally:
        responses.append(
            _control_result("end weight update", engine.end_weight_update())
        )
    return responses
