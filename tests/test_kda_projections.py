"""Packed weight layout, state-dict stability and in-place refresh contracts."""

import pytest
import torch
from torch import nn

from olmo_sglang.kda import projections


def test_projection_mode_validation(monkeypatch):
    monkeypatch.delenv(projections.PROJECTION_VARIABLE, raising=False)
    assert projections.projection_mode() == "torch"
    monkeypatch.setenv(projections.PROJECTION_VARIABLE, " Packed ")
    assert projections.projection_mode() == "packed"
    monkeypatch.setenv(projections.PROJECTION_VARIABLE, "unknown")
    with pytest.raises(ValueError, match=projections.PROJECTION_VARIABLE):
        projections.projection_mode()


def test_packed_weights_preserve_parameter_objects_and_live_partial_loads():
    model = nn.ModuleList([nn.Linear(8, rows, bias=False) for rows in [12, 4, 2, 4]])
    names = list(model.state_dict())
    objects = list(model.parameters())
    marker = object()
    model[0].weight.weight_loader = marker
    packed = projections.pack_weights(model)
    pointers = [weight.data_ptr() for weight in objects]
    assert list(model.parameters()) == objects
    assert model[0].weight.weight_loader is marker
    assert list(model.state_dict()) == names
    assert all(
        weight.untyped_storage().data_ptr() == packed.data_ptr() for weight in objects
    )
    # Exact small integers avoid conflating layout correctness with GEMM rounding.
    for i, weight in enumerate(objects):
        with torch.no_grad():
            weight.fill_(i + 1)
    hidden = torch.ones(3, 8)
    actual = projections.project(hidden, packed, [12, 4, 2, 4])
    for i, output in enumerate(actual):
        torch.testing.assert_close(
            output, torch.full_like(output, 8 * (i + 1)), rtol=0, atol=0
        )
    model.load_state_dict({"1.weight": torch.full((4, 8), 7.0)}, strict=False)
    after = projections.project(hidden, packed, [12, 4, 2, 4])
    torch.testing.assert_close(
        after[1], torch.full_like(after[1], 56.0), rtol=0, atol=0
    )
    for i in [0, 2, 3]:
        torch.testing.assert_close(after[i], actual[i], rtol=0, atol=0)
    assert [weight.data_ptr() for weight in objects] == pointers


def test_packing_rejects_bias_or_mismatched_width():
    with pytest.raises(ValueError, match="bias-free"):
        projections.pack_weights([nn.Linear(8, 4)])
    with pytest.raises(ValueError, match="compatible"):
        projections.pack_weights(
            [nn.Linear(8, 4, bias=False), nn.Linear(9, 4, bias=False)]
        )
