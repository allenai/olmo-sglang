import pytest
import torch
from torch import nn

from olmo_sglang.validation.layerwise import (
    capture_block,
    compare_captures,
    tensor_error,
)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_attention_layernorm = nn.Identity()
        self.self_attn = nn.Linear(4, 4, bias=False)
        self.post_attention_layernorm = nn.Identity()
        self.pre_feedforward_layernorm = nn.Identity()
        self.post_feedforward_layernorm = nn.Identity()

    def forward(self, hidden_states):
        x = hidden_states
        x = x + self.post_attention_layernorm(
            self.self_attn(self.pre_attention_layernorm(x))
        )
        return x + self.post_feedforward_layernorm(
            self.pre_feedforward_layernorm(x) * 2
        )


def test_capture_preserves_math_and_removes_hooks():
    block = Block()
    x = torch.randn(1, 3, 4)
    expected = block(x)
    with capture_block(block, backend="hf") as captured:
        actual = block(hidden_states=x)
    assert torch.equal(actual, expected)
    assert len(captured) == 12
    assert torch.equal(captured["block.input"], x)
    assert all(not module._forward_hooks for module in block.modules())
    assert all(not module._forward_pre_hooks for module in block.modules())
    result = compare_captures(captured, captured)
    assert all(value["max_abs"] == 0 for value in result.values())
    actual.add_(1)
    assert torch.equal(captured["block.output"], expected)


def test_changed_boundary_is_not_hidden_by_aggregate():
    expected = {"layer": torch.ones(2, 3)}
    actual = {"layer": expected["layer"].clone()}
    actual["layer"][0, 1] += 0.25
    result = compare_captures(actual, expected)["layer"]
    assert result["max_abs"] == 0.25
    assert result["mean_abs"] == pytest.approx(0.25 / 6)
    assert result["relative_l2"] > 0
    with pytest.raises(ValueError, match="boundaries"):
        compare_captures({}, expected)


def test_bad_activations_are_rejected():
    with pytest.raises(ValueError, match="shape"):
        tensor_error(torch.ones(2), torch.ones(3))
    with pytest.raises(ValueError, match="Non-finite"):
        tensor_error(torch.tensor([float("nan")]), torch.ones(1))
    assert tensor_error(torch.ones(2), torch.zeros(2))["relative_l2"] > 1e20


def test_capture_cleanup_on_forward_error():
    block = Block()
    with pytest.raises(RuntimeError):
        with capture_block(block, backend="hf"):
            block(torch.ones(7))
    assert all(not module._forward_hooks for module in block.modules())
