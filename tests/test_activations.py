import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from olmo_sglang.activations import (
    install_sglang_moe_activation_fallback,
    native_silu_and_mul,
)


def test_native_silu_and_mul_supports_unaligned_olmo_width():
    gate_up = torch.arange(28, dtype=torch.float32).reshape(2, 14)

    output = native_silu_and_mul(gate_up)

    gate, up = gate_up.chunk(2, dim=-1)
    torch.testing.assert_close(output, F.silu(gate) * up)
    assert output.shape == (2, 7)


def test_native_silu_and_mul_preserves_filtered_rows():
    gate_up = torch.randn(4, 14)
    out = torch.full((4, 7), float("nan"))
    expert_ids = torch.tensor([0, -1], dtype=torch.int32)

    result = native_silu_and_mul(gate_up, out, expert_ids, expert_step=2)

    assert result is out
    expected = F.silu(gate_up[:, :7]) * gate_up[:, 7:]
    torch.testing.assert_close(out[:2], expected[:2])
    assert torch.isnan(out[2:]).all()


def test_install_sglang_moe_activation_fallback(monkeypatch: pytest.MonkeyPatch):
    calls = []

    def original(*args):
        calls.append(args)
        return torch.full((1, 8), 3.0)

    fused_moe = SimpleNamespace(silu_and_mul=original)
    module_names = (
        "sglang",
        "sglang.srt",
        "sglang.srt.layers",
        "sglang.srt.layers.moe",
        "sglang.srt.layers.moe.moe_runner",
    )
    for name in module_names:
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    triton_utils_name = "sglang.srt.layers.moe.moe_runner.triton_utils"
    triton_utils = ModuleType(triton_utils_name)
    triton_utils.fused_moe = fused_moe
    monkeypatch.setitem(sys.modules, triton_utils_name, triton_utils)

    install_sglang_moe_activation_fallback()

    unaligned = torch.randn(1, 14)
    actual = fused_moe.silu_and_mul(unaligned)
    expected = F.silu(unaligned[:, :7]) * unaligned[:, 7:]
    torch.testing.assert_close(actual, expected)
    assert not calls

    aligned = torch.randn(1, 16)
    assert torch.equal(fused_moe.silu_and_mul(aligned), torch.full((1, 8), 3.0))
    assert len(calls) == 1
