"""The router projection is stored in FP32 and loads BF16 checkpoint tensors exactly."""

import torch
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.model_loader.weight_utils import default_weight_loader


def test_router_gate_fp32_storage_loads_bf16_tensor_exactly():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        gate = ReplicatedLinear(16, 8, bias=False, params_dtype=torch.float32)
    finally:
        torch.set_default_dtype(previous)
    assert gate.weight.dtype == torch.float32

    checkpoint = torch.randn(8, 16, generator=torch.Generator().manual_seed(3)).to(
        torch.bfloat16
    )
    loader = getattr(gate.weight, "weight_loader", default_weight_loader)
    loader(gate.weight, checkpoint)

    assert gate.weight.dtype == torch.float32
    assert torch.equal(gate.weight.data, checkpoint.float())
