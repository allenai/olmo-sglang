import torch
import torch.nn.functional as F

from olmo_sglang.activations import native_silu_and_mul


def test_native_silu_and_mul_supports_unaligned_olmo_width():
    gate_up = torch.arange(28, dtype=torch.float32).reshape(2, 14)

    output = native_silu_and_mul(gate_up)

    gate, up = gate_up.chunk(2, dim=-1)
    torch.testing.assert_close(output, F.silu(gate) * up)
    assert output.shape == (2, 7)
