"""Routed experts must preserve the decoder residual and shared-expert input."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from olmo_sglang.models import olmo3_moe


@pytest.mark.parametrize("shared", [False, True])
def test_sparse_mlp_preserves_input_across_fused_experts(monkeypatch, shared):
    class Linear(nn.Module):
        def __init__(self, input_size, output_size, **kwargs):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(output_size, input_size))

    class TopK(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

        def forward(self, value, logits):
            return None

    class Experts(nn.Module):
        def __init__(self, *, inplace=True, **kwargs):
            super().__init__()
            self.inplace = inplace

        def forward(self, value, routes):
            output = value * 2
            if self.inplace:
                value.copy_(output)
                return value
            return output

    class Shared(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.seen = None

        def forward(self, value):
            self.seen = value.clone()
            return value * 3

    monkeypatch.setattr(olmo3_moe.core_compat, "enabled", lambda: False)
    monkeypatch.setattr(olmo3_moe.core_compat, "rounding_enabled", lambda: False)
    monkeypatch.setattr(olmo3_moe, "ReplicatedLinear", Linear)
    monkeypatch.setattr(olmo3_moe, "TopK", TopK)
    monkeypatch.setattr(olmo3_moe, "FusedMoE", Experts)
    monkeypatch.setattr(olmo3_moe, "Olmo3MoeDenseMLP", Shared)
    monkeypatch.setattr(olmo3_moe, "_log_ep_parallelism", lambda **kwargs: (0, 2))
    config = SimpleNamespace(
        hidden_size=2,
        n_routed_experts=2,
        num_experts_per_tok=2,
        normalize_expert_weights=1.0,
        restore_weight_scale=True,
        original_num_experts_per_tok=None,
        moe_intermediate_size=2,
        num_hidden_layers=1,
        dense_layers_indices=[],
        shared_expert_intermediate_size=2 if shared else None,
    )
    model = olmo3_moe.Olmo3MoeSparseMLP(
        config, layer_id=0, quant_config=None, prefix="model.layers.0.mlp"
    )
    inputs = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
    original = inputs.clone()
    output = model(inputs)
    # The caller retains inputs as the residual even without a shared expert.
    assert torch.equal(inputs, original)
    assert torch.equal(output, original * (5 if shared else 2))
    if shared:
        assert torch.equal(model.shared_expert.seen, original)
