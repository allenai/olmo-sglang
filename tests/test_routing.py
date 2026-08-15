import torch

from olmo_sglang.routing import olmo3_moe_topk


def test_olmo_router_normalizes_and_restores_scale():
    logits = torch.tensor([[1.0, 3.0, 2.0]])
    weights, indices = olmo3_moe_topk(
        torch.zeros(1, 4),
        logits,
        2,
        True,
        normalize_expert_weights=1.0,
        restore_weight_scale=True,
        original_num_experts_per_tok=None,
    )

    assert indices.tolist() == [[1, 2]]
    torch.testing.assert_close(weights.sum(dim=-1), torch.tensor([2.0]))


def test_olmo_router_applies_original_topk_correction():
    weights, _ = olmo3_moe_topk(
        torch.zeros(1, 4),
        torch.tensor([[1.0, 3.0, 2.0]]),
        2,
        True,
        normalize_expert_weights=1.0,
        restore_weight_scale=True,
        original_num_experts_per_tok=8,
    )

    torch.testing.assert_close(weights.sum(dim=-1), torch.tensor([4.0]))
