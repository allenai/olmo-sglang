import pytest
import torch

from olmo_sglang.routing import (
    ROUTING_KERNELS_VARIABLE,
    fp32_router_logits,
    olmo3_moe_topk,
    routing_kernel_mode,
)


def test_olmo_router_projection_runs_in_fp32():
    hidden_states = torch.tensor([[1.0039, -0.9961]], dtype=torch.bfloat16)
    weight = torch.tensor([[1.0039, 1.0039], [0.9961, 1.0039]], dtype=torch.bfloat16)

    logits = fp32_router_logits(hidden_states, weight)

    assert logits.dtype == torch.float32
    torch.testing.assert_close(
        logits, torch.nn.functional.linear(hidden_states.float(), weight.float())
    )


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
    assert indices.dtype == torch.int32
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


def test_routing_kernel_mode_defaults_to_fused_and_rejects_unknown(monkeypatch):
    monkeypatch.delenv(ROUTING_KERNELS_VARIABLE, raising=False)
    assert routing_kernel_mode() == "fused"
    monkeypatch.setenv(ROUTING_KERNELS_VARIABLE, " Torch ")
    assert routing_kernel_mode() == "torch"
    monkeypatch.setenv(ROUTING_KERNELS_VARIABLE, "cuda")
    with pytest.raises(ValueError, match=ROUTING_KERNELS_VARIABLE):
        routing_kernel_mode()


def _route(device, mode, monkeypatch, original=None, normalize=1.0):
    monkeypatch.setenv(ROUTING_KERNELS_VARIABLE, mode)
    generator = torch.Generator().manual_seed(7)
    logits = torch.randn(64, 512, generator=generator).to(device)
    hidden = torch.zeros(64, 8, dtype=torch.bfloat16, device=device)
    return olmo3_moe_topk(
        hidden,
        logits,
        16,
        True,
        normalize_expert_weights=normalize,
        restore_weight_scale=True,
        original_num_experts_per_tok=original,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("original", [None, 32])
def test_fused_routing_matches_tensor_routing_on_cuda(monkeypatch, original):
    fused_weights, fused_ids = _route("cuda", "fused", monkeypatch, original)
    torch_weights, torch_ids = _route("cuda", "torch", monkeypatch, original)

    assert fused_ids.dtype == torch_ids.dtype == torch.int32
    # Both paths select the same experts; the order within a row is kernel-specific.
    fused_sorted, fused_order = fused_ids.sort(dim=-1)
    torch_sorted, torch_order = torch_ids.sort(dim=-1)
    assert torch.equal(fused_sorted, torch_sorted)
    torch.testing.assert_close(
        fused_weights.gather(1, fused_order),
        torch_weights.gather(1, torch_order),
        rtol=1e-5,
        atol=1e-6,
    )
    expected_sum = 16.0 * ((original / 16) ** 0.5 if original else 1.0)
    torch.testing.assert_close(
        fused_weights.sum(-1), torch.full((64,), expected_sum, device="cuda")
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_routing_falls_back_to_tensor_path_for_other_norms(monkeypatch):
    weights, _ = _route("cuda", "fused", monkeypatch, normalize=2.0)
    torch.testing.assert_close(
        torch.linalg.vector_norm(weights / 16.0, ord=2, dim=-1),
        torch.ones(64, device="cuda"),
    )


def test_fp32_router_storage_matches_per_step_widening():
    generator = torch.Generator().manual_seed(11)
    hidden_states = torch.randn(5, 16, generator=generator).to(torch.bfloat16)
    weight = torch.randn(8, 16, generator=generator).to(torch.bfloat16)

    stored_fp32 = torch.empty(8, 16, dtype=torch.float32)
    stored_fp32.copy_(weight)  # the loader's copy_ widens exactly

    assert torch.equal(
        fp32_router_logits(hidden_states, stored_fp32),
        fp32_router_logits(hidden_states, weight),
    )
