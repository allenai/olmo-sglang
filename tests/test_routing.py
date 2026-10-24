from types import SimpleNamespace

import pytest
import torch

from olmo_sglang import routing
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


def test_routing_kernel_mode_defaults_to_torch_and_rejects_unknown(monkeypatch):
    monkeypatch.delenv(ROUTING_KERNELS_VARIABLE, raising=False)
    assert routing_kernel_mode() == "torch"
    monkeypatch.setenv(ROUTING_KERNELS_VARIABLE, " Torch ")
    assert routing_kernel_mode() == "torch"
    monkeypatch.setenv(ROUTING_KERNELS_VARIABLE, " Fused ")
    assert routing_kernel_mode() == "fused"
    monkeypatch.setenv(ROUTING_KERNELS_VARIABLE, "cuda")
    with pytest.raises(ValueError, match=ROUTING_KERNELS_VARIABLE):
        routing_kernel_mode()


@pytest.mark.parametrize("mode", [None, "torch", "fused"])
@pytest.mark.parametrize("is_cuda", [False, True])
@pytest.mark.parametrize("normalize", [None, 1.0, 2.0])
def test_fused_routing_requires_opt_in_and_supported_inputs(
    monkeypatch, mode, is_cuda, normalize
):
    if mode is None:
        monkeypatch.delenv(ROUTING_KERNELS_VARIABLE, raising=False)
    else:
        monkeypatch.setenv(ROUTING_KERNELS_VARIABLE, mode)
    supported = routing._fused_topk_supported(
        SimpleNamespace(is_cuda=is_cuda), normalize
    )
    assert supported == (mode == "fused" and is_cuda and normalize in (None, 1.0))


def _route(device, mode, monkeypatch, original=None, normalize=1.0, restore=True):
    if mode is None:
        monkeypatch.delenv(ROUTING_KERNELS_VARIABLE, raising=False)
    else:
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
        restore_weight_scale=restore,
        original_num_experts_per_tok=original,
    )


def test_cpu_fused_request_keeps_tensor_path(monkeypatch):
    def unexpected_fused_call(*args, **kwargs):
        pytest.fail("CPU routing must not load or call the fused runtime")

    monkeypatch.setattr(routing, "_fused_olmo3_moe_topk", unexpected_fused_call)
    expected = _route("cpu", "torch", monkeypatch)
    for mode in (None, "fused"):
        actual = _route("cpu", mode, monkeypatch)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_default_cuda_routing_keeps_tensor_path(monkeypatch):
    def unexpected_fused_call(*args, **kwargs):
        pytest.fail("Default CUDA routing must not call the fused runtime")

    monkeypatch.setattr(routing, "_fused_olmo3_moe_topk", unexpected_fused_call)
    expected = _route("cuda", "torch", monkeypatch)
    actual = _route("cuda", None, monkeypatch)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("original", [None, 32])
@pytest.mark.parametrize("normalize", [None, 1.0])
@pytest.mark.parametrize("restore", [False, True])
def test_fused_routing_matches_tensor_routing_on_cuda(
    monkeypatch, original, normalize, restore
):
    fused_weights, fused_ids = _route(
        "cuda", "fused", monkeypatch, original, normalize, restore
    )
    torch_weights, torch_ids = _route(
        "cuda", "torch", monkeypatch, original, normalize, restore
    )

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
    if normalize is not None:
        expected_sum = (16.0 if restore else 1.0) * (
            (original / 16) ** 0.5 if original else 1.0
        )
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


def test_routing_tail_mode_validation(monkeypatch):
    monkeypatch.delenv(routing.ROUTING_TAIL_VARIABLE, raising=False)
    assert routing.routing_tail_mode() == "torch"
    monkeypatch.setenv(routing.ROUTING_TAIL_VARIABLE, " Fused ")
    assert routing.routing_tail_mode() == "fused"
    monkeypatch.setenv(routing.ROUTING_TAIL_VARIABLE, "invalid")
    with pytest.raises(ValueError, match=routing.ROUTING_TAIL_VARIABLE):
        routing.routing_tail_mode()


@pytest.mark.parametrize("normalize", [None, 1.0, 2.0])
def test_routing_tail_cpu_request_keeps_tensor_path(monkeypatch, normalize):
    monkeypatch.setenv(routing.ROUTING_TAIL_VARIABLE, "torch")
    expected = _route("cpu", "torch", monkeypatch, normalize=normalize)
    monkeypatch.setenv(routing.ROUTING_TAIL_VARIABLE, "fused")
    actual = _route("cpu", "torch", monkeypatch, normalize=normalize)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("original", [None, 8, 32])
@pytest.mark.parametrize("normalize", [None, 1.0, 2.0])
@pytest.mark.parametrize("restore", [False, True])
def test_routing_tail_cuda_arithmetic(monkeypatch, original, normalize, restore):
    monkeypatch.setenv(routing.ROUTING_TAIL_VARIABLE, "torch")
    expected = _route("cuda", "torch", monkeypatch, original, normalize, restore)
    monkeypatch.setenv(routing.ROUTING_TAIL_VARIABLE, "fused")
    actual = _route("cuda", "torch", monkeypatch, original, normalize, restore)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
