"""Storage, publication and numerical controls for opt-in Core execution."""

from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from olmo_sglang import core_compat


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_permutation_orders_nondefault_stream_and_growing_workspace():
    utils = pytest.importorskip("olmo_core.nn.moe.utils")
    stream = torch.cuda.Stream()
    for tokens in [16, 1, 81, 4, 257, 1, 83]:
        with torch.cuda.stream(stream):
            # Delay input production so a default-stream sort cannot safely race it.
            torch.cuda._sleep(2_000_000)
            value = (
                torch.arange(tokens * 128, device="cuda")
                .reshape(tokens, 128)
                .bfloat16()
            )
            routes = torch.randint(
                0, 512, (tokens, 16), device="cuda", dtype=torch.int32
            )
            actual, reverse = core_compat.permute_on_default_stream(
                utils.moe_permute_no_compile, value, routes
            )
            order = routes.flatten().argsort(stable=True) // routes.shape[1]
            expected = value.index_select(0, order)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert reverse.numel() == routes.numel()
    stream.synchronize()


def test_default_off_and_invalid_setting(monkeypatch):
    monkeypatch.delenv(core_compat.ENVIRONMENT_VARIABLE, raising=False)
    assert not core_compat.enabled()
    monkeypatch.setenv(core_compat.ENVIRONMENT_VARIABLE, "1")
    assert core_compat.enabled()
    monkeypatch.setenv(core_compat.ENVIRONMENT_VARIABLE, "typo")
    with pytest.raises(ValueError):
        core_compat.enabled()


def test_fused_rounding_has_explicit_tensor_reference(monkeypatch):
    monkeypatch.delenv("OLMO_SGLANG_ROUNDING_KERNELS", raising=False)
    assert core_compat.fused_rounding_enabled()
    monkeypatch.setenv("OLMO_SGLANG_ROUNDING_KERNELS", "torch")
    assert not core_compat.fused_rounding_enabled()
    monkeypatch.setenv("OLMO_SGLANG_ROUNDING_KERNELS", "typo")
    with pytest.raises(ValueError, match="ROUNDING_KERNELS"):
        core_compat.fused_rounding_enabled()


def test_runtime_rejects_unsupported_execution(monkeypatch):
    monkeypatch.setenv(core_compat.ENVIRONMENT_VARIABLE, "1")
    config = SimpleNamespace(
        attention_bias=False, use_rope=False, layer_types=["full_attention"]
    )
    parallel = SimpleNamespace(tp_size=1, moe_ep_size=1)
    args = SimpleNamespace(
        dtype="bfloat16",
        cuda_graph_backend_decode="disabled",
        cuda_graph_backend_prefill="disabled",
    )
    core_compat.validate_runtime(config, parallel, args, None)
    for obj, key, value in [
        (parallel, "tp_size", 2),
        (parallel, "moe_ep_size", 2),
        (args, "cuda_graph_backend_decode", "full"),
        (args, "dtype", "float16"),
    ]:
        old = getattr(obj, key)
        setattr(obj, key, value)
        with pytest.raises(ValueError):
            core_compat.validate_runtime(config, parallel, args, None)
        setattr(obj, key, old)


def test_rounding_mode_keeps_graphs_and_default_attention(monkeypatch):
    monkeypatch.setenv(core_compat.ENVIRONMENT_VARIABLE, "rounding")
    assert core_compat.rounding_enabled()
    assert core_compat.norms_enabled()
    assert not core_compat.enabled()
    config = SimpleNamespace(
        attention_bias=False, use_rope=False, layer_types=["linear_attention"]
    )
    parallel = SimpleNamespace(tp_size=1, moe_ep_size=1)
    args = SimpleNamespace(
        dtype="bfloat16",
        cuda_graph_backend_decode="full",
        cuda_graph_backend_prefill="disabled",
    )
    core_compat.validate_runtime(config, parallel, args, None)
    parallel.tp_size = 2
    with pytest.raises(ValueError, match="TP1"):
        core_compat.validate_runtime(config, parallel, args, None)


def test_dense_layout_and_publication_preserve_used_storage():
    torch.manual_seed(42)
    model = core_compat.CoreDenseMLP(16, 32).bfloat16()
    gate, up = [torch.randn(32, 16).bfloat16() for _ in range(2)]
    down = torch.randn(16, 32).bfloat16()
    model.load_up_gate(model.gate_up_proj.weight, gate, 0)
    model.load_up_gate(model.gate_up_proj.weight, up, 1)
    with torch.no_grad():
        model.down_proj.weight.copy_(down)
    pointers = [p.data_ptr() for p in model.parameters()]
    x = torch.randn(7, 16).bfloat16()

    def reference():
        u, g = (x @ torch.cat((up, gate)).t().contiguous()).chunk(2, -1)
        return torch.bmm(
            (u * F.silu(g)).unsqueeze(0), down.t().contiguous().unsqueeze(0)
        ).squeeze(0)

    torch.testing.assert_close(model(x), reference(), rtol=0, atol=0)
    old = model(x).clone()
    gate.add_(1)
    model.load_up_gate(model.gate_up_proj.weight, gate, 0)
    assert pointers == [p.data_ptr() for p in model.parameters()]
    assert not torch.equal(old, model(x))
    torch.testing.assert_close(model(x), reference(), rtol=0, atol=0)


def test_expert_hf_and_fused_publications_have_identical_storage():
    pytest.importorskip("olmo_core")
    a = core_compat.CoreExperts(3, 16, 32).bfloat16()
    b = core_compat.CoreExperts(3, 16, 32).bfloat16()
    pointers = [p.data_ptr() for p in a.parameters()]
    for _ in range(2):
        gate, up = [torch.randn(3, 32, 16).bfloat16() for _ in range(2)]
        down = torch.randn(3, 16, 32).bfloat16()
        for index in range(3):
            for value, shard, param in [
                (gate, "w1", a.w13_weight),
                (up, "w3", a.w13_weight),
                (down, "w2", a.w2_weight),
            ]:
                a.weight_loader(
                    param, value[index], "weight", shard_id=shard, expert_id=index
                )
        b.weight_loader_fused(
            b.w13_weight, torch.cat((gate, up), 1), "weight", shard_id="w13"
        )
        b.weight_loader_fused(b.w2_weight, down, "weight", shard_id="w2")
        assert pointers == [p.data_ptr() for p in a.parameters()]
        torch.testing.assert_close(a.w13_weight, b.w13_weight, rtol=0, atol=0)
        torch.testing.assert_close(a.w2_weight, b.w2_weight, rtol=0, atol=0)
        torch.testing.assert_close(
            a.w13_weight, torch.cat((up, gate), 1), rtol=0, atol=0
        )
        assert a.w2_weight.transpose(1, 2).is_contiguous()
