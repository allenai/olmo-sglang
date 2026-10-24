"""Grouped gate contracts: distinct inputs, bias, original weights, graph refresh."""

from importlib import import_module
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from olmo_sglang.kda import gate_projections


def test_mode_and_cpu_fallback(monkeypatch):
    monkeypatch.delenv(gate_projections.VARIABLE, raising=False)
    assert gate_projections.projection_mode() == "torch"
    monkeypatch.setenv(gate_projections.VARIABLE, " GROUPED ")
    assert gate_projections.projection_mode() == "grouped"
    monkeypatch.setenv(gate_projections.VARIABLE, "invalid")
    with pytest.raises(ValueError, match=gate_projections.VARIABLE):
        gate_projections.projection_mode()
    first = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    second = first.neg()
    wf = torch.ones(5, 4)
    wg = torch.ones(7, 4) * 2
    bias = torch.arange(7, dtype=torch.float32)
    assert not gate_projections.supported(first, second, wf, wg, bias)
    actual = gate_projections.project(first, second, wf, wg, bias)
    for a, b in zip(actual, (F.linear(first, wf), F.linear(second, wg, bias))):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_cuda_grouped_gates_shapes_bias_and_refresh():
    torch.manual_seed(31)
    for rows in [0, 1, 7, 16, 64]:
        for inner, nf, ng in [(32, 37, 63), (256, 1024, 2048)]:
            first = torch.randint(-2, 3, (rows, inner + 16), device="cuda").to(
                torch.bfloat16
            )[:, :inner]
            second = torch.randint(-2, 3, (rows, inner + 32), device="cuda").to(
                torch.bfloat16
            )[:, :inner]
            wf = torch.randint(-2, 3, (nf, inner), device="cuda").to(torch.bfloat16)
            wg = torch.randint(-2, 3, (ng, inner), device="cuda").to(torch.bfloat16)
            bias = torch.randint(-2, 3, (ng,), device="cuda").to(torch.bfloat16)
            inputs = [first, second, wf, wg, bias]
            snapshots = [x.clone() for x in inputs]
            actual = gate_projections.project(*inputs)
            expected = (F.linear(first, wf), F.linear(second, wg, bias))
            for a, b in zip(actual, expected):
                assert a.is_contiguous()
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            for x, snapshot in zip(inputs, snapshots):
                torch.testing.assert_close(x, snapshot, rtol=0, atol=0)
    # Real BF16 inputs may change GEMM rounding; test against FP32 arithmetic,
    # then measure full-model common-token divergence in the engine panel.
    first = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16) / 8
    second = torch.randn_like(first) / 8
    wf = torch.randn(1024, 256, device="cuda", dtype=torch.bfloat16) / 8
    wg = torch.randn(2048, 256, device="cuda", dtype=torch.bfloat16) / 8
    bias = torch.randn(2048, device="cuda", dtype=torch.bfloat16) / 8
    inputs = [first, second, wf, wg, bias]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            gate_projections.project(*inputs)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = gate_projections.project(*inputs)
    pointers = [x.data_ptr() for x in inputs]
    for scale in [1.0, 0.5, -1.0]:
        wf.mul_(scale)
        wg.mul_(scale)
        bias.add_(0.125)
        first.add_(0.015625)
        second.sub_(0.03125)
        graph.replay()
        expected = (
            F.linear(first.float(), wf.float()).bfloat16(),
            F.linear(second.float(), wg.float(), bias.float()).bfloat16(),
        )
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0.008, atol=0.004)
        assert [x.data_ptr() for x in inputs] == pointers


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_actual_gate_loaders_and_repeated_refresh():
    runtime = import_module("sglang.srt.runtime_context")
    layer = import_module("olmo_sglang.kda.layer")
    models = import_module("olmo_sglang.models.olmo3_moe")
    config = SimpleNamespace(
        hidden_size=1024,
        linear_num_key_heads=8,
        linear_num_value_heads=8,
        linear_key_head_dim=128,
        linear_value_head_dim=256,
        linear_conv_kernel_dim=4,
        linear_norm_eps=1e-6,
        n_routed_experts=0,
    )
    with (
        runtime.get_context().override_server_args(),
        runtime.get_parallel().override(
            tp_size=1, tp_rank=0, attn_tp_size=1, attn_tp_rank=0
        ),
        torch.device("cuda"),
    ):
        attention = layer.Olmo3MoeKDAAttention(
            config, layer_id=0, quant_config=None, prefix=""
        ).bfloat16()
    model = models.Olmo3MoeForCausalLM.__new__(models.Olmo3MoeForCausalLM)
    nn.Module.__init__(model)
    model.config = config
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module()])
    model.model.layers[0].self_attn = attention
    tensors = [
        attention.f_proj_2.weight,
        attention.g_proj_2.weight,
        attention.g_proj_2.bias,
    ]
    names = list(model.state_dict())
    pointers = [x.data_ptr() for x in tensors]
    first = torch.ones((64, 256), device="cuda", dtype=torch.bfloat16)
    second = torch.full_like(first, -1)
    keys = [
        "model.layers.0.linear_attn.f_proj_2.weight",
        "model.layers.0.linear_attn.g_proj_2.weight",
        "model.layers.0.linear_attn.g_proj_2.bias",
    ]
    updates = [
        (key, torch.full_like(value, 0.125)) for key, value in zip(keys, tensors)
    ]
    assert len(model.load_weights(updates)) == 3
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        gate_projections.project(first, second, *tensors)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = gate_projections.project(first, second, *tensors)
    graph.replay()
    previous = [x.clone() for x in actual]
    for scale in [2.0, -1.0]:
        assert (
            len(model.load_weights([(name, value * scale) for name, value in updates]))
            == 3
        )
        graph.replay()
        expected = (attention.f_proj_2(first)[0], attention.g_proj_2(second)[0])
        for a, b, old in zip(actual, expected, previous):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
            assert not torch.equal(a, old)
        previous = [x.clone() for x in actual]
        assert [x.data_ptr() for x in tensors] == pointers
        assert list(model.state_dict()) == names
