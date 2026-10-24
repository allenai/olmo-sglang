"""Interactions between independently qualified decode optimizations."""

import os
from importlib import import_module
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch.nn import functional as F

from olmo_sglang import core_compat


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_norm_combines_pre_add_scale_residual_and_fp32_output(dtype):
    torch.manual_seed(1031)
    norm = core_compat.CoreRMSNorm(128, 1e-6).to(dtype=dtype)
    value = torch.randn(7, 128).to(dtype)
    shared = torch.randn_like(value)
    residual = torch.randn_like(value)
    with torch.no_grad():
        expected = residual + norm((value + shared) * 1.7)
        actual, auxiliary = norm(
            value, pre_add=shared, scale=1.7, residual=residual, return_fp32=True
        )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(auxiliary, expected.float(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_composed_norm_graph_keeps_both_rounding_boundaries():
    kernels = import_module("olmo_sglang.rounding_kernels")
    torch.manual_seed(1032)
    for width in [64, 1024, 8192]:
        value = torch.randn(64, width, device="cuda", dtype=torch.bfloat16)
        shared = torch.randn_like(value)
        residual = torch.randn_like(value)
        norm = core_compat.CoreRMSNorm(width, 1e-6).cuda().bfloat16()

        def forward():
            return kernels.rms_norm(
                value,
                norm.weight,
                1e-6,
                pre_add=shared,
                residual=residual,
                scale=1.7,
                return_fp32=True,
            )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            forward()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual, auxiliary = forward()
        for _ in range(2):
            value.normal_()
            shared.normal_()
            residual.normal_()
            norm.weight.uniform_(0.5, 1.5)
            graph.replay()
            expected = residual + norm((value + shared) * 1.7)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(auxiliary, expected.float(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime required")
@torch.inference_mode()
def test_kda_both_projection_stages_refresh_one_captured_graph():
    runtime = import_module("sglang.srt.runtime_context")
    layers = import_module("olmo_sglang.kda.layer")
    config = SimpleNamespace(
        hidden_size=256,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        linear_norm_eps=1e-6,
    )
    with (
        mock.patch.dict(
            os.environ,
            {
                "OLMO_SGLANG_KDA_PROJECTIONS": "packed",
                "OLMO_SGLANG_KDA_GATE_PROJECTIONS": "grouped",
            },
        ),
        runtime.get_context().override_server_args(),
        runtime.get_parallel().override(
            tp_size=1,
            tp_rank=0,
            attn_tp_size=1,
            attn_tp_rank=0,
        ),
        torch.device("cuda"),
    ):
        layer = layers.Olmo3MoeKDAAttention(
            config, layer_id=0, quant_config=None, prefix=""
        ).bfloat16()
    names = ["qkv_proj", "f_proj_1", "beta_proj", "g_proj_1", "f_proj_2", "g_proj_2"]
    modules = [getattr(layer, name) for name in names]
    for module in modules:
        module.weight.normal_(std=0.02)
    layer.g_proj_2.bias.normal_(std=0.02)
    pointers = [module.weight.data_ptr() for module in modules]
    state_keys = list(layer.state_dict())
    hidden = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
    original = hidden.clone()

    # Every switch combination must consume the original weights. The grouped
    # GEMM can round differently, so this is a tolerance check, not a parity claim.
    for packed in [False, True]:
        for grouped in [False, True]:
            layer._pack_kda_projections = packed
            layer._group_kda_gates = grouped
            actual = layer._project_decode(hidden)
            first = F.linear(hidden, layer.f_proj_1.weight)
            second = F.linear(hidden, layer.g_proj_1.weight)
            expected = (
                F.linear(hidden, layer.qkv_proj.weight),
                F.linear(first, layer.f_proj_2.weight),
                F.linear(hidden, layer.beta_proj.weight),
                F.linear(second, layer.g_proj_2.weight, layer.g_proj_2.bias),
            )
            for output, reference in zip(actual, expected, strict=True):
                torch.testing.assert_close(output, reference, rtol=0.008, atol=0.004)
    layer._pack_kda_projections = layer._group_kda_gates = True
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        layer._project_decode(hidden)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = layer._project_decode(hidden)
    before = [output.clone() for output in outputs]
    torch.testing.assert_close(hidden, original, rtol=0, atol=0)
    for _ in range(2):
        hidden.normal_()
        for module in modules:
            module.weight.normal_(std=0.02)
        layer.g_proj_2.bias.normal_(std=0.02)
        graph.replay()
        expected = layer._project_decode(hidden)
        for old, output, reference in zip(before, outputs, expected, strict=True):
            torch.testing.assert_close(output, reference, rtol=0, atol=0)
            assert not torch.equal(old, output)
    assert pointers == [module.weight.data_ptr() for module in modules]
    assert state_keys == list(layer.state_dict())
