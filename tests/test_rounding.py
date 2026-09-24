"""Numerical and replay controls for graph-compatible expert rounding."""

from importlib import import_module

import pytest
import torch
from torch.nn import functional as F

from olmo_sglang import core_compat


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tokens", [1, 4, 81])
def test_rounding_graph_replays_changed_routes_inputs_and_weights(tokens, monkeypatch):
    rounded_experts = import_module("olmo_sglang.rounding").rounded_experts
    runtime = import_module("sglang.srt.runtime_context")
    server_args = import_module("sglang.srt.server_args")
    monkeypatch.setattr(
        runtime, "_CONTEXT", runtime.RuntimeContext(parallel=runtime.ParallelContext())
    )
    # Only dataclass defaults are needed by the kernel tuner. Engine startup's
    # post-init resolves a checkpoint and hardware, neither part of this test.
    with monkeypatch.context() as setup:
        setup.setattr(server_args.ServerArgs, "__post_init__", lambda self: None)
        args = server_args.ServerArgs(model_path="unused", device="cuda")
    runtime.publish(args, role="test")
    torch.manual_seed(841)
    device = "cuda"
    x = torch.randn(tokens, 64, device=device, dtype=torch.bfloat16) * 0.1
    w13 = torch.randn(8, 128, 64, device=device, dtype=torch.bfloat16) * 0.1
    w2 = torch.randn(8, 64, 64, device=device, dtype=torch.bfloat16) * 0.1
    routes = torch.randint(0, 8, (tokens, 2), device=device, dtype=torch.int32)
    weights = torch.rand(tokens, 2, device=device)
    norm = core_compat.CoreRMSNorm(64, 1e-6).to(device=device, dtype=torch.bfloat16)

    def forward():
        return norm(rounded_experts(x, w13, w2, weights, routes))

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.no_grad(), torch.cuda.stream(stream):
        for _ in range(3):
            forward()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(graph):
        captured = forward()
    pointers = [v.data_ptr() for v in [w13, w2, norm.weight]]
    previous = None
    for _ in range(3):
        x.normal_(std=0.1)
        routes.random_(0, 8)
        weights.uniform_()
        w13.normal_(std=0.1)
        w2.normal_(std=0.1)
        with torch.no_grad():
            norm.weight.uniform_(0.5, 1.5)
            eager = forward()
            graph.replay()
        torch.testing.assert_close(captured, eager, rtol=0, atol=0)
        assert pointers == [v.data_ptr() for v in [w13, w2, norm.weight]]
        if previous is not None:
            assert not torch.equal(captured, previous)
        previous = captured.clone()
        # Independent per-route Torch GEMMs check weight ordering, unweighted
        # BF16 down outputs and FP32 combine. GEMM accumulation may differ.
        expected = torch.zeros_like(x, dtype=torch.float32)
        for token in range(tokens):
            for slot in range(2):
                expert = routes[token, slot].item()
                gate, up = F.linear(x[token], w13[expert]).chunk(2)
                down = F.linear(F.silu(gate) * up, w2[expert])
                expected[token] += down.float() * weights[token, slot]
        with torch.no_grad():
            reference = norm(expected.bfloat16())
        torch.testing.assert_close(eager, reference, rtol=0.04, atol=0.04)
