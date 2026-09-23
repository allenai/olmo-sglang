import importlib
from functools import partial

import pytest
import torch
from torch import nn

from olmo_sglang.attention import (
    PerHeadRMSNorm,
    check_head_weights_loaded,
    load_head_weight,
    local_head_range,
    scale_attention_queries,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_per_head_norm_matches_fp32_reference_and_head_identity(dtype):
    module = PerHeadRMSNorm(3, 8, 1e-6).to(dtype)
    module.weight.data.copy_(torch.linspace(0.4, 1.6, 24).reshape(3, 8))
    x = torch.linspace(-2, 3, 5 * 3 * 8).reshape(5, 3, 8).to(dtype)
    expected = torch.stack(
        [
            (
                x[:, h].float()
                * torch.rsqrt(x[:, h].float().square().mean(-1, keepdim=True) + 1e-6)
                * module.weight[h].float()
            ).to(dtype)
            for h in range(3)
        ],
        dim=1,
    )
    torch.testing.assert_close(module(x), expected, atol=0, rtol=0)
    with pytest.raises(ValueError, match="heads"):
        module(x.reshape(15, 8))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_scaling_absolute_packed_positions_and_rounding(dtype):
    q = torch.linspace(-1.3, 2.1, 5 * 3 * 8).reshape(5, 3, 8).to(dtype)
    positions = torch.tensor([0, 1, 81, 0, 8191])
    gains = torch.tensor([0.7, 1.31, -0.3], dtype=dtype)
    expected = torch.stack(
        [
            q[i] * ((positions[i] + 1).log().to(dtype) * gains).unsqueeze(-1)
            for i in range(5)
        ]
    )
    actual = scale_attention_queries(q, positions, gains)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.count_nonzero(actual[[0, 3]]) == 0
    if dtype == torch.bfloat16:
        sequential = (q * (positions + 1).log().to(dtype)[:, None, None]) * gains[
            None, :, None
        ]
        assert not torch.equal(actual, sequential)
    # A cached suffix keeps its absolute positions regardless of other batch tokens.
    torch.testing.assert_close(
        scale_attention_queries(q[2:3], positions[2:3], gains), actual[2:3]
    )


@pytest.mark.parametrize("heads,tp", [(8, 1), (8, 2), (4, 8)])
@pytest.mark.parametrize("width", [None, 3])
def test_full_and_local_weight_loads_shard_or_replicate_in_place(heads, tp, width):
    shape = (heads,) if width is None else (heads, width)
    source = torch.arange(
        torch.tensor(shape).prod().item(), dtype=torch.float32
    ).reshape(shape)
    for rank in range(tp):
        start, count = local_head_range(heads, tp, rank)
        target = nn.Parameter(torch.zeros((count, *shape[1:]), dtype=torch.bfloat16))
        pointer = target.data_ptr()
        loader = partial(load_head_weight, total_heads=heads, tp_size=tp, tp_rank=rank)
        loader(target, source)
        torch.testing.assert_close(target, source[start : start + count].bfloat16())
        loader(target, source[start : start + count] + 10)
        torch.testing.assert_close(
            target, (source[start : start + count] + 10).bfloat16()
        )
        assert target.data_ptr() == pointer
        with pytest.raises(ValueError, match="shape"):
            loader(target, source.unsqueeze(-1))


def test_replicated_kv_assignment_matches_contiguous_query_groups():
    assert [local_head_range(4, 8, rank)[0] for rank in range(8)] == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
    ]
    with pytest.raises(ValueError):
        local_head_range(3, 8, 0)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="native SGLang model import requires CUDA"
)
def test_live_partial_hf_loading_preserves_parameters_and_changes_both_features():
    Olmo3MoeForCausalLM = importlib.import_module(
        "olmo_sglang.models.olmo3_moe"
    ).Olmo3MoeForCausalLM
    # Use the actual model loader with minimal named parameters; no scheduler is needed.
    model = Olmo3MoeForCausalLM.__new__(Olmo3MoeForCausalLM)
    nn.Module.__init__(model)
    model.config = type("Config", (), {"n_routed_experts": 0})()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module()])
    attention = nn.Module()
    model.model.layers[0].self_attn = attention
    attention.q_norm = PerHeadRMSNorm(2, 4, 1e-6)
    attention.ssmax_scale = nn.Parameter(torch.ones(2))
    for parameter in (attention.q_norm.weight, attention.ssmax_scale):
        parameter.weight_loader = partial(
            load_head_weight, total_heads=2, tp_size=1, tp_rank=0
        )
    q = torch.randn(3, 2, 4)
    positions = torch.tensor([0, 8, 19])
    before = scale_attention_queries(
        attention.q_norm(q), positions, attention.ssmax_scale
    )
    names = [
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.self_attn.ssmax_scale",
    ]
    pointers = [attention.q_norm.weight.data_ptr(), attention.ssmax_scale.data_ptr()]
    assert model.load_weights([(names[0], torch.full((2, 4), 0.7))]) == {names[0]}
    assert model.load_weights([(names[1], torch.tensor([0.5, 1.5]))]) == {names[1]}
    after = scale_attention_queries(
        attention.q_norm(q), positions, attention.ssmax_scale
    )
    assert not torch.equal(before, after)
    assert pointers == [
        attention.q_norm.weight.data_ptr(),
        attention.ssmax_scale.data_ptr(),
    ]


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graph capture requires CUDA"
)
@torch.no_grad()
def test_cuda_graph_replay_reads_updated_positions_and_gains():
    module = PerHeadRMSNorm(3, 8, 1e-6).cuda().bfloat16()
    query = torch.randn(4, 3, 8, device="cuda", dtype=torch.bfloat16)
    positions = torch.tensor([0, 7, 13, 31], device="cuda")
    gains = nn.Parameter(
        torch.tensor([0.7, 1.1, 1.6], device="cuda", dtype=torch.bfloat16)
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            scale_attention_queries(module(query), positions, gains)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = scale_attention_queries(module(query), positions, gains)
    graph.replay()
    torch.testing.assert_close(
        output, scale_attention_queries(module(query), positions, gains), atol=0, rtol=0
    )
    before = output.clone()
    load_head_weight(
        module.weight,
        torch.full_like(module.weight, 0.6),
        total_heads=3,
        tp_size=1,
        tp_rank=0,
    )
    load_head_weight(
        gains,
        torch.tensor([1.7, 0.4, -0.2], device="cuda"),
        total_heads=3,
        tp_size=1,
        tp_rank=0,
    )
    positions.copy_(torch.tensor([9, 1, 81, 512], device="cuda"))
    graph.replay()
    torch.testing.assert_close(
        output, scale_attention_queries(module(query), positions, gains), atol=0, rtol=0
    )
    assert not torch.equal(before, output)


def test_missing_head_weights_fail_before_inference_but_partial_loads_can_complete():
    parameters = {
        "q_norm": nn.Parameter(torch.ones(2, 4)),
        "ssmax_scale": nn.Parameter(torch.ones(2)),
    }
    with pytest.raises(RuntimeError, match="q_norm, ssmax_scale"):
        check_head_weights_loaded(parameters)
    load_head_weight(
        parameters["q_norm"], torch.ones(2, 4), total_heads=2, tp_size=1, tp_rank=0
    )
    with pytest.raises(RuntimeError, match="ssmax_scale"):
        check_head_weights_loaded(parameters)
    load_head_weight(
        parameters["ssmax_scale"], torch.ones(2), total_heads=2, tp_size=1, tp_rank=0
    )
    check_head_weights_loaded(parameters)
