"""Actual SGLang checkpoint loaders and graph replay on shared packed storage."""

from importlib import import_module
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from olmo_sglang.kda import projections


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="SGLang CUDA runtime required"
)
def test_packed_kda_checkpoint_refresh_and_graph_replay(monkeypatch):
    runtime = import_module("sglang.srt.runtime_context")
    layer = import_module("olmo_sglang.kda.layer")
    models = import_module("olmo_sglang.models.olmo3_moe")
    monkeypatch.setenv(projections.PROJECTION_VARIABLE, "packed")
    config = SimpleNamespace(
        hidden_size=256,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
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
        )
    # Exercise _apply: dtype conversion must restore views into the moved buffer.
    attention = attention.bfloat16()
    model = models.Olmo3MoeForCausalLM.__new__(models.Olmo3MoeForCausalLM)
    nn.Module.__init__(model)
    model.config = config
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module()])
    model.model.layers[0].self_attn = attention
    modules = [getattr(attention, name) for name in projections.PROJECTION_NAMES]
    for module in modules:
        assert (
            module.weight.untyped_storage().data_ptr()
            == attention._packed_input_weight.data_ptr()
        )
    names = list(model.state_dict())
    assert not any("_packed" in name for name in names)
    with torch.no_grad():
        for module in modules:
            module.weight.normal_(std=0.02)
    pointers = [module.weight.data_ptr() for module in modules]
    hidden = torch.randn(64, 256, dtype=torch.bfloat16, device="cuda")

    def project():
        return attention._project_inputs(hidden)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        project()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = project()
    before = [value.clone() for value in actual]
    updates = [
        (
            "model.layers.0.linear_attn.q_proj.weight",
            torch.full((256, 256), 0.03, device="cuda"),
        ),
        (
            "model.layers.0.linear_attn.f_proj_1.weight",
            torch.full((64, 256), 0.04, device="cuda"),
        ),
        (
            "model.layers.0.linear_attn.beta_proj.weight",
            torch.full((4, 256), 0.05, device="cuda"),
        ),
        (
            "model.layers.0.linear_attn.g_proj_1.weight",
            torch.full((64, 256), 0.06, device="cuda"),
        ),
    ]
    assert len(model.load_weights(updates)) == 4
    graph.replay()
    expected_weight = torch.cat([module.weight for module in modules])
    expected = projections.project(
        hidden, expected_weight, attention._packed_input_sizes
    )
    for first, output, reference in zip(before, actual, expected, strict=True):
        torch.testing.assert_close(output, reference, atol=0, rtol=0)
        assert not torch.equal(first, output)
    assert pointers == [module.weight.data_ptr() for module in modules]
    assert list(model.state_dict()) == names
    # Repeated loader calls must update the same storage used by captured graphs.
    assert len(model.load_weights([(name, value * 2) for name, value in updates])) == 4
    graph.replay()
    expected = projections.project(
        hidden, torch.cat([m.weight for m in modules]), attention._packed_input_sizes
    )
    for output, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(output, reference, atol=0, rtol=0)
