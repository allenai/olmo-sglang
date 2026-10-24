"""Packed MoE storage, output layout, checkpoint refresh and graph contracts."""

import os
from importlib import import_module
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from olmo_sglang import core_compat, moe_projections


def test_mode_and_live_parameter_storage(monkeypatch):
    monkeypatch.delenv(moe_projections.OPTION, raising=False)
    assert not moe_projections.enabled()
    monkeypatch.setenv(moe_projections.OPTION, " packed ")
    assert moe_projections.enabled()
    monkeypatch.setenv(moe_projections.OPTION, "invalid")
    with pytest.raises(ValueError, match=moe_projections.OPTION):
        moe_projections.enabled()
    modules = nn.ModuleList([nn.Linear(8, 3, bias=False), nn.Linear(8, 10, bias=False)])
    objects = list(modules.parameters())
    names = list(modules.state_dict())
    marker = object()
    objects[0].weight_loader = marker
    packed = moe_projections.pack_weights(*modules)
    assert list(modules.parameters()) == objects
    assert list(modules.state_dict()) == names
    assert objects[0].weight_loader is marker
    assert all(
        p.is_contiguous() and p.untyped_storage().data_ptr() == packed.data_ptr()
        for p in objects
    )
    modules.load_state_dict(
        {"0.weight": torch.full((3, 8), 2.0), "1.weight": torch.full((10, 8), 3.0)}
    )
    output = F.linear(torch.ones(2, 8), packed)
    torch.testing.assert_close(output[:, :3], torch.full((2, 3), 16.0), rtol=0, atol=0)
    torch.testing.assert_close(output[:, 3:], torch.full((2, 10), 24.0), rtol=0, atol=0)


def test_cpu_output_layout_and_bad_shapes():
    value = torch.randn(5, 21)[:, :17]
    latent, activated = moe_projections.split_latent_and_activate(value, 3)
    gate, up = value[:, 3:].chunk(2, dim=-1)
    assert latent.is_contiguous() and activated.is_contiguous()
    torch.testing.assert_close(latent, value[:, :3], rtol=0, atol=0)
    torch.testing.assert_close(activated, F.silu(gate) * up, rtol=0, atol=0)
    with pytest.raises(ValueError, match="projection output"):
        moe_projections.split_latent_and_activate(value, 2)
    with pytest.raises(ValueError, match="bias-free"):
        moe_projections.pack_weights(nn.Linear(8, 3), nn.Linear(8, 10, bias=False))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_split_activation_exact_for_strides_empty_and_all_bf16_values():
    torch.manual_seed(604)
    for rows in [0, 1, 16, 64]:
        for latent_width, shared_width in [(7, 17), (512, 1024)]:
            width = latent_width + 2 * shared_width
            value = torch.randn(rows, width + 9, device="cuda", dtype=torch.bfloat16)[
                :, :width
            ]
            latent, activated = moe_projections.split_latent_and_activate(
                value, latent_width
            )
            gate, up = value[:, latent_width:].chunk(2, dim=-1)
            assert latent.is_contiguous() and activated.is_contiguous()
            torch.testing.assert_close(latent, value[:, :latent_width], rtol=0, atol=0)
            torch.testing.assert_close(activated, F.silu(gate) * up, rtol=0, atol=0)
    codes = torch.arange(65536, dtype=torch.int32, device="cuda").to(torch.int16)
    all_values = codes.view(torch.bfloat16)
    gate = all_values[torch.isfinite(all_values)].unsqueeze(0)
    latent_source = torch.arange(7, device="cuda", dtype=torch.bfloat16).unsqueeze(0)
    for factor in [0.5, 1.25, -4.0]:
        up = torch.full_like(gate, factor)
        actual_latent, actual = moe_projections.split_latent_and_activate(
            torch.cat([latent_source, gate, up], dim=1), 7
        )
        torch.testing.assert_close(actual_latent, latent_source, rtol=0, atol=0)
        torch.testing.assert_close(actual, F.silu(gate) * up, rtol=0, atol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="SGLang CUDA runtime required"
)
def test_sparse_constructor_keeps_packing_in_default_rounding_mode():
    runtime = import_module("sglang.srt.runtime_context")
    models = import_module("olmo_sglang.models.olmo3_moe")
    config = SimpleNamespace(
        **core_compat._ROUNDING_PROFILE,
        dense_layers_indices=[0],
        layer_types=(["linear_attention"] * 7 + ["full_attention"]) * 2,
    )
    args = SimpleNamespace(
        dtype="bfloat16",
        cuda_graph_backend_decode="full",
        cuda_graph_backend_prefill="disabled",
        moe_runner_backend="triton",
        speculative_algorithm=None,
        enable_torch_compile=False,
    )
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        # Only the expert allocation/logging are stubbed: the real constructor,
        # auto selector, native projections and packed storage run unchanged.
        with (
            mock.patch.dict(
                os.environ,
                {
                    moe_projections.OPTION: "packed",
                    core_compat.ENVIRONMENT_VARIABLE: "auto",
                },
            ),
            mock.patch.object(
                models, "RoundingExperts", side_effect=lambda **kwargs: nn.Identity()
            ) as experts,
            mock.patch.object(models, "_log_ep_parallelism", return_value=(0, 512)),
            runtime.get_context().override_server_args(),
            runtime.get_parallel().override(
                tp_size=1, tp_rank=0, attn_tp_size=1, attn_tp_rank=0
            ),
            torch.device("cuda"),
        ):
            with core_compat.model_mode(
                config, runtime.get_parallel(), args, None, dtype=torch.bfloat16
            ) as selected:
                assert selected == "rounding"
                module = models.Olmo3MoeSparseMLP(
                    config, layer_id=1, quant_config=None, prefix=""
                )
                experts.assert_called_once()
                assert module._pack_moe_projections
            assert core_compat.mode() == "off"
            assert module._pack_moe_projections  # retained after auto context exits
            assert module._packed_projection_weight.shape == (2560, 1024)
            for weight in [
                module.latent_down_proj.weight,
                module.shared_expert.gate_up_proj.weight,
            ]:
                assert weight.is_contiguous()
                assert (
                    weight.untyped_storage().data_ptr()
                    == module._packed_projection_weight.data_ptr()
                )
    finally:
        torch.set_default_dtype(original_dtype)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="SGLang CUDA runtime required"
)
def test_actual_loaders_refresh_packed_moe_graph():
    runtime = import_module("sglang.srt.runtime_context")
    linear = import_module("sglang.srt.layers.linear")
    models = import_module("olmo_sglang.models.olmo3_moe")
    module = models.Olmo3MoeSparseMLP.__new__(models.Olmo3MoeSparseMLP)
    nn.Module.__init__(module)
    with (
        runtime.get_context().override_server_args(),
        runtime.get_parallel().override(
            tp_size=1, tp_rank=0, attn_tp_size=1, attn_tp_rank=0
        ),
        torch.device("cuda"),
    ):
        module.latent_down_proj = linear.ReplicatedLinear(
            256, 64, bias=False, quant_config=None, prefix=""
        )
        module.shared_expert = models.Olmo3MoeDenseMLP(
            256, 64, quant_config=None, prefix=""
        )
    module._pack_moe_projections = True
    module.register_buffer("_packed_projection_weight", None, persistent=False)
    module._repack_projection_weights()
    module = module.bfloat16()
    model = models.Olmo3MoeForCausalLM.__new__(models.Olmo3MoeForCausalLM)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(n_routed_experts=0)
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module()])
    model.model.layers[0].mlp = module
    names = list(model.state_dict())
    assert not any("_packed" in name for name in names)
    parameters = [
        module.latent_down_proj.weight,
        module.shared_expert.gate_up_proj.weight,
    ]
    pointers = [p.data_ptr() for p in parameters]
    with torch.no_grad():
        for weight in parameters:
            weight.normal_(std=0.02)
    hidden = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        module._project_inputs(hidden)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = module._project_inputs(hidden)
    graph.replay()
    first = [value.clone() for value in actual]
    updates = [
        (
            "model.layers.0.mlp.latent_down_proj.weight",
            torch.full((64, 256), 0.03, device="cuda"),
        ),
        (
            "model.layers.0.mlp.shared_expert.gate_proj.weight",
            torch.full((64, 256), 0.04, device="cuda"),
        ),
        (
            "model.layers.0.mlp.shared_expert.up_proj.weight",
            torch.full((64, 256), 0.05, device="cuda"),
        ),
    ]
    for multiplier, scale in [(1.0, 1.0), (2.0, 0.75)]:
        # The second call also exercises the cached checkpoint-name resolver.
        assert (
            len(
                model.load_weights(
                    [(name, value * multiplier) for name, value in updates]
                )
            )
            == 2
        )
        hidden.mul_(scale)
        original_hidden = hidden.clone()
        graph.replay()
        projected = F.linear(hidden, torch.cat(parameters))
        latent = projected[:, :64].contiguous()
        gate, up = projected[:, 64:].chunk(2, dim=-1)
        for old, output, expected in zip(
            first, actual, [latent, F.silu(gate) * up], strict=True
        ):
            assert output.is_contiguous()
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            assert not torch.equal(old, output)
        torch.testing.assert_close(hidden, original_hidden, rtol=0, atol=0)
    assert pointers == [p.data_ptr() for p in parameters]
    assert list(model.state_dict()) == names
