from types import SimpleNamespace

import pytest
import torch
from torch import nn

from olmo_sglang.validation.moe_trace import compare_branches, compare_gmms, trace_moe


def test_grouped_stages_compare_inputs_before_outputs():
    reference = [
        {"input": torch.ones(2, 4), "output": torch.ones(2, 8)},
        {"input": torch.ones(2, 4), "output": torch.ones(2, 4)},
    ]
    actual = [
        {key: value.clone() for key, value in stage.items()} for stage in reference
    ]
    actual[1]["input"][0, 0] += 0.125
    result = compare_gmms(actual, reference)
    assert result["up_gate"]["output"]["max_abs"] == 0
    assert result["down"]["input"]["max_abs"] == 0.125
    with pytest.raises(ValueError, match="exactly two"):
        compare_gmms(actual[:1], reference)


def test_routes_compare_integer_ids_exactly():
    hf = {
        name: torch.ones(1, 3, 4)
        for name in (
            "router.input",
            "router.weights",
            "latent_down.output",
            "latent_up.input",
            "latent_up.output",
            "shared.input",
            "shared.output",
            "combined",
        )
    }
    hf["router.ids"] = torch.tensor([[2**30, 2**30 + 1]])
    core = {name: value.clone() for name, value in hf.items()}
    core["shared.output"] = core["shared.output"].unsqueeze(0)
    core["router.ids"][0, 1] += 1
    result = compare_branches(core, hf)
    assert result["router.ids"]["different_slots"] == 1
    assert not result["router.ids"]["exact"]
    assert result["shared.output"]["max_abs"] == 0


class Router(nn.Module):
    def forward(self, x):
        return x, torch.zeros_like(x, dtype=torch.long)


def test_trace_does_not_change_operators_and_cleans_up_on_failure():
    module = nn.Module()
    module.router = Router()
    module.latent_down_proj = nn.Identity()
    module.latent_up_proj = nn.Identity()
    module.shared_expert = nn.Identity()

    def original(a, b):
        return a @ b

    routed = SimpleNamespace(gmm=original)
    with pytest.raises(RuntimeError, match="intentional"):
        with trace_moe(
            module, backend="hf", routed_module=routed, shared_module=None
        ) as (values, gmms):
            x = torch.randn(2, 3)
            b = torch.randn(3, 4)
            output = routed.gmm(x, b)
            assert torch.equal(output, original(x, b))
            module.router(x)
            assert torch.equal(values["router.weights"], x)
            assert torch.equal(gmms[0]["output"], output)
            raise RuntimeError("intentional")
    assert routed.gmm is original
    assert all(
        not child._forward_hooks and not child._forward_pre_hooks
        for child in module.modules()
    )
