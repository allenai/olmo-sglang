"""Missing packed slices must fail even if every destination was touched."""

import pytest

from olmo_sglang.vision.weights import validate_language_coverage
from olmo_sglang.weight_loading import STACKED_PARAMS_MAPPING, WeightTargets


def test_missing_expert_and_projection_are_detected():
    params = {
        "model.layers.0.self_attn.qkv_proj.weight": None,
        "model.layers.0.mlp.experts.w13_weight": None,
        "model.layers.0.mlp.experts.w2_weight": None,
        "model.norm.weight": None,
    }
    resolver = WeightTargets(params, 4)
    names = [f"model.layers.0.self_attn.{p}_proj.weight" for p in ("q", "k", "v")]
    names += [
        f"model.layers.0.mlp.experts.{e}.{p}_proj.weight"
        for e in range(4)
        for p in ("gate", "up", "down")
    ]
    names += ["model.norm.weight"]
    targets = {resolver.resolve(n) for n in names}
    validate_language_coverage(params, targets, STACKED_PARAMS_MAPPING, 4)
    for name in names:
        with pytest.raises(ValueError, match="Incomplete"):
            validate_language_coverage(
                params, targets - {resolver.resolve(name)}, STACKED_PARAMS_MAPPING, 4
            )


def test_whole_parameter_and_partial_writes_cannot_overlap():
    name = "model.layers.0.self_attn.qkv_proj.weight"
    whole = ("direct", name, None, None)
    validate_language_coverage({name}, {whole}, STACKED_PARAMS_MAPPING, 4)
    with pytest.raises(ValueError, match="Overlapping"):
        validate_language_coverage(
            {name}, {whole, ("stacked", name, "q", None)}, STACKED_PARAMS_MAPPING, 4
        )


def test_complete_fused_expert_tensor_is_supported():
    name = "model.layers.0.mlp.experts.w13_weight"
    resolver = WeightTargets({name: None}, 4, fused_layers={name: None})
    target = resolver.resolve("model.layers.0.mlp.experts.gate_up_proj.weight")
    validate_language_coverage({name}, {target}, STACKED_PARAMS_MAPPING, 4)
