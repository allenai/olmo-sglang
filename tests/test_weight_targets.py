"""Constant-time checkpoint-name resolution matches the scanning loader it replaced."""

import pytest

from olmo_sglang.models.olmo3_moe import STACKED_PARAMS_MAPPING, WeightTargets


def _legacy_resolve(name, params, num_experts):
    """The scanning loop the adapter used before memoized resolution, as a pure function."""
    expert_params_mapping = [
        (
            "experts.w13_"
            if weight_name in ("gate_proj", "up_proj")
            else "experts.w2_",
            f"experts.{expert_id}.{weight_name}.",
            expert_id,
            shard_id,
        )
        for expert_id in range(num_experts)
        for shard_id, weight_name in (
            ("w1", "gate_proj"),
            ("w2", "down_proj"),
            ("w3", "up_proj"),
        )
    ]
    if "rotary_emb.inv_freq" in name:
        return None
    name = name.replace(".linear_attn.", ".self_attn.")
    for param_name, weight_name, shard_id in STACKED_PARAMS_MAPPING:
        if weight_name not in name or ".mlp.experts." in name:
            continue
        mapped_name = name.replace(weight_name, param_name)
        if mapped_name in params:
            return ("stacked", mapped_name, shard_id, None)
    for param_name, weight_name, expert_id, shard_id in expert_params_mapping:
        if weight_name not in name:
            continue
        mapped_name = name.replace(weight_name, param_name)
        if mapped_name in params:
            return ("expert", mapped_name, shard_id, expert_id)
    if name.endswith(".bias") and name not in params:
        return None
    if name not in params:
        raise KeyError(name)
    return ("direct", name, None, None)


def _synthetic_params(num_layers, kda_layers):
    params = {
        "model.embed_tokens.weight": None,
        "model.norm.weight": None,
        "lm_head.weight": None,
    }
    for layer in range(num_layers):
        prefix = f"model.layers.{layer}."
        attn = prefix + "self_attn."
        params[attn + "qkv_proj.weight"] = None
        params[attn + "o_proj.weight"] = None
        if layer in kda_layers:
            params[attn + "qkv_conv1d.weight"] = None
            params[attn + "A_log"] = None
        params[prefix + "mlp.router.gate.weight"] = None
        params[prefix + "mlp.experts.w13_weight"] = None
        params[prefix + "mlp.experts.w2_weight"] = None
        params[prefix + "mlp.shared_experts.gate_up_proj.weight"] = None
        params[prefix + "mlp.shared_experts.down_proj.weight"] = None
        params[prefix + "input_layernorm.weight"] = None
    return params


def _checkpoint_names(num_layers, num_experts, kda_layers):
    names = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
    for layer in range(num_layers):
        prefix = f"model.layers.{layer}."
        attn = prefix + ("linear_attn." if layer in kda_layers else "self_attn.")
        names += [
            attn + p
            for p in (
                "q_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
                "o_proj.weight",
            )
        ]
        names.append(attn + "rotary_emb.inv_freq")
        if layer in kda_layers:
            names += [
                attn + p
                for p in (
                    "q_conv1d.weight",
                    "k_conv1d.weight",
                    "v_conv1d.weight",
                    "A_log",
                )
            ]
        names.append(prefix + "mlp.router.gate.weight")
        for expert in range(num_experts):
            names += [
                prefix + f"mlp.experts.{expert}.{p}.weight"
                for p in ("gate_proj", "up_proj", "down_proj")
            ]
        names += [
            prefix + f"mlp.shared_experts.{p}.weight"
            for p in ("gate_proj", "up_proj", "down_proj")
        ]
        names.append(prefix + "input_layernorm.weight")
        names.append(
            prefix + "mlp.shared_experts.down_proj.bias"
        )  # skipped: bias absent from params
    return names


@pytest.mark.parametrize(("num_layers", "num_experts"), [(2, 4), (16, 512)])
def test_resolution_matches_legacy_scan_on_full_inventory(num_layers, num_experts):
    kda_layers = {layer for layer in range(num_layers) if layer % 2 == 0}
    params = _synthetic_params(num_layers, kda_layers)
    targets = WeightTargets(params, num_experts)
    names = _checkpoint_names(num_layers, num_experts, kda_layers)
    assert len(names) > num_layers * num_experts * 3
    for name in names:
        assert targets.resolve(name) == _legacy_resolve(name, params, num_experts), name
    # Memoized: a second pass returns the identical objects without recomputation.
    assert all(targets.resolve(name) is targets.resolve(name) for name in names)


def test_expert_names_route_to_fused_parameters():
    params = _synthetic_params(1, set())
    targets = WeightTargets(params, 8)
    assert targets.resolve("model.layers.0.mlp.experts.7.gate_proj.weight") == (
        "expert",
        "model.layers.0.mlp.experts.w13_weight",
        "w1",
        7,
    )
    assert targets.resolve("model.layers.0.mlp.experts.0.up_proj.weight") == (
        "expert",
        "model.layers.0.mlp.experts.w13_weight",
        "w3",
        0,
    )
    assert targets.resolve("model.layers.0.mlp.experts.3.down_proj.weight") == (
        "expert",
        "model.layers.0.mlp.experts.w2_weight",
        "w2",
        3,
    )
    assert targets.resolve("model.layers.0.mlp.shared_experts.gate_proj.weight") == (
        "stacked",
        "model.layers.0.mlp.shared_experts.gate_up_proj.weight",
        0,
        None,
    )


def test_unknown_names_and_out_of_range_experts_raise_like_before():
    params = _synthetic_params(1, set())
    targets = WeightTargets(params, 8)
    for name in (
        "model.layers.0.mlp.experts.8.gate_proj.weight",
        "model.layers.0.nonexistent.weight",
    ):
        with pytest.raises(KeyError):
            targets.resolve(name)
        with pytest.raises(KeyError):
            _legacy_resolve(name, params, 8)
    assert targets.resolve("model.layers.0.self_attn.rotary_emb.inv_freq") is None
    assert targets.resolve("model.layers.0.mlp.experts.w2_weight.bias") is None


def test_stacked_expert_names_resolve_to_the_fused_loader_only_when_registered():
    params = _synthetic_params(1, set())
    layers = {
        "model.layers.0.mlp.experts.w13_weight": object(),
        "model.layers.0.mlp.experts.w2_weight": object(),
    }
    targets = WeightTargets(params, 8, fused_layers=layers)
    assert targets.resolve("model.layers.0.mlp.experts.gate_up_proj.weight") == (
        "fused",
        "model.layers.0.mlp.experts.w13_weight",
        "w13",
        None,
    )
    assert targets.resolve("model.layers.0.mlp.experts.down_proj.weight") == (
        "fused",
        "model.layers.0.mlp.experts.w2_weight",
        "w2",
        None,
    )
    # Per-expert names keep resolving to per-expert loads alongside the fused ones.
    assert (
        targets.resolve("model.layers.0.mlp.experts.2.down_proj.weight")[0] == "expert"
    )
    # A stacked name is unknown when no fused layer owns the parameter.
    with pytest.raises(KeyError):
        WeightTargets(params, 8).resolve("model.layers.0.mlp.experts.down_proj.weight")
