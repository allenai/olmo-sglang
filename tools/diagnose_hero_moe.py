"""One-block latent-MoE operator diagnosis on frozen HF activations; no training."""

import argparse
import copy
import hashlib
import json
import os
from importlib import import_module
from pathlib import Path
from unittest import mock

import torch
from safetensors import safe_open
from torch.nn import functional as F
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from olmo_sglang.validation.layerwise import tensor_error
from olmo_sglang.validation.moe_trace import compare_branches, compare_gmms, trace_moe


def layer_state(root, layer):
    prefix = f"model.layers.{layer}."
    result = {}
    for path in sorted(root.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as reader:
            for name in reader.keys():
                if name.startswith(prefix):
                    if name in result:
                        raise ValueError(f"Duplicate checkpoint tensor {name}")
                    result[name] = reader.get_tensor(name)
    if not result:
        raise ValueError("Requested layer is absent from the checkpoint")
    return result


def make_native(block_state, config, layer, factory, core_config, attention, convert):
    model_config = factory.build_olmo3_moe_config_from_hf_config(
        config,
        dtype=core_config.DType.bfloat16,
        attention_backend=attention.AttentionBackendName.torch,
        router_aux_loss_weight=0.0,
        router_z_loss_weight=0.0,
    )
    native = (
        model_config.resolved_block_configs[layer]
        .build(
            d_model=config.hidden_size,
            block_idx=layer,
            n_layers=config.num_hidden_layers,
            init_device="meta",
        )
        .to_empty(device="cuda")
    )
    single = copy.deepcopy(config)
    single.num_hidden_layers = 1
    single.layer_types = [config.layer_types[layer]]
    single.dense_layers_indices = []
    state = {
        name.replace(f"model.layers.{layer}.", "model.layers.0.", 1): value
        for name, value in block_state.items()
    }
    # The canonical converter includes global weights; metadata-only placeholders
    # let it transform exactly one real layer. No placeholders enter the block.
    for name in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
        state[name] = torch.empty(0, device="meta")
    if getattr(config, "embed_norm", False):
        state["model.embed_norm.weight"] = torch.empty(0, device="meta")
    converted = convert.convert_state_from_hf(single, state, model_type="olmo3moe")
    selected = {
        name.removeprefix("blocks.0."): value
        for name, value in converted.items()
        if name.startswith("blocks.0.")
    }
    native.load_state_dict(selected, strict=True)
    return native.eval()


def observe_core(core, residual, routed, shared, *, grad_enabled):
    with torch.set_grad_enabled(grad_enabled):
        with trace_moe(
            core, backend="core", routed_module=routed, shared_module=shared
        ) as (values, gmms):

            def combined_input(_module, inputs):
                values["combined"] = inputs[0].detach().cpu().clone()

            handle = core.feed_forward_norm.register_forward_pre_hook(combined_input)
            try:
                with mock.patch.object(core, "_res_norm_attn", return_value=residual):
                    output = core(residual)
                del output
            finally:
                handle.remove()
    return values, gmms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path)
    args = parser.parse_args()
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    if (
        args.layer in (config.dense_layers_indices or [])
        or config.latent_moe_dim is None
    ):
        parser.error("Choose a routed latent-MoE layer")
    source_hashes = {
        name: hashlib.sha256((args.model / name).read_bytes()).hexdigest()
        for name in ("configuration_olmo3moe.py", "modeling_olmo3moe.py")
    }
    if args.reference_report:
        previous = json.loads(args.reference_report.read_text())
        if previous["hf_source_sha256"] != source_hashes:
            raise ValueError("Activation reference used different HF source")
    saved = torch.load(args.activations, map_location="cpu", weights_only=True)
    if not saved["hf"] or len(saved["hf"]) != len(saved["prompts"]):
        raise ValueError("Frozen activation/prompt counts differ")
    state = layer_state(args.model, args.layer)
    hf_class = get_class_from_dynamic_module(
        "modeling_olmo3moe.Olmo3MoeSparseMLP", args.model
    )
    with torch.device("meta"):
        hf = hf_class(config).to(torch.bfloat16)
    hf = hf.to_empty(device="cuda")
    prefix = f"model.layers.{args.layer}.mlp."
    hf.load_state_dict(
        {
            name.removeprefix(prefix): value
            for name, value in state.items()
            if name.startswith(prefix)
        },
        strict=True,
    )
    hf.eval()
    factory = import_module("olmo_core.nn.moe.v2.olmo3")
    core_config = import_module("olmo_core.config")
    attention = import_module("olmo_core.nn.attention")
    convert = import_module("olmo_core.nn.hf.convert")
    routed = import_module("olmo_core.nn.moe.v2.routed_experts")
    shared = import_module("olmo_core.nn.moe.v2.shared_experts")
    core = make_native(
        state, config, args.layer, factory, core_config, attention, convert
    )
    del state
    records = []
    with torch.no_grad():
        for index, captures in enumerate(saved["hf"]):
            original = captures[args.layer]
            residual = original["ffn_pre_norm.input"].cuda()
            mlp_input = original["ffn_pre_norm.output"].cuda()
            expected_shape = (1, len(saved["prompts"][index]), config.hidden_size)
            if (
                tuple(residual.shape) != expected_shape
                or mlp_input.shape != residual.shape
            ):
                raise ValueError("Frozen prompt/activation shapes differ")
            core_values, core_gmms = observe_core(
                core, residual, routed, shared, grad_enabled=False
            )
            grad_values, grad_gmms = observe_core(
                core, residual, routed, shared, grad_enabled=True
            )
            modes = []
            for controlled in (False, True):
                with mock.patch.dict(
                    os.environ, {"OLMO_HF_MOE_CORE_REFERENCE": str(int(controlled))}
                ):
                    with trace_moe(
                        hf, backend="hf", routed_module=routed, shared_module=shared
                    ) as (hf_values, hf_gmms):
                        hf_values["combined"] = hf(mlp_input).detach().cpu()
                mode = {
                    "hf_core_reference": controlled,
                    "branches": compare_branches(core_values, hf_values),
                    "grad_enabled_core_branches": compare_branches(
                        grad_values, hf_values
                    ),
                }
                if controlled:
                    mode["routed_gmms"] = compare_gmms(core_gmms, hf_gmms)
                    mode["grad_enabled_core_routed_gmms"] = compare_gmms(
                        grad_gmms, hf_gmms
                    )
                    up_gate = core_values["activation.input"].cuda()
                    up, gate = up_gate.chunk(2, dim=-1)
                    eager = up * F.silu(gate)
                    fp32 = (up.float() * gate.float() * torch.sigmoid(gate.float())).to(
                        up.dtype
                    )
                    with torch.enable_grad():
                        training_path = core.routed_experts.chunk_and_activate(
                            up_gate,
                            num_elements=torch.tensor(up_gate.shape[0], device="cuda"),
                        )
                    mode["same_input_activation"] = {
                        "native_no_grad_vs_hf_eager": tensor_error(
                            core_values["activation.output"], eager.cpu()
                        ),
                        "native_no_grad_vs_fp32_single_cast": tensor_error(
                            core_values["activation.output"], fp32.cpu()
                        ),
                        "native_grad_enabled_vs_hf_eager": tensor_error(
                            training_path.cpu(), eager.cpu()
                        ),
                        "hf_actual_down_input_vs_eager_core_input": tensor_error(
                            hf_gmms[1]["input"], eager.cpu()
                        ),
                    }
                modes.append(mode)
            records.append(
                {
                    "prompt_index": index,
                    "prompt_ids": saved["prompts"][index],
                    "modes": modes,
                }
            )
    report = {
        "diagnosis_only": True,
        "model": str(args.model),
        "layer": args.layer,
        "hf_source_sha256": source_hashes,
        "activation_sha256": hashlib.sha256(args.activations.read_bytes()).hexdigest(),
        "core_sources_sha256": {
            name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for name, module in (
                ("factory", factory),
                ("routed_experts", routed),
                ("shared_experts", shared),
            )
        },
        "records": records,
        "core_model_training_flag": False,
        "core_routed_profile_pairwise_swiglu": (
            core.routed_experts._profile_pairwise_swiglu
        ),
        "core_routed_profile_rounded_wgrad": core.routed_experts._profile_rounded_wgrad,
        "core_torch_grouped_mm": routed.use_torch_grouped_mm(),
        "grad_enabled_forward": (
            "Actual single-block MoE forward with autograd enabled, "
            "eval mode, no backward"
        ),
        "input_boundary": (
            "Frozen HF attention residual / pre-FFN norm output; "
            "attention bypassed; actual no-EP MoE math unchanged"
        ),
        "limitations": (
            "One block, no backward/optimizer; grad-enabled activation is a "
            "labeled same-input diagnostic only. No gate or tolerance changes."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "prompts": len(records),
                "diagnosis_only": True,
            }
        )
    )


if __name__ == "__main__":
    main()
