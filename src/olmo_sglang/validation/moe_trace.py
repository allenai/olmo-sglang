"""Observe native MoE operators without replacing their arithmetic."""

from contextlib import ExitStack, contextmanager
from unittest import mock

import torch

from olmo_sglang.validation.layerwise import tensor_error, tensor_value


@contextmanager
def trace_moe(module, *, backend, routed_module, shared_module):
    values = {}
    gmms = []
    handles = []

    def attach(child, label, router=False):
        def before(_module, args):
            values[label + ".input"] = tensor_value(args[0])

        def after(_module, _args, output):
            if router:
                values["router.weights"] = tensor_value(output[0])
                values["router.ids"] = tensor_value(output[1])
            else:
                values[label + ".output"] = tensor_value(output)

        handles.extend(
            (
                child.register_forward_pre_hook(before),
                child.register_forward_hook(after),
            )
        )

    with ExitStack() as stack:
        for label, name in (
            ("router", "router" if backend == "hf" else "routed_experts_router"),
            ("latent_down", "latent_down_proj"),
            ("latent_up", "latent_up_proj"),
            ("shared", "shared_expert" if backend == "hf" else "shared_experts"),
        ):
            child = getattr(module, name, None)
            if child is None:
                raise ValueError(f"Required latent/shared MoE component absent: {name}")
            attach(child, label, router=label == "router")
        original_gmm = routed_module.gmm

        def gmm(a, b, *args, **kwargs):
            output = original_gmm(a, b, *args, **kwargs)
            gmms.append(
                {
                    "input": tensor_value(a),
                    "output": tensor_value(output),
                    "weight_shape": list(b.shape),
                    "weight_stride": list(b.stride()),
                }
            )
            return output

        stack.enter_context(mock.patch.object(routed_module, "gmm", side_effect=gmm))
        if backend == "core":
            original_activation = module.routed_experts.chunk_and_activate

            def activation(x, **kwargs):
                output = original_activation(x, **kwargs)
                values["activation.input"] = tensor_value(x)
                values["activation.output"] = tensor_value(output)
                return output

            stack.enter_context(
                mock.patch.object(
                    module.routed_experts, "chunk_and_activate", side_effect=activation
                )
            )
            original_swiglu = shared_module._swiglu

            def shared(up, gate):
                output = original_swiglu(up, gate)
                values["shared.up"] = tensor_value(up)
                values["shared.gate"] = tensor_value(gate)
                values["shared.activated"] = tensor_value(output)
                return output

            stack.enter_context(
                mock.patch.object(shared_module, "_swiglu", side_effect=shared)
            )
        try:
            yield values, gmms
        finally:
            for handle in handles:
                handle.remove()


def compare_branches(core, hf):
    result = {}
    for key in (
        "router.input",
        "router.weights",
        "latent_down.output",
        "latent_up.input",
        "latent_up.output",
        "shared.input",
        "shared.output",
        "combined",
    ):
        a, b = core[key], hf[key]
        if key == "shared.output":
            a = a.squeeze(0)
        result[key] = tensor_error(a, b)
    a, b = core["router.ids"], hf["router.ids"]
    result["router.ids"] = {
        "shape": list(a.shape),
        "exact": torch.equal(a, b),
        "different_slots": int((a != b).sum()),
    }
    return result


def compare_gmms(core, hf):
    if len(core) != 2 or len(hf) != 2:
        raise ValueError(
            "Controlled reference requires exactly two routed grouped GEMMs"
        )
    return {
        stage: {kind: tensor_error(a[kind], b[kind]) for kind in ("input", "output")}
        for stage, a, b in zip(("up_gate", "down"), core, hf, strict=True)
    }
