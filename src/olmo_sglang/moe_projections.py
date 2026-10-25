"""Optional latent/shared projection storage and contiguous output materialization."""

import os
from importlib import import_module

import torch
from torch.nn import functional as F

OPTION = "OLMO_SGLANG_MOE_PROJECTIONS"


def enabled():
    value = os.environ.get(OPTION, "torch").strip().lower()
    if value not in {"torch", "packed"}:
        raise ValueError(f"Invalid {OPTION}: {value!r}")
    return value == "packed"


@torch.no_grad()
def pack_weights(latent, shared):
    """Keep the original Parameter objects, contiguous row views and loaders."""
    weights = [latent.weight, shared.weight]
    if any(
        weight.ndim != 2
        or not weight.is_contiguous()
        or weight.shape[1] != weights[0].shape[1]
        or weight.dtype != weights[0].dtype
        or weight.device != weights[0].device
        or getattr(module, "bias", None) is not None
        for module, weight in zip([latent, shared], weights, strict=True)
    ):
        raise ValueError("packed MoE projections require compatible bias-free weights")
    packed = torch.cat(weights, dim=0)
    split = weights[0].shape[0]
    weights[0].data = packed[:split]
    weights[1].data = packed[split:]
    return packed


def split_latent_and_activate(value, latent_width):
    """Return contiguous expert input and shared SwiGLU with BF16 SiLU rounding."""
    if (
        value.ndim != 2
        or latent_width <= 0
        or value.shape[1] <= latent_width
        or (value.shape[1] - latent_width) % 2
    ):
        raise ValueError("expected [tokens, latent + 2 * shared] projection output")
    if value.is_cuda and value.dtype == torch.bfloat16 and value.stride(-1) == 1:
        kernels = import_module("olmo_sglang.moe_projection_kernels")
        return kernels.split_latent_and_activate(value, latent_width)
    latent, gate_up = value.split([latent_width, value.shape[1] - latent_width], dim=1)
    gate, up = gate_up.chunk(2, dim=1)
    return latent.contiguous(), (F.silu(gate) * up).contiguous()
