"""Optional packed first-stage KDA projections, retaining checkpoint parameter names."""

import os

import torch
from torch.nn import functional as F

PROJECTION_VARIABLE = "OLMO_SGLANG_KDA_PROJECTIONS"
PROJECTION_NAMES = ("qkv_proj", "f_proj_1", "beta_proj", "g_proj_1")


def projection_mode():
    value = os.environ.get(PROJECTION_VARIABLE, "torch").strip().lower()
    if value not in {"torch", "packed"}:
        raise ValueError(f"Invalid {PROJECTION_VARIABLE}: {value!r}")
    return value


@torch.no_grad()
def pack_weights(modules):
    """Rebind existing Parameter storage, preserving names, objects and loaders.

    The returned tensor owns storage; modules retain contiguous row views into it.
    In-place checkpoint and online weight copies therefore update the packed GEMM.
    Call again after device/dtype moves, before CUDA graph capture.
    """
    weights = [module.weight for module in modules]
    if not weights or any(
        weight.ndim != 2
        or not weight.is_contiguous()
        or weight.shape[1] != weights[0].shape[1]
        or weight.dtype != weights[0].dtype
        or weight.device != weights[0].device
        or getattr(module, "bias", None) is not None
        for module, weight in zip(modules, weights, strict=True)
    ):
        raise ValueError(
            "packed projections require compatible bias-free matrix weights"
        )
    packed = torch.cat(weights, dim=0)
    offset = 0
    for weight in weights:
        rows = weight.shape[0]
        weight.data = packed[offset : offset + rows]
        offset += rows
    return packed


def project(hidden_states, packed_weight, output_sizes):
    """Keep the first-stage BF16 materialization; slices may have a padded stride."""
    return F.linear(hidden_states, packed_weight).split(output_sizes, dim=-1)
