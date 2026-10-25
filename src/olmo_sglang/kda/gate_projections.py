"""Optional grouped second-stage KDA gate GEMMs, with original parameters."""

import os
from importlib import import_module

import torch
from torch.nn import functional as F

VARIABLE = "OLMO_SGLANG_KDA_GATE_PROJECTIONS"


def projection_mode():
    mode = os.environ.get(VARIABLE, "torch").strip().lower()
    if mode not in {"torch", "grouped"}:
        raise ValueError(f"{VARIABLE} must be 'torch' or 'grouped', got {mode!r}")
    return mode


def supported(first, second, first_weight, second_weight, second_bias):
    tensors = (first, second, first_weight, second_weight, second_bias)
    return (
        all(t.is_cuda and t.dtype == torch.bfloat16 for t in tensors)
        and all(t.device == first.device for t in tensors)
        and first.ndim == second.ndim == first_weight.ndim == second_weight.ndim == 2
        and second_bias.ndim == 1
        and first.shape == second.shape
        and first.shape[1] == first_weight.shape[1] == second_weight.shape[1]
        and second_bias.shape[0] == second_weight.shape[0]
        and first.stride(1) == second.stride(1) == 1
        and first_weight.is_contiguous()
        and second_weight.is_contiguous()
        and second_bias.is_contiguous()
    )


def project(first, second, first_weight, second_weight, second_bias):
    """Keep both BF16 first-stage inputs; no zero-filled block-diagonal work."""
    if supported(first, second, first_weight, second_weight, second_bias):
        kernels = import_module("olmo_sglang.kda.gate_projection_kernels")
        return kernels.project(first, second, first_weight, second_weight, second_bias)
    return F.linear(first, first_weight), F.linear(second, second_weight, second_bias)
