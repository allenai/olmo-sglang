"""Materialize latent expert input while applying the shared expert activation."""

import torch
import triton
from triton import language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _split_latent_activate(
    X,
    LATENT,
    SHARED,
    INPUT_STRIDE: tl.constexpr,
    LATENT_WIDTH: tl.constexpr,
    SHARED_WIDTH: tl.constexpr,
    TOTAL,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    width = LATENT_WIDTH + SHARED_WIDTH
    row, col = offset // width, offset % width
    valid = offset < TOTAL
    is_latent = col < LATENT_WIDTH
    latent = tl.load(X + row * INPUT_STRIDE + col, valid & is_latent, 0)
    tl.store(LATENT + row * LATENT_WIDTH + col, latent, valid & is_latent)
    gate_col = col - LATENT_WIDTH
    gate = tl.load(
        X + row * INPUT_STRIDE + LATENT_WIDTH + gate_col,
        valid & ~is_latent,
        0,
    ).to(tl.float32)
    up = tl.load(
        X + row * INPUT_STRIDE + LATENT_WIDTH + SHARED_WIDTH + gate_col,
        valid & ~is_latent,
        0,
    ).to(tl.float32)
    # Same separate SiLU/multiply BF16 boundary as native_silu_and_mul.
    silu = tl.div_rn(gate, 1.0 + libdevice.exp(-gate))
    rounded = silu.to(tl.bfloat16).to(tl.float32)
    tl.store(
        SHARED + row * SHARED_WIDTH + gate_col,
        (rounded * up).to(tl.bfloat16),
        valid & ~is_latent,
    )


def split_latent_and_activate(value, latent_width):
    rows = value.shape[0]
    shared_width = (value.shape[1] - latent_width) // 2
    latent = torch.empty((rows, latent_width), device=value.device, dtype=value.dtype)
    shared = torch.empty((rows, shared_width), device=value.device, dtype=value.dtype)
    total = rows * (latent_width + shared_width)
    if total:
        _split_latent_activate[(triton.cdiv(total, 256),)](
            value,
            latent,
            shared,
            value.stride(0),
            latent_width,
            shared_width,
            total,
            256,
            enable_fp_fusion=False,
        )
    return latent, shared
