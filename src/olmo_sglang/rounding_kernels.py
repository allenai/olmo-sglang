"""Small fused operations with explicit Core-compatible precision boundaries.

The expert GEMMs and their tuning remain SGLang's. These kernels fuse only the
surrounding elementwise work; they never move weighting ahead of the BF16 down
output. FP32 reductions can differ in association from PyTorch's reductions.
"""

import torch
import triton
from triton import language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _silu_mul(X, Y, WIDTH: tl.constexpr, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = offset // WIDTH, offset % WIDTH
    gate = tl.load(X + row * (2 * WIDTH) + col, offset < TOTAL, 0).to(tl.float32)
    up = tl.load(X + row * (2 * WIDTH) + WIDTH + col, offset < TOTAL, 0).to(tl.float32)
    # Explicit round-trip keeps SiLU's BF16 rounding inside the fused kernel.
    silu = tl.div_rn(gate, 1.0 + libdevice.exp(-gate))
    rounded = silu.to(tl.bfloat16).to(tl.float32)
    tl.store(Y + offset, (rounded * up).to(tl.bfloat16), offset < TOTAL)


def silu_mul(value):
    width = value.shape[-1] // 2
    output = torch.empty(
        (value.shape[0], width), device=value.device, dtype=value.dtype
    )
    _silu_mul[(triton.cdiv(output.numel(), 256),)](
        value, output, width, output.numel(), 256, enable_fp_fusion=False
    )
    return output


@triton.jit
def _weighted_sum(
    X,
    W,
    Y,
    WIDTH: tl.constexpr,
    TOPK: tl.constexpr,
    WEIGHT_STRIDE: tl.constexpr,
    WEIGHT_SLOT_STRIDE: tl.constexpr,
    ROUTES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    routes = tl.arange(0, ROUTES)
    values = tl.load(
        X + (token * TOPK + routes[:, None]) * WIDTH + columns[None, :],
        (routes[:, None] < TOPK) & (columns[None, :] < WIDTH),
        0,
    ).to(tl.float32)
    weights = tl.load(
        W + token * WEIGHT_STRIDE + routes * WEIGHT_SLOT_STRIDE, routes < TOPK, 0
    ).to(tl.float32)
    # Input is already BF16. Products and reduction stay FP32, without an FMA
    # changing the intermediate product's FP32 rounding.
    result = tl.sum(values * weights[:, None], axis=0)
    tl.store(Y + token * WIDTH + columns, result.to(tl.bfloat16), columns < WIDTH)


def weighted_sum(value, weights):
    tokens, topk, width = value.shape
    output = torch.empty((tokens, width), device=value.device, dtype=value.dtype)
    _weighted_sum[(tokens, triton.cdiv(width, 128))](
        value,
        weights,
        output,
        width,
        topk,
        weights.stride(0),
        weights.stride(1),
        triton.next_power_of_2(topk),
        128,
        enable_fp_fusion=False,
    )
    return output


@triton.jit
def _rms_norm(X, W, Y, WIDTH: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    value = tl.load(X + row * WIDTH + col, col < WIDTH, 0).to(tl.float32)
    weight = tl.load(W + col, col < WIDTH, 0).to(tl.float32)
    variance = tl.sum(value * value, axis=0) / WIDTH
    normalized = value * tl.rsqrt(variance + EPS)
    tl.store(Y + row * WIDTH + col, (normalized * weight).to(tl.bfloat16), col < WIDTH)


def rms_norm(value, weight, eps):
    value = value.contiguous()
    output = torch.empty_like(value)
    width = value.shape[-1]
    if value.numel():
        _rms_norm[(value.numel() // width,)](
            value,
            weight,
            output,
            width,
            eps,
            triton.next_power_of_2(width),
            enable_fp_fusion=False,
        )
    return output
