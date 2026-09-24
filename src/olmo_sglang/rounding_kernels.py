"""Small fused operations with explicit Core-compatible precision boundaries.

The expert GEMMs and their tuning remain SGLang's. These kernels fuse only the
surrounding elementwise work; they never move weighting ahead of the BF16 down
output. For the qualified shapes, reductions preserve the pinned PyTorch
CUDA four-accumulator grouping as well as the BF16 boundaries.
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
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    a0 = tl.full((BLOCK,), 0, tl.float32)
    a1 = tl.full((BLOCK,), 0, tl.float32)
    a2 = tl.full((BLOCK,), 0, tl.float32)
    a3 = tl.full((BLOCK,), 0, tl.float32)
    # PyTorch's strided sum uses four independent accumulators, then a
    # left-associated combination. Keep each product's FP32 rounding too.
    for base in tl.static_range(0, TOPK, 4):
        v0 = tl.load(
            X + (token * TOPK + base) * WIDTH + columns, columns < WIDTH, 0
        ).to(tl.float32)
        w0 = tl.load(W + token * WEIGHT_STRIDE + base * WEIGHT_SLOT_STRIDE).to(
            tl.float32
        )
        a0 += v0 * w0
        if base + 1 < TOPK:
            v1 = tl.load(
                X + (token * TOPK + base + 1) * WIDTH + columns, columns < WIDTH, 0
            ).to(tl.float32)
            w1 = tl.load(
                W + token * WEIGHT_STRIDE + (base + 1) * WEIGHT_SLOT_STRIDE
            ).to(tl.float32)
            a1 += v1 * w1
        if base + 2 < TOPK:
            v2 = tl.load(
                X + (token * TOPK + base + 2) * WIDTH + columns, columns < WIDTH, 0
            ).to(tl.float32)
            w2 = tl.load(
                W + token * WEIGHT_STRIDE + (base + 2) * WEIGHT_SLOT_STRIDE
            ).to(tl.float32)
            a2 += v2 * w2
        if base + 3 < TOPK:
            v3 = tl.load(
                X + (token * TOPK + base + 3) * WIDTH + columns, columns < WIDTH, 0
            ).to(tl.float32)
            w3 = tl.load(
                W + token * WEIGHT_STRIDE + (base + 3) * WEIGHT_SLOT_STRIDE
            ).to(tl.float32)
            a3 += v3 * w3
    result = ((a0 + a1) + a2) + a3
    tl.store(Y + token * WIDTH + columns, result.to(tl.bfloat16), columns < WIDTH)


def weighted_sum(value, weights):
    tokens, topk, width = value.shape
    # Larger reductions can select a different PyTorch block decomposition.
    if topk > 16:
        return (value.float() * weights.float().unsqueeze(-1)).sum(1).to(value.dtype)
    output = torch.empty((tokens, width), device=value.device, dtype=value.dtype)
    _weighted_sum[(tokens, triton.cdiv(width, 128))](
        value,
        weights,
        output,
        width,
        topk,
        weights.stride(0),
        weights.stride(1),
        128,
        enable_fp_fusion=False,
    )
    return output


@triton.jit
def _rms_norm(
    X,
    W,
    Y,
    WIDTH: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
    THREADS: tl.constexpr,
    VECTOR: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, THREADS)
    a0 = tl.full((THREADS,), 0, tl.float32)
    a1 = tl.full((THREADS,), 0, tl.float32)
    a2 = tl.full((THREADS,), 0, tl.float32)
    a3 = tl.full((THREADS,), 0, tl.float32)
    # Match Reduce.cuh's vectorized per-thread accumulation and final grouping.
    for base in tl.static_range(0, tl.cdiv(WIDTH, THREADS * 4)):
        if VECTOR == 4:
            col = (lane + base * THREADS) * 4
            step = 1
        else:
            col = lane + base * THREADS * 4
            step = THREADS
        v0 = tl.load(X + row * WIDTH + col, col < WIDTH, 0).to(tl.float32)
        v1 = tl.load(X + row * WIDTH + col + step, col + step < WIDTH, 0).to(tl.float32)
        v2 = tl.load(X + row * WIDTH + col + 2 * step, col + 2 * step < WIDTH, 0).to(
            tl.float32
        )
        v3 = tl.load(X + row * WIDTH + col + 3 * step, col + 3 * step < WIDTH, 0).to(
            tl.float32
        )
        a0 += v0 * v0
        a1 += v1 * v1
        a2 += v2 * v2
        a3 += v3 * v3
    total = ((a0 + a1) + a2) + a3
    # PyTorch folds across warps before its descending intra-warp reduction.
    if THREADS >= 512:
        total = tl.sum(total.reshape((2, 256)), axis=0)
    if THREADS >= 256:
        total = tl.sum(total.reshape((2, 128)), axis=0)
    if THREADS >= 128:
        total = tl.sum(total.reshape((2, 64)), axis=0)
    if THREADS >= 64:
        total = tl.sum(total.reshape((2, 32)), axis=0)
    variance = tl.sum(total, axis=0) * (1.0 / WIDTH)
    col = tl.arange(0, BLOCK)
    value = tl.load(X + row * WIDTH + col, col < WIDTH, 0).to(tl.float32)
    weight = tl.load(W + col, col < WIDTH, 0).to(tl.float32)
    normalized = value * libdevice.rsqrt(variance + EPS)
    tl.store(Y + row * WIDTH + col, (normalized * weight).to(tl.bfloat16), col < WIDTH)


def rms_norm(value, weight, eps):
    value = value.contiguous()
    width = value.shape[-1]
    if width > 4096 or (width >= 128 and width % 4):
        fp32 = value.float()
        normalized = fp32 * torch.rsqrt(fp32.square().mean(-1, keepdim=True) + eps)
        return (normalized * weight.float()).to(value.dtype)
    output = torch.empty_like(value)
    rows = value.numel() // width
    if rows:
        vector = 4 if width >= 128 else 1
        dim0 = min(512, 1 << ((width // vector).bit_length() - 1))
        dim1 = min(512, 1 << (rows.bit_length() - 1))
        block_height = min(dim1, 512 // min(dim0, 32))
        threads = min(dim0, 512 // block_height)
        _rms_norm[(rows,)](
            value,
            weight,
            output,
            width,
            eps,
            triton.next_power_of_2(width),
            threads,
            vector,
            enable_fp_fusion=False,
        )
    return output
