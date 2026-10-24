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
def _silu_mul(X, Y, WIDTH: tl.constexpr, TOTAL, BLOCK: tl.constexpr):
    # A runtime mask bound reuses the kernel across unseen prefill lengths.
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = offset // WIDTH, offset % WIDTH
    gate = tl.load(X + row * (2 * WIDTH) + col, offset < TOTAL, 0).to(tl.float32)
    up = tl.load(X + row * (2 * WIDTH) + WIDTH + col, offset < TOTAL, 0).to(tl.float32)
    # Explicit round-trip keeps SiLU's BF16 rounding inside the fused kernel.
    silu = tl.div_rn(gate, 1.0 + libdevice.exp(-gate))
    rounded = silu.to(tl.bfloat16).to(tl.float32)
    tl.store(Y + offset, (rounded * up).to(tl.bfloat16), offset < TOTAL)


def silu_mul(value, out=None):
    """``silu(gate) * up`` with each operation rounded to BF16, as separate ops would."""
    if value.ndim != 2 or value.dtype != torch.bfloat16:
        raise ValueError("silu_mul requires a 2D BF16 [rows, 2 * width] tensor")
    value = value.contiguous()
    width = value.shape[-1] // 2
    if out is None:
        out = torch.empty(
            (value.shape[0], width), device=value.device, dtype=value.dtype
        )
    elif (
        out.shape != (value.shape[0], width)
        or out.dtype != value.dtype
        or not out.is_contiguous()
    ):
        raise ValueError("silu_mul out must be a contiguous BF16 [rows, width] tensor")
    if out.numel():
        _silu_mul[(triton.cdiv(out.numel(), 256),)](
            value, out, width, out.numel(), 256, enable_fp_fusion=False
        )
    return out


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
def _scaled(v, SCALE, HAS_SCALE: tl.constexpr):
    # PyTorch multiplies BF16 by a Python scalar in FP32 and rounds to BF16; the
    # norm then reads that rounded tensor. Keep that rounding point.
    if HAS_SCALE:
        return (v * SCALE).to(tl.bfloat16).to(tl.float32)
    return v


@triton.jit
def _norm_input(
    X, A, offset, mask, SCALE, HAS_SCALE: tl.constexpr, HAS_PRE_ADD: tl.constexpr
):
    value = tl.load(X + offset, mask, 0).to(tl.float32)
    if HAS_PRE_ADD:
        addition = tl.load(A + offset, mask, 0).to(tl.float32)
        # The separate shared-expert addition materializes BF16 before RMS norm.
        value = (value + addition).to(tl.bfloat16).to(tl.float32)
    return _scaled(value, SCALE, HAS_SCALE)


@triton.jit
def _rms_norm(
    X,
    W,
    Y,
    R,
    A,
    SCALE,
    WIDTH: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
    THREADS: tl.constexpr,
    VECTOR: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    HAS_PRE_ADD: tl.constexpr,
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
        v0 = _norm_input(
            X,
            A,
            row * WIDTH + col + 0 * step,
            col + 0 * step < WIDTH,
            SCALE,
            HAS_SCALE,
            HAS_PRE_ADD,
        )
        v1 = _norm_input(
            X,
            A,
            row * WIDTH + col + 1 * step,
            col + 1 * step < WIDTH,
            SCALE,
            HAS_SCALE,
            HAS_PRE_ADD,
        )
        v2 = _norm_input(
            X,
            A,
            row * WIDTH + col + 2 * step,
            col + 2 * step < WIDTH,
            SCALE,
            HAS_SCALE,
            HAS_PRE_ADD,
        )
        v3 = _norm_input(
            X,
            A,
            row * WIDTH + col + 3 * step,
            col + 3 * step < WIDTH,
            SCALE,
            HAS_SCALE,
            HAS_PRE_ADD,
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
    value = _norm_input(
        X, A, row * WIDTH + col, col < WIDTH, SCALE, HAS_SCALE, HAS_PRE_ADD
    )
    weight = tl.load(W + col, col < WIDTH, 0).to(tl.float32)
    normalized = value * libdevice.rsqrt(variance + EPS)
    output = (normalized * weight).to(tl.bfloat16)
    if HAS_RESIDUAL:
        # Core adds the BF16 norm output to the BF16 residual: FP32 add, one rounding.
        residual = tl.load(R + row * WIDTH + col, col < WIDTH, 0).to(tl.float32)
        output = (output.to(tl.float32) + residual).to(tl.bfloat16)
    tl.store(Y + row * WIDTH + col, output, col < WIDTH)


def rms_norm(value, weight, eps, residual=None, scale=None, pre_add=None):
    """RMS norm with Core's rounding; optionally ``scale`` the input and add ``residual``.

    ``scale`` reproduces ``value * scale`` as a separate BF16 multiply before the
    norm; ``residual`` reproduces ``residual + norm(value)`` after it.
    ``pre_add`` adds a BF16 tensor before scaling and normalization, preserving
    the intermediate BF16 rounding of the separate addition.
    """
    value = value.contiguous()
    width = value.shape[-1]
    if residual is not None and (
        residual.shape != value.shape or residual.dtype != value.dtype
    ):
        raise ValueError("rms_norm residual must match the input shape and dtype")
    if pre_add is not None and (
        pre_add.shape != value.shape
        or pre_add.dtype != value.dtype
        or pre_add.device != value.device
    ):
        raise ValueError(
            "rms_norm pre_add must match the input shape, dtype and device"
        )
    if width > 4096 or (width >= 128 and width % 4):
        if pre_add is not None:
            value = value + pre_add
        if scale is not None:
            value = value * scale
        fp32 = value.float()
        normalized = fp32 * torch.rsqrt(fp32.square().mean(-1, keepdim=True) + eps)
        output = (normalized * weight.float()).to(value.dtype)
        return output if residual is None else residual + output
    if residual is not None:
        residual = residual.contiguous()
    if pre_add is not None:
        pre_add = pre_add.contiguous()
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
            value if residual is None else residual,
            value if pre_add is None else pre_add,
            1.0 if scale is None else float(scale),
            width,
            eps,
            triton.next_power_of_2(width),
            threads,
            vector,
            residual is not None,
            scale is not None,
            pre_add is not None,
            enable_fp_fusion=False,
        )
    return output
