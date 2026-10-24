"""One launch containing two distinct BF16 GEMMs; no weight packing or copies."""

import torch
import triton
from triton import language as tl


@triton.jit
def _one_gate(
    X,
    W,
    BIAS,
    Y,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    INNER: tl.constexpr,
    STRIDE: tl.constexpr,
    PID_N,
    HAS_BIAS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = PID_N * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0.0, tl.float32)
    for start in range(tl.cdiv(INNER, BK)):
        kk = start * BK + k
        x = tl.load(
            X + m[:, None] * STRIDE + kk[None, :],
            (m[:, None] < ROWS) & (kk[None, :] < INNER),
            0,
        )
        w = tl.load(
            W + n[None, :] * INNER + kk[:, None],
            (n[None, :] < COLS) & (kk[:, None] < INNER),
            0,
        )
        acc = tl.dot(x, w, acc)
    if HAS_BIAS:
        bias = tl.load(BIAS + n, n < COLS, 0).to(tl.float32)
        acc = acc + bias[None, :]
    tl.store(
        Y + m[:, None] * COLS + n[None, :],
        acc.to(Y.dtype.element_ty),
        (m[:, None] < ROWS) & (n[None, :] < COLS),
    )


@triton.jit
def _grouped_gates(
    FIRST,
    SECOND,
    FIRST_W,
    SECOND_W,
    BIAS,
    FIRST_OUT,
    SECOND_OUT,
    ROWS: tl.constexpr,
    FIRST_COLS: tl.constexpr,
    SECOND_COLS: tl.constexpr,
    INNER: tl.constexpr,
    FIRST_STRIDE: tl.constexpr,
    SECOND_STRIDE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(1)
    first_tiles = tl.cdiv(FIRST_COLS, BN)
    if pid < first_tiles:
        _one_gate(
            FIRST,
            FIRST_W,
            BIAS,
            FIRST_OUT,
            ROWS,
            FIRST_COLS,
            INNER,
            FIRST_STRIDE,
            pid,
            False,
            BM,
            BN,
            BK,
        )
    else:
        _one_gate(
            SECOND,
            SECOND_W,
            BIAS,
            SECOND_OUT,
            ROWS,
            SECOND_COLS,
            INNER,
            SECOND_STRIDE,
            pid - first_tiles,
            True,
            BM,
            BN,
            BK,
        )


def project(first, second, first_weight, second_weight, second_bias):
    rows, inner = first.shape
    nf, ng = first_weight.shape[0], second_weight.shape[0]
    outputs = (
        torch.empty((rows, nf), device=first.device, dtype=first.dtype),
        torch.empty((rows, ng), device=first.device, dtype=first.dtype),
    )
    if rows:
        _grouped_gates[
            (triton.cdiv(rows, 16), triton.cdiv(nf, 64) + triton.cdiv(ng, 64))
        ](
            first,
            second,
            first_weight,
            second_weight,
            second_bias,
            *outputs,
            rows,
            nf,
            ng,
            inner,
            first.stride(0),
            second.stride(0),
            16,
            64,
            64,
            num_warps=4,
        )
    return outputs
