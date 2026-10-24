"""Selected-expert weight normalization without replacing softmax or top-k."""

import torch
import triton
from triton import language as tl


@triton.jit
def _routing_tail(
    W,
    IDS,
    OUT_W,
    OUT_I,
    TOKENS,
    NORMALIZE: tl.constexpr,
    RESTORE: tl.constexpr,
    CORRECTION: tl.constexpr,
    ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    slots = tl.arange(0, 16)
    offsets = rows[:, None] * 16 + slots[None, :]
    weights = tl.load(W + offsets, rows[:, None] < TOKENS, 0)
    if NORMALIZE:
        # The pinned PyTorch 16-element L1 reduction groups a row across 16 lanes.
        denominator = tl.sum(tl.abs(weights), axis=1)
        weights = tl.div_rn(weights, denominator[:, None])
    if RESTORE:
        weights = weights * 16.0
    if CORRECTION != 1.0:
        # Keep this separate from restore: Torch rounds each FP32 multiply.
        weights = weights * CORRECTION
    indices = tl.load(IDS + offsets, rows[:, None] < TOKENS, 0).to(tl.int32)
    tl.store(OUT_W + offsets, weights, rows[:, None] < TOKENS)
    tl.store(OUT_I + offsets, indices, rows[:, None] < TOKENS)


def finish_topk(weights, indices, *, normalize, restore, original_topk):
    """Fuse the tail for FP32 CUDA top-16 scores; caller handles other cases."""
    if (
        not weights.is_cuda
        or weights.dtype != torch.float32
        or indices.dtype != torch.int64
        or weights.ndim != 2
        or weights.shape[-1] != 16
        or indices.shape != weights.shape
        or indices.device != weights.device
    ):
        raise ValueError(
            "routing tail requires matching CUDA FP32/int64 top-16 tensors"
        )
    weights, indices = weights.contiguous(), indices.contiguous()
    out_weights = torch.empty_like(weights)
    out_indices = torch.empty_like(indices, dtype=torch.int32)
    tokens = weights.shape[0]
    if tokens:
        correction = 1.0 if original_topk is None else (original_topk / 16) ** 0.5
        _routing_tail[(triton.cdiv(tokens, 8),)](
            weights,
            indices,
            out_weights,
            out_indices,
            tokens,
            normalize,
            restore,
            correction,
            8,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out_weights, out_indices
