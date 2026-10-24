"""Fused exponential-race selection with Torch-compatible argmax tie handling."""

import torch
import triton
from triton import language as tl


@triton.jit
def _argmax_parts(
    P,
    Q,
    VALUES,
    INDICES,
    FLAGS,
    VOCAB: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    part = tl.program_id(1)
    columns = part * BLOCK + tl.arange(0, BLOCK)
    mask = columns < VOCAB
    probabilities = tl.load(P + row * VOCAB + columns, mask, 0)
    bad = (
        tl.sum(
            (
                mask
                & (
                    (probabilities < 0)
                    | (probabilities != probabilities)
                    | (probabilities == float("inf"))
                )
            ).to(tl.int32),
            0,
        )
        > 0
    )
    positive = tl.sum((mask & (probabilities > 0)).to(tl.int32), 0) > 0
    noise = tl.load(Q + row * VOCAB + columns, mask, 1)
    ratio = tl.div_rn(probabilities, noise)
    is_nan = ratio != ratio
    first_nan = tl.min(tl.where(mask & is_nan, columns, 2147483647), 0)
    maximum = tl.max(tl.where(mask & ~is_nan, ratio, -float("inf")), 0)
    first_max = tl.min(tl.where(mask & (ratio == maximum), columns, 2147483647), 0)
    has_nan = first_nan < 2147483647
    offset = row * PARTS + part
    tl.store(VALUES + offset, tl.where(has_nan, float("nan"), maximum))
    tl.store(INDICES + offset, tl.where(has_nan, first_nan, first_max))
    tl.store(FLAGS + offset, bad.to(tl.int32) | (positive.to(tl.int32) * 2))


@triton.jit
def _argmax_finish(
    VALUES, INDICES, FLAGS, OUTPUT, VALID, PARTS: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    mask = columns < PARTS
    values = tl.load(VALUES + row * PARTS + columns, mask, -float("inf"))
    indices = tl.load(INDICES + row * PARTS + columns, mask, 2147483647)
    flags = tl.load(FLAGS + row * PARTS + columns, mask, 0)
    is_nan = values != values
    first_nan = tl.min(tl.where(mask & is_nan, indices, 2147483647), 0)
    maximum = tl.max(tl.where(mask & ~is_nan, values, -float("inf")), 0)
    first_max = tl.min(tl.where(mask & (values == maximum), indices, 2147483647), 0)
    tl.store(OUTPUT + row, tl.where(first_nan < 2147483647, first_nan, first_max))
    bad = tl.sum(flags & 1, 0) > 0
    positive = tl.sum(flags & 2, 0) > 0
    tl.store(VALID + row, ~bad & positive)


def select_with_noise(probabilities, noise):
    """Return selected indices and a per-row validity flag; does not consume RNG."""
    if (
        not probabilities.is_cuda
        or probabilities.dtype != torch.float32
        or probabilities.ndim != 2
        or not probabilities.is_contiguous()
        or noise.shape != probabilities.shape
        or noise.dtype != probabilities.dtype
        or noise.device != probabilities.device
        or not noise.is_contiguous()
    ):
        raise ValueError(
            "sampler kernels require matching contiguous CUDA FP32 matrices"
        )
    rows, vocab = probabilities.shape
    if rows == 0 or vocab == 0:
        raise ValueError("sampler requires nonempty matrices")
    parts = triton.cdiv(vocab, 4096)
    values = torch.empty(
        (rows, parts), dtype=torch.float32, device=probabilities.device
    )
    indices = torch.empty((rows, parts), dtype=torch.int32, device=probabilities.device)
    flags = torch.empty_like(indices)
    output = torch.empty((rows,), dtype=torch.int32, device=probabilities.device)
    valid = torch.empty((rows,), dtype=torch.bool, device=probabilities.device)
    _argmax_parts[(rows, parts)](
        probabilities,
        noise,
        values,
        indices,
        flags,
        vocab,
        parts,
        4096,
        num_warps=4,
        enable_fp_fusion=False,
    )
    _argmax_finish[(rows,)](
        values,
        indices,
        flags,
        output,
        valid,
        parts,
        triton.next_power_of_2(parts),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output, valid
