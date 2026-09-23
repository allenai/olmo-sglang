"""Per-head attention gains and position-dependent query scaling."""

from __future__ import annotations

import torch
from torch import nn


def local_head_range(total_heads: int, tp_size: int, tp_rank: int) -> tuple[int, int]:
    """Match QKVParallelLinear's contiguous shards and replicated KV heads."""
    if total_heads < 1 or tp_size < 1 or not 0 <= tp_rank < tp_size:
        raise ValueError("Invalid attention head count or TP topology")
    if total_heads >= tp_size:
        if total_heads % tp_size:
            raise ValueError("Attention heads must divide evenly across TP ranks")
        count = total_heads // tp_size
        return tp_rank * count, count
    if tp_size % total_heads:
        raise ValueError("Replicated KV heads must divide the TP size")
    return tp_rank // (tp_size // total_heads), 1


@torch.no_grad()
def load_head_weight(
    parameter: nn.Parameter,
    loaded_weight: torch.Tensor,
    *,
    total_heads: int,
    tp_size: int,
    tp_rank: int,
) -> None:
    """Load full HF head tensors or an already-local shard without replacing storage."""
    start, count = local_head_range(total_heads, tp_size, tp_rank)
    if parameter.shape[0] != count:
        raise ValueError("Parameter head count disagrees with attention TP topology")
    full_shape = (total_heads, *parameter.shape[1:])
    if tuple(loaded_weight.shape) == full_shape:
        local = loaded_weight.narrow(0, start, count)
    elif tuple(loaded_weight.shape) == tuple(parameter.shape):
        local = loaded_weight
    else:
        raise ValueError(
            f"Head weight shape {tuple(loaded_weight.shape)} does not match "
            f"HF {full_shape} or local {tuple(parameter.shape)}"
        )
    parameter.copy_(local)
    parameter._olmo_head_weight_loaded = True


class PerHeadRMSNorm(nn.Module):
    """FP32 normalization and per-head gain, followed by one activation-dtype cast."""

    def __init__(self, num_heads: int, head_dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_heads, head_dim))
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-2:] != self.weight.shape:
            raise ValueError("Per-head RMSNorm expects [..., heads, head_dim]")
        values = inputs.float()
        normalized = values * torch.rsqrt(
            values.square().mean(-1, keepdim=True) + self.eps
        )
        return (normalized * self.weight.float()).to(inputs.dtype)


def scale_attention_queries(
    query: torch.Tensor, positions: torch.Tensor, head_scale: torch.Tensor
) -> torch.Tensor:
    """Scale packed THD queries using SGLang's absolute per-request positions."""
    if query.ndim != 3 or positions.ndim != 1 or head_scale.ndim != 1:
        raise ValueError(
            "Scalable softmax expects THD queries, positions[T], scales[H]"
        )
    if query.shape[0] != positions.shape[0] or query.shape[1] != head_scale.shape[0]:
        raise ValueError("Scalable softmax token/head dimensions disagree")
    if positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("Scalable softmax requires integer token positions")
    # HF/Core form both scale factors in activation dtype before multiplying Q.
    visible = (positions + 1).log().to(query.dtype)
    combined = visible[:, None] * head_scale.to(query.dtype)[None, :]
    return query * combined[:, :, None]


def check_head_weights_loaded(parameters: dict[str, nn.Parameter]) -> None:
    """Reject missing per-head gain and softmax-scale tensors before the first forward pass."""
    missing = [
        name
        for name, parameter in parameters.items()
        if not getattr(parameter, "_olmo_head_weight_loaded", False)
    ]
    if missing:
        raise RuntimeError(
            f"Attention gain or scale weights were not loaded: {', '.join(missing)}"
        )
