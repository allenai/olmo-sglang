# SPDX-License-Identifier: Apache-2.0

"""SwiGLU activation with Core's separate-operation rounding for OLMo shapes."""

import logging
from collections.abc import Callable

import torch
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)


def _fused_silu_mul_supported(gate_up: torch.Tensor) -> bool:
    return gate_up.is_cuda and gate_up.dtype == torch.bfloat16 and gate_up.ndim == 2


def native_silu_and_mul(
    gate_up: torch.Tensor,
    out: torch.Tensor | None = None,
    expert_ids: torch.Tensor | None = None,
    expert_step: int = 1,
) -> torch.Tensor:
    """Apply SwiGLU, rounding ``silu(gate)`` to the activation dtype before the product.

    BF16 CUDA inputs without an expert filter use one Triton kernel that reproduces
    the separate ``F.silu`` and multiply roundings bitwise (see ``rounding_kernels``
    and its exhaustive test). Other inputs use the tensor operations, which also
    avoid SGLang's vector-width-constrained JIT kernel for unaligned widths.
    """

    if expert_ids is None and _fused_silu_mul_supported(gate_up):
        # Imported lazily so the CPU-only suite can import this module without Triton.
        from olmo_sglang import rounding_kernels

        return rounding_kernels.silu_mul(gate_up, out=out)

    gate, up = gate_up.chunk(2, dim=-1)
    result = F.silu(gate) * up
    if expert_ids is None:
        if out is None:
            return result
        out.copy_(result)
        return out

    if expert_step < 1:
        raise ValueError(f"expert_step must be positive, got {expert_step}")

    out_2d = out.view(-1, out.shape[-1])
    result_2d = result.view_as(out_2d)
    row_ids = torch.arange(out_2d.shape[0], device=expert_ids.device)
    active_rows = expert_ids[row_ids // expert_step] != -1
    # torch.where keeps this free of host synchronization and graph-capturable.
    out_2d.copy_(torch.where(active_rows.unsqueeze(-1), result_2d, out_2d))
    return out


def _requires_native_silu_and_mul(gate_up: torch.Tensor) -> bool:
    """Return whether SGLang's JIT vector width cannot represent this shape."""

    hidden_size = gate_up.shape[-1] // 2
    max_vector_bytes = 16
    if gate_up.is_cuda and torch.cuda.get_device_capability(gate_up.device)[0] >= 10:
        max_vector_bytes = 32
    vector_size = max_vector_bytes // gate_up.element_size()
    return hidden_size % vector_size != 0


def install_sglang_moe_activation_fallback() -> None:
    """Install a native fallback for routed MoE widths rejected by SGLang JIT."""

    from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe

    if getattr(fused_moe, "_olmo_unaligned_silu_fallback_installed", False):
        return

    original: Callable[..., torch.Tensor] = fused_moe.silu_and_mul

    def compatible_silu_and_mul(
        gate_up: torch.Tensor,
        out: torch.Tensor | None = None,
        expert_ids: torch.Tensor | None = None,
        expert_step: int = 1,
    ) -> torch.Tensor:
        if not _requires_native_silu_and_mul(gate_up):
            return original(gate_up, out, expert_ids, expert_step)
        return native_silu_and_mul(gate_up, out, expert_ids, expert_step)

    fused_moe.silu_and_mul = compatible_silu_and_mul
    fused_moe._olmo_unaligned_silu_fallback_installed = True
    LOGGER.info(
        "Installed SGLang MoE activation fallback for unaligned OLMo expert widths"
    )
