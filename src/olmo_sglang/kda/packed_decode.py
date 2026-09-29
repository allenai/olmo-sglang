# Copyright 2026 Allen Institute for AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Packed one-token KDA decode with OLMo's beta activation semantics."""

# Adapted from SGLang's fused_recurrent_kda_packed_decode kernel, which in turn
# derives from Flash Linear Attention (Copyright 2023-2025 Songlin Yang and
# Yu Zhang). The overlay keeps the kernel local so the pinned SGLang checkout
# remains unmodified.
# Original source notice: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# See THIRD_PARTY_NOTICES.md and LICENSES/FLA-MIT.txt for provenance and the
# retained MIT permission and warranty notice.

from typing import Any

import torch
import triton
import triton.language as tl
from sglang.kernels.ops.attention.fla.op import exp


@triton.jit
def _olmo_packed_kda_decode_kernel(
    mixed_qkv,
    a,
    b,
    a_log,
    dt_bias,
    output,
    state,
    state_indices,
    scale,
    stride_mixed_token: tl.constexpr,
    stride_a_token: tl.constexpr,
    stride_b_token: tl.constexpr,
    stride_state_slot: tl.constexpr,
    stride_index: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_value_heads: tl.constexpr,
    key_dim: tl.constexpr,
    value_dim: tl.constexpr,
    block_key: tl.constexpr,
    block_value: tl.constexpr,
    beta_multiplier: tl.constexpr,
):
    """Fuse unpack, normalization, gates, recurrence, and state commit."""
    value_block = tl.program_id(0)
    sequence_head = tl.program_id(1)
    sequence = sequence_head // num_value_heads
    value_head = sequence_head % num_value_heads
    query_head = value_head // (num_value_heads // num_q_heads)

    key_offsets = tl.arange(0, block_key)
    value_offsets = value_block * block_value + tl.arange(0, block_value)
    key_mask = key_offsets < key_dim
    value_mask = value_offsets < value_dim
    state_mask = value_mask[:, None] & key_mask[None, :]

    state_index = tl.load(state_indices + sequence * stride_index).to(tl.int64)
    output_ptr = (
        output + (sequence * num_value_heads + value_head) * value_dim + value_offsets
    )
    if state_index < 0:
        tl.store(
            output_ptr,
            tl.zeros([block_value], dtype=tl.float32).to(output_ptr.dtype.element_ty),
            mask=value_mask,
        )
        return

    state_ptr = (
        state
        + state_index * stride_state_slot
        + value_head * value_dim * key_dim
        + value_offsets[:, None] * key_dim
        + key_offsets[None, :]
    )
    recurrent_state = tl.load(state_ptr, mask=state_mask, other=0).to(tl.float32)

    mixed_ptr = mixed_qkv + sequence * stride_mixed_token
    query_offsets = query_head * key_dim + key_offsets
    key_offsets_packed = num_q_heads * key_dim + query_head * key_dim + key_offsets
    value_offsets_packed = (
        2 * num_q_heads * key_dim + value_head * value_dim + value_offsets
    )
    query = tl.load(mixed_ptr + query_offsets, mask=key_mask, other=0).to(tl.float32)
    key = tl.load(mixed_ptr + key_offsets_packed, mask=key_mask, other=0).to(tl.float32)
    value = tl.load(mixed_ptr + value_offsets_packed, mask=value_mask, other=0).to(
        tl.float32
    )

    query = query / tl.sqrt(tl.sum(query * query) + 1e-6)
    key = key / tl.sqrt(tl.sum(key * key) + 1e-6)
    query *= scale

    gate_ptr = a + sequence * stride_a_token + value_head * key_dim + key_offsets
    bias_ptr = dt_bias + value_head * key_dim + key_offsets
    raw_gate = tl.load(gate_ptr, mask=key_mask, other=0).to(tl.float32)
    gate_bias = tl.load(bias_ptr, mask=key_mask, other=0).to(tl.float32)
    decay_parameter = tl.load(a_log + value_head).to(tl.float32)
    gate_input = raw_gate + gate_bias
    softplus = tl.where(
        gate_input <= 20.0,
        tl.log(1.0 + tl.exp(gate_input)),
        gate_input,
    )
    log_decay = -tl.exp(decay_parameter) * softplus

    raw_beta = tl.load(b + sequence * stride_b_token + value_head).to(tl.float32)
    beta = beta_multiplier * tl.sigmoid(raw_beta)

    recurrent_state *= exp(log_decay)[None, :]
    value -= tl.sum(recurrent_state * key[None, :], axis=1)
    value *= beta
    recurrent_state += value[:, None] * key[None, :]
    result = tl.sum(recurrent_state * query[None, :], axis=1)

    tl.store(output_ptr, result.to(output_ptr.dtype.element_ty), mask=value_mask)
    tl.store(state_ptr, recurrent_state.to(state_ptr.dtype.element_ty), mask=state_mask)


def olmo_packed_kda_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    num_value_heads: int,
    value_dim: int,
    allow_neg_eigval: bool,
    **_: Any,
) -> torch.Tensor:
    """Run OLMo's packed KDA decode and update selected state slots in place."""
    if mixed_qkv.ndim != 2 or mixed_qkv.stride(-1) != 1:
        raise ValueError("mixed_qkv must be a last-dimension-contiguous 2D tensor")
    if state.ndim != 4 or state.stride(-1) != 1:
        raise ValueError(
            "state must be a last-dimension-contiguous [slots, HV, V, K] tensor"
        )
    batch_size = mixed_qkv.shape[0]
    state_heads, state_value_dim, key_dim = state.shape[-3:]
    if (state_heads, state_value_dim) != (num_value_heads, value_dim):
        raise ValueError(
            "state geometry does not match the requested value heads/dimension: "
            f"state={tuple(state.shape)}, heads={num_value_heads}, value_dim={value_dim}"
        )
    a = a.reshape(batch_size, num_value_heads * key_dim).contiguous()
    b = b.reshape(batch_size, num_value_heads).contiguous()
    a_log = a_log.reshape(num_value_heads).contiguous()
    dt_bias = dt_bias.reshape(num_value_heads * key_dim).contiguous()
    state_indices = state_indices.reshape(batch_size)
    packed_qk = mixed_qkv.shape[1] - num_value_heads * value_dim
    if packed_qk <= 0 or packed_qk % (2 * key_dim) != 0:
        raise ValueError(
            "mixed_qkv does not encode equal-width packed query and key heads"
        )
    num_q_heads = packed_qk // (2 * key_dim)
    if num_q_heads < 1 or num_value_heads % num_q_heads:
        raise ValueError(
            "num_value_heads must be divisible by the inferred query-head count"
        )
    tensors = (a, b, a_log, dt_bias, state, state_indices)
    if any(tensor.device != mixed_qkv.device for tensor in tensors):
        raise ValueError("all packed KDA tensors must be on the same device")

    output = mixed_qkv.new_empty(batch_size, 1, num_value_heads, value_dim)
    block_key = triton.next_power_of_2(key_dim)
    if triton.cdiv(key_dim, block_key) != 1:
        raise ValueError("packed KDA decode requires a single key-dimension block")
    block_value = min(triton.next_power_of_2(value_dim), 32)
    grid = (triton.cdiv(value_dim, block_value), batch_size * num_value_heads)
    _olmo_packed_kda_decode_kernel[grid](
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        a_log=a_log,
        dt_bias=dt_bias,
        output=output,
        state=state,
        state_indices=state_indices,
        scale=scale,
        stride_mixed_token=mixed_qkv.stride(0),
        stride_a_token=a.stride(0),
        stride_b_token=b.stride(0),
        stride_state_slot=state.stride(0),
        stride_index=state_indices.stride(0),
        num_q_heads=num_q_heads,
        num_value_heads=num_value_heads,
        key_dim=key_dim,
        value_dim=value_dim,
        block_key=block_key,
        block_value=block_value,
        beta_multiplier=2.0 if allow_neg_eigval else 1.0,
        num_warps=1,
        num_stages=3,
    )
    return output.transpose(0, 1)
