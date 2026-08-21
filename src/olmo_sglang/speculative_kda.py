# Copyright 2026 Allen Institute for AI
# Copyright 2026 NVIDIA CORPORATION. All rights reserved.
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

"""Fused OLMo KDA target verification for speculative decoding."""

import torch
import triton
import triton.language as tl
from sglang.kernels.ops.attention.fla.op import exp


@triton.jit(do_not_specialize=["total_tokens"])
def _olmo_kda_target_verify_kernel(
    a_log,
    raw_gate,
    dt_bias,
    q,
    k,
    v,
    raw_beta,
    output,
    state,
    state_indices,
    query_start_loc,
    scratch,
    scratch_indices,
    retrieve_parent_token,
    scale,
    total_tokens,
    stride_gate_token: tl.constexpr,
    stride_gate_head: tl.constexpr,
    stride_q_token: tl.constexpr,
    stride_q_head: tl.constexpr,
    stride_k_token: tl.constexpr,
    stride_k_head: tl.constexpr,
    stride_v_token: tl.constexpr,
    stride_v_head: tl.constexpr,
    stride_beta_token: tl.constexpr,
    stride_beta_head: tl.constexpr,
    stride_output_token: tl.constexpr,
    stride_output_head: tl.constexpr,
    stride_state_slot: tl.constexpr,
    stride_state_head: tl.constexpr,
    stride_state_value: tl.constexpr,
    stride_scratch_slot: tl.constexpr,
    stride_scratch_step: tl.constexpr,
    stride_scratch_head: tl.constexpr,
    stride_scratch_value: tl.constexpr,
    stride_parent_request: tl.constexpr,
    stride_parent_step: tl.constexpr,
    num_query_heads: tl.constexpr,
    num_key_heads: tl.constexpr,
    num_value_heads: tl.constexpr,
    key_dim: tl.constexpr,
    value_dim: tl.constexpr,
    block_key: tl.constexpr,
    block_value: tl.constexpr,
    beta_multiplier: tl.constexpr,
    has_parent_tree: tl.constexpr,
):
    """Verify one request/head/value block and save each scratch state."""
    value_block = tl.program_id(0)
    request_head = tl.program_id(1)
    request = request_head // num_value_heads
    value_head = request_head % num_value_heads
    query_head = value_head // (num_value_heads // num_query_heads)
    key_head = value_head // (num_value_heads // num_key_heads)

    start = tl.load(query_start_loc + request).to(tl.int64)
    end = tl.load(query_start_loc + request + 1).to(tl.int64)
    request_steps = end - start
    state_index = tl.load(state_indices + request).to(tl.int64)
    scratch_index = tl.load(scratch_indices + request).to(tl.int64)
    valid_state = state_index >= 0

    key_offsets = tl.arange(0, block_key)
    value_offsets = value_block * block_value + tl.arange(0, block_value)
    key_mask = key_offsets < key_dim
    value_mask = value_offsets < value_dim
    state_mask = value_mask[:, None] & key_mask[None, :]

    state_ptr = (
        state
        + state_index * stride_state_slot
        + value_head * stride_state_head
        + value_offsets[:, None] * stride_state_value
        + key_offsets[None, :]
    )
    recurrent_state = tl.load(
        state_ptr,
        mask=state_mask & valid_state,
        other=0.0,
    ).to(tl.float32)

    step = 0
    for _ in range(0, request_steps):  # noqa: PIE808
        if has_parent_tree:  # noqa: SIM102
            if step != 0:
                parent_step = tl.load(
                    retrieve_parent_token
                    + request * stride_parent_request
                    + step * stride_parent_step
                ).to(tl.int64)
                parent_ptr = (
                    scratch
                    + scratch_index * stride_scratch_slot
                    + parent_step * stride_scratch_step
                    + value_head * stride_scratch_head
                    + value_offsets[:, None] * stride_scratch_value
                    + key_offsets[None, :]
                )
                recurrent_state = tl.load(
                    parent_ptr,
                    mask=state_mask & valid_state,
                    other=0.0,
                ).to(tl.float32)

        token = start + step
        query_ptr = (
            q + token * stride_q_token + query_head * stride_q_head + key_offsets
        )
        key_ptr = k + token * stride_k_token + key_head * stride_k_head + key_offsets
        value_ptr = (
            v + token * stride_v_token + value_head * stride_v_head + value_offsets
        )
        query = tl.load(query_ptr, mask=key_mask, other=0.0).to(tl.float32)
        key = tl.load(key_ptr, mask=key_mask, other=0.0).to(tl.float32)
        value = tl.load(value_ptr, mask=value_mask, other=0.0).to(tl.float32)
        query /= tl.sqrt(tl.sum(query * query) + 1e-6)
        key /= tl.sqrt(tl.sum(key * key) + 1e-6)
        query *= scale

        gate_ptr = (
            raw_gate
            + token * stride_gate_token
            + value_head * stride_gate_head
            + key_offsets
        )
        bias_ptr = dt_bias + value_head * key_dim + key_offsets
        gate_input = tl.load(gate_ptr, mask=key_mask, other=0.0).to(tl.float32)
        gate_input += tl.load(bias_ptr, mask=key_mask, other=0.0).to(tl.float32)
        softplus = tl.where(
            gate_input <= 20.0,
            tl.log(1.0 + tl.exp(gate_input)),
            gate_input,
        )
        decay_parameter = tl.load(a_log + value_head).to(tl.float32)
        log_decay = -tl.exp(decay_parameter) * softplus

        beta_ptr = raw_beta + token * stride_beta_token + value_head * stride_beta_head
        beta = beta_multiplier * tl.sigmoid(tl.load(beta_ptr).to(tl.float32))

        recurrent_state *= exp(log_decay)[None, :]
        value -= tl.sum(recurrent_state * key[None, :], axis=1)
        value *= beta
        recurrent_state += value[:, None] * key[None, :]
        result = tl.sum(recurrent_state * query[None, :], axis=1)

        output_ptr = (
            output
            + token * stride_output_token
            + value_head * stride_output_head
            + value_offsets
        )
        result = tl.where(valid_state, result, 0.0)
        tl.store(output_ptr, result.to(output_ptr.dtype.element_ty), mask=value_mask)

        scratch_ptr = (
            scratch
            + scratch_index * stride_scratch_slot
            + step * stride_scratch_step
            + value_head * stride_scratch_head
            + value_offsets[:, None] * stride_scratch_value
            + key_offsets[None, :]
        )
        saved_state = tl.where(valid_state, recurrent_state, 0.0)
        tl.store(
            scratch_ptr,
            saved_state.to(scratch_ptr.dtype.element_ty),
            mask=state_mask,
        )
        step += 1  # noqa: SIM113


def olmo_kda_target_verify(
    *,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    scratch: torch.Tensor,
    scratch_indices: torch.Tensor,
    cache_steps: int,
    retrieve_parent_token: torch.Tensor | None,
    allow_neg_eigval: bool,
) -> torch.Tensor:
    """Run graph-compatible fused KDA verification without committing state."""
    if q.ndim != 4 or q.shape[0] != 1:
        raise ValueError("q must use packed [1, tokens, heads, key_dim] layout")
    if k.ndim != 4 or k.shape[0] != 1 or k.shape[1] != q.shape[1]:
        raise ValueError("k must match q's packed batch and token dimensions")
    if v.ndim != 4 or v.shape[:2] != q.shape[:2]:
        raise ValueError("v must match q's packed batch and token dimensions")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must have the same key dimension")

    total_tokens = q.shape[1]
    num_query_heads = q.shape[2]
    num_key_heads = k.shape[2]
    num_value_heads = v.shape[2]
    key_dim = q.shape[-1]
    value_dim = v.shape[-1]
    if num_value_heads % num_query_heads or num_value_heads % num_key_heads:
        raise ValueError("value heads must be divisible by query and key heads")
    if state.ndim != 4 or state.shape[1:] != (
        num_value_heads,
        value_dim,
        key_dim,
    ):
        raise ValueError(
            "state must have shape [slots, value_heads, value_dim, key_dim]"
        )
    if scratch.ndim != 5 or scratch.shape[2:] != state.shape[1:]:
        raise ValueError(
            "scratch must have shape [slots, steps, value_heads, value_dim, key_dim]"
        )
    if cache_steps < 1 or scratch.shape[1] < cache_steps:
        raise ValueError("scratch must provide at least cache_steps entries per slot")
    if query_start_loc.ndim != 1 or query_start_loc.numel() < 2:
        raise ValueError("query_start_loc must contain packed request boundaries")

    batch_size = query_start_loc.numel() - 1
    if state_indices.numel() < batch_size or scratch_indices.numel() < batch_size:
        raise ValueError("state and scratch indices must cover the verify batch")
    if retrieve_parent_token is not None and (
        retrieve_parent_token.ndim != 2
        or retrieve_parent_token.shape[0] < batch_size
        or retrieve_parent_token.shape[1] < cache_steps
    ):
        raise ValueError("retrieve_parent_token must have shape [batch, cache_steps]")

    tensors = (
        a_log,
        dt_bias,
        q,
        k,
        v,
        raw_gate,
        raw_beta,
        state,
        state_indices,
        query_start_loc,
        scratch,
        scratch_indices,
    )
    if not q.is_cuda or any(tensor.device != q.device for tensor in tensors):
        raise ValueError("all fused KDA verification tensors must share a CUDA device")
    if retrieve_parent_token is not None and retrieve_parent_token.device != q.device:
        raise ValueError("retrieve_parent_token must share the KDA CUDA device")

    raw_gate = raw_gate.reshape(1, total_tokens, num_value_heads, key_dim)
    raw_beta = raw_beta.reshape(1, total_tokens, num_value_heads)
    a_log = a_log.reshape(num_value_heads).contiguous()
    dt_bias = dt_bias.reshape(num_value_heads, key_dim).contiguous()
    state_indices = state_indices.reshape(-1)
    scratch_indices = scratch_indices.reshape(-1)
    output = v.new_empty(v.shape)

    block_key = triton.next_power_of_2(key_dim)
    if triton.cdiv(key_dim, block_key) != 1:
        raise ValueError("fused KDA verification requires one key-dimension block")
    block_value = min(triton.next_power_of_2(value_dim), 32)
    grid = (
        triton.cdiv(value_dim, block_value),
        batch_size * num_value_heads,
    )
    parent = query_start_loc if retrieve_parent_token is None else retrieve_parent_token
    parent_request_stride = 0 if retrieve_parent_token is None else parent.stride(0)
    parent_step_stride = 0 if retrieve_parent_token is None else parent.stride(1)

    _olmo_kda_target_verify_kernel[grid](
        a_log=a_log,
        raw_gate=raw_gate,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        raw_beta=raw_beta,
        output=output,
        state=state,
        state_indices=state_indices,
        query_start_loc=query_start_loc,
        scratch=scratch,
        scratch_indices=scratch_indices,
        retrieve_parent_token=parent,
        scale=key_dim**-0.5,
        total_tokens=total_tokens,
        stride_gate_token=raw_gate.stride(1),
        stride_gate_head=raw_gate.stride(2),
        stride_q_token=q.stride(1),
        stride_q_head=q.stride(2),
        stride_k_token=k.stride(1),
        stride_k_head=k.stride(2),
        stride_v_token=v.stride(1),
        stride_v_head=v.stride(2),
        stride_beta_token=raw_beta.stride(1),
        stride_beta_head=raw_beta.stride(2),
        stride_output_token=output.stride(1),
        stride_output_head=output.stride(2),
        stride_state_slot=state.stride(0),
        stride_state_head=state.stride(1),
        stride_state_value=state.stride(2),
        stride_scratch_slot=scratch.stride(0),
        stride_scratch_step=scratch.stride(1),
        stride_scratch_head=scratch.stride(2),
        stride_scratch_value=scratch.stride(3),
        stride_parent_request=parent_request_stride,
        stride_parent_step=parent_step_stride,
        num_query_heads=num_query_heads,
        num_key_heads=num_key_heads,
        num_value_heads=num_value_heads,
        key_dim=key_dim,
        value_dim=value_dim,
        block_key=block_key,
        block_value=block_value,
        beta_multiplier=2.0 if allow_neg_eigval else 1.0,
        has_parent_tree=retrieve_parent_token is not None,
        num_warps=1,
        num_stages=3,
    )
    return output
