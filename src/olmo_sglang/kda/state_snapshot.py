"""Inference-only FLA forward with a selected FP32 recurrent snapshot.

The ordinary FLA path stays in use for batches without tracked cache states.
Tracked batches retain the same gate, intra-chunk and output operations; only
the recurrence launcher writes an additional selected state before BF16 rounding.
"""

import torch
from fla.modules.l2norm import l2norm_fwd
from fla.ops.gla.chunk import chunk_gla_fwd_o_gk
from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra
from fla.ops.kda.gate import kda_gate_chunk_cumsum
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.constant import RCP_LN2

from olmo_sglang.kda.state_snapshot_kernel import state_forward


def chunk_kda(
    *,
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    initial_state,
    output_final_state,
    use_qk_l2norm_in_kernel,
    use_gate_in_kernel,
    return_intermediate_states,
    transpose_state_layout,
    cu_seqlens,
    track_state,
    track_chunk_idx,
):
    """Match the adapter's FLA call contract; beta is already Olmo-activated."""
    if not torch.is_inference_mode_enabled():
        raise RuntimeError("Tracked recurrent snapshots require inference mode")
    if not (output_final_state and use_qk_l2norm_in_kernel and use_gate_in_kernel):
        raise ValueError(
            "Snapshot forward requires the Olmo inference gate/state contract"
        )
    batch, _, _, key_width = q.shape
    sequences = batch if cu_seqlens is None else cu_seqlens.numel() - 1
    heads, value_width = v.shape[-2:]
    shape = (
        (sequences, heads, value_width, key_width)
        if transpose_state_layout
        else (sequences, heads, key_width, value_width)
    )
    # Metadata-only validation: no GPU reads or synchronization.
    if (
        track_state.dtype != torch.float32
        or tuple(track_state.shape) != shape
        or not track_state.is_contiguous()
    ):
        raise ValueError(
            "Track state must be contiguous FP32 with the recurrent-state layout"
        )
    if tuple(track_chunk_idx.shape) != (sequences,) or track_chunk_idx.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("Track chunk indices must be one integer per sequence")
    if track_state.device != q.device or track_chunk_idx.device != q.device:
        raise ValueError("Track buffers must be on the input device")
    # FLA's public autograd entry point applies input_guard before dispatch.
    # Match those contiguous input views without copying the output buffer.
    q, k, v, g, beta, A_log, dt_bias, cu_seqlens = (
        tensor.contiguous() if tensor is not None else None
        for tensor in (q, k, v, g, beta, A_log, dt_bias, cu_seqlens)
    )
    initial_state = initial_state.contiguous() if initial_state is not None else None
    track_chunk_idx = track_chunk_idx.contiguous()
    q, _ = l2norm_fwd(q)
    k, _ = l2norm_fwd(k)
    scale = key_width**-0.5
    chunk_size = 64
    chunks = (
        prepare_chunk_indices(cu_seqlens, chunk_size)
        if cu_seqlens is not None
        else None
    )
    gate = kda_gate_chunk_cumsum(
        g=g,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=RCP_LN2,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunks,
        lower_bound=None,
    )
    w, u, _, kg, aqk, _ = chunk_kda_fwd_intra(
        q=q,
        k=k,
        v=v,
        gk=gate,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunks,
        safe_gate=False,
        disable_recompute=False,
    )
    h, v_new, final = state_forward(
        track_state=track_state,
        track_chunk_idx=track_chunk_idx,
        k=kg,
        w=w,
        u=u,
        gk=gate,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunks,
        chunk_size=chunk_size,
        state_v_first=transpose_state_layout,
    )
    output = chunk_gla_fwd_o_gk(
        q=q,
        v=v_new,
        g=gate,
        A=aqk,
        h=h,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunks,
        state_v_first=transpose_state_layout,
    ).type_as(q)
    return (output, final, h) if return_intermediate_states else (output, final)
