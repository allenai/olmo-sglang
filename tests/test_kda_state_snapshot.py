"""Selected cache states must retain precision lost by BF16 chunk histories."""

import pytest
import torch
from fla.ops.kda import chunk_kda

from olmo_sglang.kda import backend, state_snapshot


@pytest.mark.parametrize("core_layout", [False, True])
def test_adapter_forwards_selected_track_buffers(monkeypatch, core_layout):
    observed = []

    def tracked(**kwargs):
        assert torch.is_inference_mode_enabled()
        observed.append(kwargs)
        kwargs["track_state"].fill_(1.00390625)
        return (
            torch.zeros_like(kwargs["v"]),
            kwargs["initial_state"] + 1,
            torch.zeros(1, 1, *kwargs["initial_state"].shape[1:]),
        )

    monkeypatch.setattr(state_snapshot, "chunk_kda", tracked)
    kernel = object.__new__(backend.OlmoFLAKDAKernel)
    kernel.allow_neg_eigval = True
    kernel.core_compat = core_layout
    q = torch.zeros(1, 2, 1, 2)
    states = torch.ones(1, 1, 3, 2)
    track = torch.zeros_like(states)
    indices = torch.tensor([0], dtype=torch.int32)
    kernel.extend(
        q,
        q,
        torch.zeros(1, 2, 1, 3),
        q,
        torch.zeros(1, 2, 1),
        A_log=torch.zeros(1),
        dt_bias=torch.zeros(2),
        ssm_states=states,
        cache_indices=indices,
        query_start_loc=torch.tensor([0, 2]),
        return_intermediate_states=True,
        track_state=track,
        track_chunk_idx=indices,
    )
    assert len(observed) == 1
    assert observed[0]["track_chunk_idx"].tolist() == [0]
    torch.testing.assert_close(
        track, torch.full_like(track, 1.00390625), atol=0, rtol=0
    )
    torch.testing.assert_close(states, torch.full_like(states, 2), atol=0, rtol=0)
    assert backend.OlmoPackedKDAKernel.supports_track_state_snapshot


@pytest.mark.parametrize("state_v_first", [False, True])
@pytest.mark.parametrize("key_width,value_width", [(64, 96), (128, 64)])
@pytest.mark.parametrize("variable", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA recurrence validation")
def test_gpu_snapshot_matches_fp32_prefix_and_preserves_forward(
    state_v_first, key_width, value_width, variable
):
    backend.require_fla_0_5_2()
    torch.manual_seed(731)
    lengths = [129, 193, 70, 65] if variable else [193] * 4
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    batch, tokens = (1, sum(lengths)) if variable else (4, lengths[0])
    q = torch.randn(batch, tokens, 2, key_width, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(batch, tokens, 4, value_width, device="cuda", dtype=torch.bfloat16)
    gate = (
        torch.randn(batch, tokens, 4, key_width, device="cuda", dtype=torch.bfloat16)
        - 2
    )
    beta = 2 * torch.randn(batch, tokens, 4, device="cuda").sigmoid()
    shape = (
        (4, 4, value_width, key_width)
        if state_v_first
        else (4, 4, key_width, value_width)
    )
    initial = torch.randn(shape, device="cuda", dtype=torch.float32) * 0.1
    a_log = torch.zeros(4, device="cuda")
    bias = torch.zeros(4 * key_width, device="cuda")
    cu = torch.tensor(starts, dtype=torch.int32, device="cuda") if variable else None
    args = dict(
        q=q,
        k=k,
        v=v,
        g=gate,
        beta=beta,
        A_log=a_log,
        dt_bias=bias,
        initial_state=initial,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        return_intermediate_states=True,
        transpose_state_layout=state_v_first,
        cu_seqlens=cu,
    )
    track = torch.full_like(initial, 12345)
    selected = [1, 2, -1, 0]
    with torch.inference_mode():
        expected = chunk_kda(**args)
        actual = state_snapshot.chunk_kda(
            **args,
            track_state=track,
            track_chunk_idx=torch.tensor(selected, dtype=torch.int32, device="cuda"),
        )
        torch.testing.assert_close(actual[0], expected[0], atol=0.008, rtol=0.008)
        torch.testing.assert_close(actual[1], expected[1], atol=2e-5, rtol=2e-5)
        assert actual[2].dtype == expected[2].dtype == torch.bfloat16
        torch.testing.assert_close(actual[2], expected[2], atol=0.008, rtol=0.008)
        assert torch.all(track[2] == 12345)
        for index in (0, 1, 3):
            end = selected[index] * 64
            if end == 0:
                reference = initial[index]
            else:
                prefix = dict(args)
                for name in ("q", "k", "v", "g", "beta"):
                    tensor = args[name]
                    prefix[name] = (
                        tensor[:, starts[index] : starts[index] + end]
                        if variable
                        else tensor[index : index + 1, :end]
                    ).contiguous()
                prefix.update(initial_state=initial[index : index + 1], cu_seqlens=None)
                reference = chunk_kda(**prefix)[1][0]
            torch.testing.assert_close(track[index], reference, atol=2e-5, rtol=2e-5)
        # Ensure the fixture would catch a BF16 history copied into an FP32 buffer.
        assert (track[0] - track[0].bfloat16().float()).abs().max().item() > 2e-5
