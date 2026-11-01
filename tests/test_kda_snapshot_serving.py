"""Exercise SGLang's actual track planning, prefill dispatch and cache commit."""

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.layers.attention.linear import kda_backend as serving

from olmo_sglang.kda import backend, state_snapshot


@pytest.mark.parametrize("core_layout", [False, True])
def test_serving_commits_fp32_snapshots_without_bf16_history_overwrite(
    monkeypatch, core_layout
):
    # Isolate convolution and arithmetic; keep real track planning, SGLang
    # forward_extend, dispatcher, Olmo adapter and final cache-slot writes.
    monkeypatch.setattr(serving, "causal_conv1d_fn", lambda x, *args, **kwargs: x)
    calls = []

    def snapshot(**kw):
        calls.append(kw["track_chunk_idx"].tolist())
        kw["track_state"].fill_(1.00390625)  # Deliberately not representable in BF16.
        return (
            torch.zeros_like(kw["v"]),
            kw["initial_state"] + 2,
            torch.full((1, 7, 1, 3, 2), 99, dtype=torch.bfloat16),
        )

    monkeypatch.setattr(state_snapshot, "chunk_kda", snapshot)
    kernel = object.__new__(backend.OlmoPackedKDAKernel)
    kernel.allow_neg_eigval = True
    kernel.core_compat = core_layout
    dispatcher = object.__new__(serving.KDAKernelDispatcher)
    dispatcher.extend_kernel = kernel
    engine = object.__new__(backend.OlmoKDAAttnBackend)
    engine.kernel_dispatcher = dispatcher
    engine.device = "cpu"
    engine._mamba_chunk_size = 64
    engine.accept_lens_pool = None
    states = torch.zeros(6, 1, 3, 2)
    states[:3].fill_(1)
    cache = SimpleNamespace(temporal=states, conv=[torch.zeros(6, 3, 7)])
    engine.req_to_token_pool = SimpleNamespace(mamba2_layer_cache=lambda _: cache)
    indices = torch.tensor([0, 1, 2])
    lengths = [129, 128, 65]
    batch = SimpleNamespace(
        batch_size=3,
        forward_mode=SimpleNamespace(
            is_target_verify=lambda: False,
            is_extend=lambda: True,
            is_draft_extend_v2=lambda: False,
        ),
        extend_prefix_lens=torch.zeros(3, dtype=torch.int64),
        extend_prefix_lens_cpu=[0, 0, 0],
        extend_seq_lens=torch.tensor(lengths),
        extend_seq_lens_cpu=lengths,
        mamba_track_mask=torch.tensor([True, True, False]),
        mamba_prefill_track_mask_cpu=[True, True, False],
        mamba_track_indices=torch.tensor([3, 4, 5]),
        mamba_track_seqlens=torch.tensor(lengths),
        mamba_track_seqlens_cpu=lengths,
    )
    names = (
        "track_chunk_idx",
        "track_ssm_h_src",
        "track_ssm_h_dst",
        "track_ssm_h_batch_src",
        "track_ssm_final_src",
        "track_ssm_final_dst",
        "track_ssm_seq_idx",
        "track_ssm_end_locs",
        "track_ssm_recompute_dst",
    )
    tracking = dict(
        zip(names, engine._init_track_ssm_indices(indices, batch), strict=True)
    )
    assert tracking["track_chunk_idx"].tolist() == [2, -1, -1]
    engine.forward_metadata = SimpleNamespace(
        **tracking,
        query_start_loc=torch.tensor([0, 129, 257, 322]),
        mamba_cache_indices=indices,
        logical_num_tokens=322,
        has_mamba_track_mask=True,
        conv_states_mask_indices=torch.tensor([3, 4]),
        track_conv_indices=torch.tensor([[125, 126, 127], [254, 255, 256]]),
    )
    layer = SimpleNamespace(
        layer_id=0,
        conv_weights=None,
        bias=None,
        q_dim=2,
        k_dim=2,
        v_dim=3,
        head_q_dim=2,
        head_k_dim=2,
        head_v_dim=3,
        lower_bound=None,
        A_log=torch.zeros(1),
        dt_bias=torch.zeros(2),
    )
    engine.forward_extend(
        layer,
        batch,
        torch.zeros(322, 7),
        torch.zeros(1, 322, 2),
        torch.zeros(1, 322, 1),
    )
    assert calls == ([[2], [-1], [-1]] if core_layout else [[2, -1, -1]])
    torch.testing.assert_close(
        states[3], torch.full_like(states[3], 1.00390625), atol=0, rtol=0
    )
    torch.testing.assert_close(states[4], states[1], atol=0, rtol=0)
    assert torch.all(states[4] == 3)  # Aligned row uses its final state.
    assert torch.count_nonzero(states[5]) == 0  # Untracked destination untouched.
