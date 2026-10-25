import pytest
import torch

from olmo_sglang.kda.backend import OlmoPackedKDAKernel
from olmo_sglang.kda.packed_decode import olmo_packed_kda_decode


def _verify(
    kernel,
    *,
    q,
    k,
    v,
    raw_gate,
    raw_beta,
    a_log,
    dt_bias,
    state,
    state_indices,
    query_start_loc,
    scratch,
    scratch_indices,
    parents,
):
    return kernel.target_verify(
        A_log=a_log,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        a=raw_gate,
        b=raw_beta,
        ssm_states=state,
        cache_indices=state_indices,
        query_start_loc=query_start_loc,
        intermediate_states_buffer=scratch,
        intermediate_state_indices=scratch_indices,
        cache_steps=scratch.shape[1],
        retrieve_parent_token=parents,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("allow_neg_eigval", [False, True])
def test_fused_target_verify_matches_eager_ragged_tree(allow_neg_eigval):
    torch.manual_seed(23)
    token_count = 5
    query_heads, key_heads, value_heads = 1, 1, 2
    key_dim, value_dim = 8, 12
    dtype = torch.bfloat16
    q = torch.randn(1, token_count, query_heads, key_dim, dtype=dtype)
    k = torch.randn(1, token_count, key_heads, key_dim, dtype=dtype)
    v = torch.randn(1, token_count, value_heads, value_dim, dtype=dtype)
    raw_gate = torch.randn(token_count, value_heads, key_dim, dtype=dtype)
    raw_beta = torch.randn(1, token_count, value_heads, dtype=dtype)
    a_log = torch.randn(value_heads)
    dt_bias = torch.randn(value_heads, key_dim)
    state = torch.randn(4, value_heads, value_dim, key_dim)
    state_before = state.clone()
    state_indices = torch.tensor([2, -1], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 3, 5], dtype=torch.int32)
    scratch_indices = torch.tensor([1, 3], dtype=torch.int32)
    scratch = torch.full(
        (4, 3, value_heads, value_dim, key_dim),
        float("nan"),
    )
    parents = torch.tensor([[0, 0, 1], [0, 0, 0]], dtype=torch.int32)

    eager_kernel = object.__new__(OlmoPackedKDAKernel)
    eager_kernel.allow_neg_eigval = allow_neg_eigval
    expected = _verify(
        eager_kernel,
        q=q,
        k=k,
        v=v,
        raw_gate=raw_gate,
        raw_beta=raw_beta,
        a_log=a_log,
        dt_bias=dt_bias,
        state=state,
        state_indices=state_indices,
        query_start_loc=query_start_loc,
        scratch=scratch,
        scratch_indices=scratch_indices,
        parents=parents,
    )

    fused_kernel = object.__new__(OlmoPackedKDAKernel)
    fused_kernel.allow_neg_eigval = allow_neg_eigval
    cuda_state = state_before.cuda()
    cuda_scratch = torch.full_like(scratch, float("nan"), device="cuda")
    actual = _verify(
        fused_kernel,
        q=q.cuda(),
        k=k.cuda(),
        v=v.cuda(),
        raw_gate=raw_gate.cuda(),
        raw_beta=raw_beta.cuda(),
        a_log=a_log.cuda(),
        dt_bias=dt_bias.cuda(),
        state=cuda_state,
        state_indices=state_indices.cuda(),
        query_start_loc=query_start_loc.cuda(),
        scratch=cuda_scratch,
        scratch_indices=scratch_indices.cuda(),
        parents=parents.cuda(),
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual.cpu(), expected, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(cuda_state.cpu(), state_before)
    for request, steps in enumerate((3, 2)):
        slot = scratch_indices[request]
        torch.testing.assert_close(
            cuda_scratch[slot, :steps].cpu(),
            scratch[slot, :steps],
            atol=3e-3,
            rtol=3e-3,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_target_verify_cuda_graph_replays_new_inputs():
    torch.manual_seed(29)
    device = torch.device("cuda")
    token_count, heads, key_dim, value_dim = 4, 2, 8, 16
    dtype = torch.bfloat16
    kernel = object.__new__(OlmoPackedKDAKernel)
    kernel.allow_neg_eigval = True

    q = torch.randn(1, token_count, heads, key_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn(1, token_count, heads, value_dim, device=device, dtype=dtype)
    raw_gate = torch.randn(token_count, heads, key_dim, device=device, dtype=dtype)
    raw_beta = torch.randn(1, token_count, heads, device=device, dtype=dtype)
    a_log = torch.randn(heads, device=device)
    dt_bias = torch.randn(heads, key_dim, device=device)
    state = torch.randn(1, heads, value_dim, key_dim, device=device)
    state_indices = torch.zeros(1, device=device, dtype=torch.int32)
    query_start_loc = torch.tensor([0, token_count], device=device, dtype=torch.int32)
    scratch_indices = torch.zeros(1, device=device, dtype=torch.int32)
    scratch = torch.empty(1, token_count, heads, value_dim, key_dim, device=device)
    parents = torch.tensor([[0, 0, 0, 1]], device=device, dtype=torch.int32)

    _verify(
        kernel,
        q=q,
        k=k,
        v=v,
        raw_gate=raw_gate,
        raw_beta=raw_beta,
        a_log=a_log,
        dt_bias=dt_bias,
        state=state,
        state_indices=state_indices,
        query_start_loc=query_start_loc,
        scratch=scratch,
        scratch_indices=scratch_indices,
        parents=parents,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = _verify(
            kernel,
            q=q,
            k=k,
            v=v,
            raw_gate=raw_gate,
            raw_beta=raw_beta,
            a_log=a_log,
            dt_bias=dt_bias,
            state=state,
            state_indices=state_indices,
            query_start_loc=query_start_loc,
            scratch=scratch,
            scratch_indices=scratch_indices,
            parents=parents,
        )

    q.copy_(torch.randn_like(q))
    k.copy_(torch.randn_like(k))
    v.copy_(torch.randn_like(v))
    raw_gate.copy_(torch.randn_like(raw_gate))
    raw_beta.copy_(torch.randn_like(raw_beta))
    state.copy_(torch.randn_like(state))
    scratch.fill_(float("nan"))
    expected_state = state.clone()
    expected_scratch = torch.empty_like(scratch)
    expected = _verify(
        kernel,
        q=q,
        k=k,
        v=v,
        raw_gate=raw_gate,
        raw_beta=raw_beta,
        a_log=a_log,
        dt_bias=dt_bias,
        state=expected_state,
        state_indices=state_indices,
        query_start_loc=query_start_loc,
        scratch=expected_scratch,
        scratch_indices=scratch_indices,
        parents=parents,
    )
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(captured_output, expected)
    torch.testing.assert_close(scratch, expected_scratch)
    torch.testing.assert_close(state, expected_state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("tree", [False, True])
@pytest.mark.parametrize("allow_neg_eigval", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("key_dim,value_dim", [(8, 12), (128, 128)])
def test_target_verify_matches_repeated_packed_decode(
    tree, allow_neg_eigval, dtype, key_dim, value_dim
):
    torch.manual_seed(41)
    device = "cuda"
    steps, query_heads, value_heads = 5, 2, 4
    q = torch.randn(1, steps, query_heads, key_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn(1, steps, value_heads, value_dim, device=device, dtype=dtype)
    raw_gate = torch.randn(steps, value_heads, key_dim, device=device, dtype=dtype)
    raw_beta = torch.randn(1, steps, value_heads, device=device, dtype=dtype)
    a_log = torch.randn(value_heads, device=device)
    dt_bias = torch.randn(value_heads, key_dim, device=device)
    state = torch.randn(1, value_heads, value_dim, key_dim, device=device)
    original_state = state.clone()
    indices = torch.zeros(1, device=device, dtype=torch.int32)
    scratch = torch.empty(1, steps, value_heads, value_dim, key_dim, device=device)
    parent_steps = [0, 0, 0, 1, 2] if tree else [0, 0, 1, 2, 3]
    parents = (
        torch.tensor([parent_steps], device=device, dtype=torch.int32) if tree else None
    )
    kernel = object.__new__(OlmoPackedKDAKernel)
    kernel.allow_neg_eigval = allow_neg_eigval
    actual = _verify(
        kernel,
        q=q,
        k=k,
        v=v,
        raw_gate=raw_gate,
        raw_beta=raw_beta,
        a_log=a_log,
        dt_bias=dt_bias,
        state=state,
        state_indices=indices,
        query_start_loc=torch.tensor([0, steps], device=device, dtype=torch.int32),
        scratch=scratch,
        scratch_indices=indices,
        parents=parents,
    )

    def decode(token, recurrent_state):
        mixed = torch.cat([x[0, token].reshape(1, -1) for x in (q, k, v)], dim=-1)
        return olmo_packed_kda_decode(
            mixed,
            raw_gate[token : token + 1],
            raw_beta[:, token],
            a_log=a_log,
            dt_bias=dt_bias,
            scale=key_dim**-0.5,
            state=recurrent_state,
            state_indices=indices,
            num_value_heads=value_heads,
            value_dim=value_dim,
            allow_neg_eigval=allow_neg_eigval,
        )

    expected_states, expected_outputs = [], []
    for step in range(steps):
        parent = original_state if step == 0 else expected_states[parent_steps[step]]
        current = parent.clone()
        expected_outputs.append(decode(step, current))
        expected_states.append(current)

    # State comparison is in FP32 and tight enough to expose missing BF16 Q/K
    # rounding; output-only BF16 tolerances can conceal the recurrence mismatch.
    torch.testing.assert_close(state, original_state, atol=0, rtol=0)
    torch.testing.assert_close(
        scratch[0], torch.cat(expected_states), atol=2e-6, rtol=2e-5
    )
    torch.testing.assert_close(
        actual, torch.cat(expected_outputs, dim=1), atol=2e-6, rtol=2e-5
    )
    # Commit a non-final accepted tree node, then continue ordinary decoding.
    committed = scratch[:, 3].clone()
    expected_committed = expected_states[3].clone()
    torch.testing.assert_close(
        decode(4, committed), decode(4, expected_committed), atol=2e-6, rtol=2e-5
    )
    torch.testing.assert_close(committed, expected_committed, atol=2e-6, rtol=2e-5)
