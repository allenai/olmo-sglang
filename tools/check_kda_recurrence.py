# SPDX-License-Identifier: Apache-2.0

"""Exercise OLMo KDA prefill/decode cache parity against FLA 0.5.2."""

from __future__ import annotations

import logging

import torch

from olmo_sglang.kda.backend import OlmoFLAKDAKernel

LOGGER = logging.getLogger(__name__)


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    allow_neg_eigval: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    from fla.ops.kda import chunk_kda

    output, final_state = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=gate,
        beta=beta.float().sigmoid() * (2.0 if allow_neg_eigval else 1.0),
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=torch.zeros(
            1,
            v.shape[2],
            v.shape[-1],
            k.shape[-1],
            dtype=torch.float32,
            device=q.device,
        ),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        transpose_state_layout=True,
        cu_seqlens=torch.tensor([0, q.shape[1]], dtype=torch.int32, device=q.device),
    )
    return output, final_state


def main() -> None:
    """Run the cached recurrence comparison on CUDA."""

    if not torch.cuda.is_available():
        raise RuntimeError("This recurrence check requires a CUDA GPU")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.manual_seed(2026)
    device = torch.device("cuda")
    dtype = torch.float16
    prompt_length = 4
    total_length = prompt_length + 1
    num_heads = 2
    key_dim = 8
    value_dim = 16

    q = torch.randn(1, total_length, num_heads, key_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn(1, total_length, num_heads, value_dim, device=device, dtype=dtype)
    gate = torch.randn_like(q)
    beta = torch.randn(1, total_length, num_heads, device=device, dtype=dtype)
    A_log = torch.linspace(0.0, 1.0, num_heads, device=device)
    dt_bias = torch.zeros(num_heads * key_dim, device=device)
    query_start_loc = torch.tensor([0, prompt_length], dtype=torch.int32, device=device)
    decode_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=device)
    cache_indices = torch.tensor([0], dtype=torch.int32, device=device)
    state_pool = torch.zeros(
        1, num_heads, value_dim, key_dim, dtype=torch.float32, device=device
    )

    kernel = OlmoFLAKDAKernel(allow_neg_eigval=True)
    prefill = kernel.extend(
        q[:, :prompt_length],
        k[:, :prompt_length],
        v[:, :prompt_length],
        gate[:, :prompt_length],
        beta[:, :prompt_length],
        A_log=A_log,
        dt_bias=dt_bias,
        ssm_states=state_pool,
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
    )
    decode = kernel.decode(
        q[:, prompt_length:],
        k[:, prompt_length:],
        v[:, prompt_length:],
        gate[:, prompt_length:],
        beta[:, prompt_length:],
        A_log=A_log,
        dt_bias=dt_bias,
        ssm_states=state_pool,
        cache_indices=cache_indices,
        query_start_loc=decode_start_loc,
    )
    cached_output = torch.cat((prefill, decode), dim=1)

    reference_output, reference_state = _reference(
        q, k, v, gate, beta, A_log, dt_bias, allow_neg_eigval=True
    )
    ordinary_output, _ = _reference(
        q, k, v, gate, beta, A_log, dt_bias, allow_neg_eigval=False
    )
    output_diff = (cached_output - reference_output).abs().max().item()
    state_diff = (state_pool - reference_state).abs().max().item()
    semantic_delta = (reference_output - ordinary_output).abs().max().item()
    torch.testing.assert_close(cached_output, reference_output, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(state_pool, reference_state, atol=2e-3, rtol=2e-3)
    if semantic_delta == 0.0:
        raise AssertionError(
            "negative-eigenvalue KDA unexpectedly matched ordinary KDA"
        )

    LOGGER.info(
        "OLMO_KDA_RECURRENCE_PASS "
        f"cached_max_diff={output_diff:.6g} "
        f"state_max_diff={state_diff:.6g} "
        f"ordinary_kda_semantic_delta={semantic_delta:.6g}"
    )


if __name__ == "__main__":
    main()
