"""Core attention retains SGLang request/cache isolation through chunks and decode."""

from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from olmo_sglang import core_attention


@pytest.mark.parametrize(
    "query_lengths,lengths", [([3, 5], [3, 5]), ([2, 3], [5, 7]), ([1, 1], [5, 7])]
)
def test_packed_requests_match_independent_causal_attention(
    monkeypatch, query_lengths, lengths
):
    torch.manual_seed(3)
    keys, values = [torch.randn(32, 2, 16) for _ in range(2)]
    request_map = torch.arange(32).reshape(2, 16)
    current_keys, current_values = [
        torch.randn(sum(query_lengths), 2, 16) for _ in range(2)
    ]
    queries = torch.randn(sum(query_lengths), 4, 16)
    locations = torch.cat(
        [
            request_map[r, n - q : n]
            for r, (q, n) in enumerate(zip(query_lengths, lengths, strict=True))
        ]
    )

    def set_buffer(layer, location, k, v):
        keys[locations] = k.reshape_as(current_keys)
        values[locations] = v.reshape_as(current_values)

    pool = SimpleNamespace(
        set_kv_buffer=set_buffer,
        get_key_buffer=lambda _: keys,
        get_value_buffer=lambda _: values,
    )
    backend = SimpleNamespace(
        token_to_kv_pool=pool,
        req_to_token_pool=SimpleNamespace(req_to_token=request_map),
    )
    monkeypatch.setattr(core_attention, "get_attn_backend", lambda: backend)
    decode = query_lengths == [1, 1]
    batch = SimpleNamespace(
        seq_lens=torch.tensor(lengths),
        req_pool_indices=torch.tensor([0, 1]),
        extend_seq_lens=torch.tensor(query_lengths),
        out_cache_loc=locations,
        forward_mode=SimpleNamespace(
            is_decode=lambda: decode, is_extend=lambda: not decode
        ),
    )
    layer = core_attention.CoreRadixAttention(4, 16, 0.25, num_kv_heads=2, layer_id=0)
    output = layer(
        queries.reshape(-1, 64),
        current_keys.reshape(-1, 32),
        current_values.reshape(-1, 32),
        batch,
    )
    start = 0
    for r, (query_length, length) in enumerate(
        zip(query_lengths, lengths, strict=True)
    ):
        q = queries[start : start + query_length].transpose(0, 1).unsqueeze(0)
        k = (
            keys[request_map[r, :length]]
            .transpose(0, 1)
            .unsqueeze(0)
            .repeat_interleave(2, 1)
        )
        v = (
            values[request_map[r, :length]]
            .transpose(0, 1)
            .unsqueeze(0)
            .repeat_interleave(2, 1)
        )
        mask = (
            torch.arange(length)[None, :]
            <= torch.arange(length - query_length, length)[:, None]
        )
        expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=0.25)
        torch.testing.assert_close(
            output[start : start + query_length],
            expected.squeeze(0).transpose(0, 1).reshape(-1, 64),
        )
        start += query_length
