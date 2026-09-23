"""Core SDPA arithmetic using SGLang's request mapping and persistent KV cache."""

import torch
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from torch.nn import functional as F


class CoreRadixAttention(RadixAttention):
    """Eager TP1 full attention; handles packed requests, cached decode and chunks."""

    def forward(self, q, k, v, forward_batch, save_kv_cache=True, **kwargs):
        if kwargs:
            raise ValueError("Unsupported Core attention forward options")
        backend = forward_batch.attn_backend
        pool = backend.token_to_kv_pool
        requests = backend.req_to_token_pool.req_to_token
        if save_kv_cache:
            pool.set_kv_buffer(
                self, KVWriteLoc(forward_batch.out_cache_loc, None), k, v
            )
        keys = pool.get_key_buffer(self.layer_id)
        values = pool.get_value_buffer(self.layer_id)
        query = q.reshape(-1, self.tp_q_head_num, self.qk_head_dim)
        output = torch.empty_like(query)
        lengths = forward_batch.seq_lens.tolist()
        request_ids = forward_batch.req_pool_indices.tolist()
        if forward_batch.forward_mode.is_decode():
            query_lengths = [1] * len(lengths)
        elif forward_batch.forward_mode.is_extend():
            query_lengths = forward_batch.extend_seq_lens.tolist()
        else:
            raise ValueError("Core attention supports ordinary extend/decode only")
        offset = 0
        for request, length, query_length in zip(
            request_ids, lengths, query_lengths, strict=True
        ):
            if query_length == 0:
                continue
            locations = requests[request, :length].long()
            key = keys[locations].transpose(0, 1).unsqueeze(0)
            value = values[locations].transpose(0, 1).unsqueeze(0)
            repeats = self.tp_q_head_num // self.tp_k_head_num
            key = key.repeat_interleave(repeats, dim=1)
            value = value.repeat_interleave(repeats, dim=1)
            q_value = query[offset : offset + query_length].transpose(0, 1).unsqueeze(0)
            mask = None
            causal = query_length == length
            if query_length != length and query_length != 1:
                q_pos = torch.arange(length - query_length, length, device=q.device)
                k_pos = torch.arange(length, device=q.device)
                mask = k_pos[None, :] <= q_pos[:, None]
            attended = F.scaled_dot_product_attention(
                q_value,
                key,
                value,
                attn_mask=mask,
                is_causal=causal,
                scale=self.scaling,
            )
            output[offset : offset + query_length] = attended.squeeze(0).transpose(0, 1)
            offset += query_length
        return output.reshape(q.shape[0], -1)
