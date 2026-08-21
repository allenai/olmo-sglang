from types import SimpleNamespace

import pytest
import torch
from sglang.srt.runtime_context import get_parallel

from olmo_sglang.kda_layer import Olmo3MoeKDAAttention


def _config():
    return SimpleNamespace(
        hidden_size=32,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        linear_norm_eps=1e-5,
    )


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_kda_layer_builds_tp2_local_projections_and_loads_recurrent_shards(tp_rank):
    with get_parallel().override(
        tp_size=2,
        tp_rank=tp_rank,
        attn_tp_size=2,
        attn_tp_rank=tp_rank,
    ):
        layer = Olmo3MoeKDAAttention(
            _config(), layer_id=0, quant_config=None, prefix="test"
        )
        layer.A_log.weight_loader(layer.A_log, torch.arange(4, dtype=torch.float32))
        layer.dt_bias.weight_loader(
            layer.dt_bias, torch.arange(32, dtype=torch.float32)
        )

    assert layer.num_k_heads == 2
    assert layer.num_v_heads == 2
    assert layer.key_dim == 16
    assert layer.value_dim == 32
    assert tuple(layer.qkv_proj.weight.shape) == (64, 32)
    assert tuple(layer.f_proj_2.weight.shape) == (16, 16)
    assert tuple(layer.beta_proj.weight.shape) == (2, 32)
    assert tuple(layer.g_proj_2.weight.shape) == (32, 16)
    assert tuple(layer.qkv_conv1d.weight.shape) == (64, 1, 4)
    assert tuple(layer.o_proj.weight.shape) == (32, 32)
    torch.testing.assert_close(
        layer.A_log, torch.arange(2 * tp_rank, 2 * (tp_rank + 1), dtype=torch.float32)
    )
    torch.testing.assert_close(
        layer.dt_bias,
        torch.arange(16 * tp_rank, 16 * (tp_rank + 1), dtype=torch.float32),
    )


def test_kda_layer_rejects_distinct_model_and_attention_tp_groups():
    with (
        get_parallel().override(
            tp_size=2,
            tp_rank=0,
            attn_tp_size=1,
            attn_tp_rank=0,
        ),
        pytest.raises(NotImplementedError, match="matching model and attention"),
    ):
        Olmo3MoeKDAAttention(_config(), layer_id=0, quant_config=None, prefix="test")
