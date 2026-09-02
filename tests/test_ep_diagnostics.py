import logging
from types import SimpleNamespace

import pytest
import torch
from olmo_sglang.models.olmo3_moe import (
    _first_sparse_layer_id,
    _local_expert_range,
    _log_ep_activity,
    _log_ep_parallelism,
)
from sglang.srt.runtime_context import get_parallel


def test_parallelism_log_reports_resolved_ep_topology(caplog) -> None:
    with (
        get_parallel().override(
            world_rank=1,
            tp_size=2,
            tp_rank=1,
            attn_tp_size=2,
            attn_tp_rank=1,
            attn_dp_size=1,
            moe_ep_size=2,
            moe_ep_rank=1,
            moe_tp_size=1,
            moe_tp_rank=0,
            moe_dp_size=1,
        ),
        caplog.at_level(logging.INFO),
    ):
        local_range = _log_ep_parallelism(emit=True, layer_id=1, num_experts=8)

    assert local_range == (4, 8)
    assert "world_rank=1 outer_tp=2" in caplog.text
    assert "attention_tp=2" in caplog.text
    assert "ep=2 ep_rank=1 moe_tp=1" in caplog.text
    assert "local_experts=[4,8)" in caplog.text


def test_ep_activity_log_counts_this_ranks_experts(caplog) -> None:
    topk_ids = torch.tensor([[0, 5], [6, 7]], dtype=torch.int32)
    with (
        get_parallel().override(world_rank=1, moe_ep_rank=1),
        caplog.at_level(logging.INFO),
    ):
        _log_ep_activity(
            topk_ids,
            layer_id=0,
            local_start=4,
            local_end=8,
        )

    assert "world_rank=1 ep_rank=1 layer=0" in caplog.text
    assert "routed_assignments=3 total_assignments=4" in caplog.text


def test_local_expert_range_requires_even_ownership() -> None:
    with pytest.raises(ValueError, match="divisible by inference EP size"):
        _local_expert_range(7, 2, 0)


def test_first_sparse_layer_skips_dense_prefix() -> None:
    config = SimpleNamespace(num_hidden_layers=4, dense_layers_indices=[0, 2])

    assert _first_sparse_layer_id(config) == 1


def test_first_sparse_layer_requires_sparse_layer() -> None:
    config = SimpleNamespace(num_hidden_layers=2, dense_layers_indices=[0, 1])

    with pytest.raises(ValueError, match="at least one sparse layer"):
        _first_sparse_layer_id(config)
