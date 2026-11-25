"""Olmo's packed routed-expert weight loading for SGLang."""

from sglang.srt.layers.moe.fused_moe_triton import FusedMoE


class OlmoFusedMoE(FusedMoE):
    """Shard a published gate/up pair as two column-parallel projections."""

    def _load_w13(
        self, expert_data, shard_dim, shard_id, loaded_weight, tp_rank, is_bias=False
    ):
        if (
            shard_id == "w13"
            and self.moe_tp_size > 1
            and self.quant_config is None
            and not self.use_presharded_weights
            and self.moe_runner_config.is_gated
            and not is_bias
            and loaded_weight.ndim == 3
        ):
            # Olmo publishes [E, 2H, D] with all gate rows before all up rows.
            # Split the source before SGLang shards or transposes either half.
            gate, up = loaded_weight.chunk(2, dim=1)
            super()._load_w13(expert_data, shard_dim, "w1", gate, tp_rank)
            super()._load_w13(expert_data, shard_dim, "w3", up, tp_rank)
            return
        super()._load_w13(
            expert_data, shard_dim, shard_id, loaded_weight, tp_rank, is_bias
        )
