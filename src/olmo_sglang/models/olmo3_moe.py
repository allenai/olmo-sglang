# SPDX-License-Identifier: Apache-2.0

"""Inference-only Olmo3MoE implementation for SGLang.

This module is loaded through ``SGLANG_EXTERNAL_MODEL_PACKAGE=olmo_sglang.models``
and supports full, sliding-window, and OLMo KDA attention layers.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterable
from functools import partial
from typing import Any

import torch
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix, make_layers
from torch import nn
from transformers import PretrainedConfig

from olmo_sglang.activations import native_silu_and_mul
from olmo_sglang.attention import (
    PerHeadRMSNorm,
    check_head_weights_loaded,
    load_head_weight,
    scale_attention_queries,
)
from olmo_sglang.config import validate_olmo3_moe_config
from olmo_sglang.kda.layer import Olmo3MoeKDAAttention
from olmo_sglang.routing import fp32_router_logits, olmo3_moe_topk

logger = logging.getLogger(__name__)


def _local_expert_range(
    num_experts: int, ep_size: int, ep_rank: int
) -> tuple[int, int]:
    if num_experts % ep_size:
        raise ValueError("n_routed_experts must be divisible by inference EP size")
    experts_per_rank = num_experts // ep_size
    start = ep_rank * experts_per_rank
    return start, start + experts_per_rank


def _first_sparse_layer_id(config: PretrainedConfig) -> int:
    dense_layer_ids = set(config.dense_layers_indices)
    for layer_id in range(config.num_hidden_layers):
        if layer_id not in dense_layer_ids:
            return layer_id
    raise ValueError("Olmo3Moe configuration must contain at least one sparse layer")


def _log_ep_parallelism(
    *, emit: bool, layer_id: int, num_experts: int
) -> tuple[int, int]:
    parallel = get_parallel()
    local_start, local_end = _local_expert_range(
        num_experts, parallel.moe_ep_size, parallel.moe_ep_rank
    )
    if emit:
        logger.info(
            "olmo_sglang_parallelism world_rank=%d outer_tp=%d "
            "outer_tp_rank=%d attention_tp=%d attention_tp_rank=%d "
            "attention_dp=%d ep=%d ep_rank=%d moe_tp=%d moe_tp_rank=%d "
            "moe_dp=%d local_experts=[%d,%d)",
            parallel.world_rank,
            parallel.tp_size,
            parallel.tp_rank,
            parallel.attn_tp_size,
            parallel.attn_tp_rank,
            parallel.attn_dp_size,
            parallel.moe_ep_size,
            parallel.moe_ep_rank,
            parallel.moe_tp_size,
            parallel.moe_tp_rank,
            parallel.moe_dp_size,
            local_start,
            local_end,
        )
    return local_start, local_end


def _log_ep_activity(
    topk_ids: torch.Tensor, *, layer_id: int, local_start: int, local_end: int
) -> None:
    local_assignments = torch.count_nonzero(
        (topk_ids >= local_start) & (topk_ids < local_end)
    ).item()
    parallel = get_parallel()
    logger.info(
        "olmo_sglang_ep_activity world_rank=%d ep_rank=%d layer=%d "
        "local_experts=[%d,%d) routed_assignments=%d total_assignments=%d",
        parallel.world_rank,
        parallel.moe_ep_rank,
        layer_id,
        local_start,
        local_end,
        local_assignments,
        topk_ids.numel(),
    )


class Olmo3MoeDenseMLP(nn.Module):
    """Dense SwiGLU block using SGLang tensor-parallel linears."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        return self.down_proj(native_silu_and_mul(gate_up))[0]


class Olmo3MoeSparseMLP(nn.Module):
    """Routed MoE with optional latent projections and a full-width shared expert."""

    def __init__(
        self,
        config: PretrainedConfig,
        *,
        layer_id: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.n_routed_experts
        self.router = nn.Module()
        self.router.gate = ReplicatedLinear(
            config.hidden_size,
            self.num_experts,
            bias=False,
            quant_config=None,
            prefix=add_prefix("router.gate", prefix),
        )

        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            layer_id=layer_id,
            renormalize=True,
            scoring_func="softmax",
            custom_routing_function=partial(
                olmo3_moe_topk,
                normalize_expert_weights=config.normalize_expert_weights,
                restore_weight_scale=config.restore_weight_scale,
                original_num_experts_per_tok=config.original_num_experts_per_tok,
            ),
        )

        latent_size = getattr(config, "latent_moe_dim", None)
        self.latent_down_proj: ReplicatedLinear | None
        self.latent_up_proj: ReplicatedLinear | None
        if latent_size is None:
            expert_hidden_size = config.hidden_size
            self.latent_down_proj = None
            self.latent_up_proj = None
        else:
            expert_hidden_size = latent_size
            self.latent_down_proj = ReplicatedLinear(
                config.hidden_size,
                latent_size,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("latent_down_proj", prefix),
            )
            self.latent_up_proj = ReplicatedLinear(
                latent_size,
                config.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("latent_up_proj", prefix),
            )

        self.experts = FusedMoE(
            num_experts=self.num_experts,
            hidden_size=expert_hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=True,
            quant_config=quant_config,
            layer_id=layer_id,
            prefix=add_prefix("experts", prefix),
        )
        self.layer_id = layer_id
        self._is_ep_diagnostic_layer = layer_id == _first_sparse_layer_id(config)
        self.local_expert_start, self.local_expert_end = _log_ep_parallelism(
            emit=self._is_ep_diagnostic_layer,
            layer_id=layer_id,
            num_experts=self.num_experts,
        )
        self._ep_activity_pending = self._is_ep_diagnostic_layer and os.environ.get(
            "OLMO_SGLANG_EP_DIAGNOSTICS", "0"
        ) in {"1", "true", "True"}

        shared_size = getattr(config, "shared_expert_intermediate_size", None)
        self.shared_expert = (
            Olmo3MoeDenseMLP(
                config.hidden_size,
                shared_size,
                quant_config=quant_config,
                prefix=add_prefix("shared_expert", prefix),
            )
            if shared_size is not None
            else None
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        # OLMo-core and the exported HF implementation evaluate the router
        # projection in FP32. Casting only the BF16 GEMM result is too late: it
        # can change top-k expert assignments and compounds over sparse layers.
        router_logits = fp32_router_logits(hidden_states, self.router.gate.weight)

        expert_inputs = hidden_states
        if self.latent_down_proj is not None:
            expert_inputs = self.latent_down_proj(expert_inputs)[0]

        topk_output = self.topk(expert_inputs, router_logits)
        routed_output = self.experts(expert_inputs, topk_output)
        if self._ep_activity_pending:
            _log_ep_activity(
                topk_output.topk_ids,
                layer_id=self.layer_id,
                local_start=self.local_expert_start,
                local_end=self.local_expert_end,
            )
            self._ep_activity_pending = False
        if self.latent_up_proj is not None:
            routed_output = self.latent_up_proj(routed_output)[0]

        if self.shared_expert is not None:
            routed_output = routed_output + self.shared_expert(hidden_states)
        return routed_output.reshape(original_shape)


class Olmo3MoeAttention(nn.Module):
    """Full or sliding-window attention with headwise QK norm and output gating."""

    def __init__(
        self,
        config: PretrainedConfig,
        *,
        layer_id: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        parallel = get_parallel()
        attn_tp_size = parallel.attn_tp_size

        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_heads % attn_tp_size != 0:
            raise ValueError(
                "num_attention_heads must be divisible by attention TP size"
            )
        if self.total_num_kv_heads >= attn_tp_size:
            if self.total_num_kv_heads % attn_tp_size != 0:
                raise ValueError(
                    "num_key_value_heads must be divisible by attention TP size"
                )
        elif attn_tp_size % self.total_num_kv_heads != 0:
            raise ValueError(
                "attention TP size must be divisible by num_key_value_heads"
            )

        self.num_heads = self.total_num_heads // attn_tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        self.head_dim = getattr(
            config,
            "head_dim",
            config.attention_hidden_size // config.num_attention_heads,
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            tp_rank=parallel.attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.per_head_gains = getattr(config, "qk_norm_per_head_gains", False)
        self.scalable_softmax = getattr(config, "scalable_softmax", False)
        self._hero_weights_checked = False
        if self.per_head_gains:
            self.q_norm = PerHeadRMSNorm(
                self.num_heads, self.head_dim, config.rms_norm_eps
            )
            self.k_norm = PerHeadRMSNorm(
                self.num_kv_heads, self.head_dim, config.rms_norm_eps
            )
            for norm, heads in (
                (self.q_norm, self.total_num_heads),
                (self.k_norm, self.total_num_kv_heads),
            ):
                norm.weight.weight_loader = partial(
                    load_head_weight,
                    total_heads=heads,
                    tp_size=attn_tp_size,
                    tp_rank=parallel.attn_tp_rank,
                )
        else:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        if self.scalable_softmax:
            self.ssmax_scale = nn.Parameter(torch.ones(self.num_heads))
            self.ssmax_scale.weight_loader = partial(
                load_head_weight,
                total_heads=self.total_num_heads,
                tp_size=attn_tp_size,
                tp_rank=parallel.attn_tp_rank,
            )
        else:
            self.register_parameter("ssmax_scale", None)

        gate_type = getattr(config, "attention_gate_type", None)
        self.g_proj = (
            ColumnParallelLinear(
                config.hidden_size,
                self.total_num_heads * self.head_dim,
                bias=False,
                quant_config=quant_config,
                tp_rank=parallel.attn_tp_rank,
                tp_size=attn_tp_size,
                prefix=add_prefix("g_proj", prefix),
            )
            if gate_type == "elementwise"
            else None
        )

        self.use_rope = getattr(config, "use_rope", True)
        if self.use_rope:
            rope_parameters = getattr(config, "rope_parameters", None) or {}
            rope_theta = getattr(config, "rope_theta", None) or rope_parameters.get(
                "rope_theta", 10000.0
            )
            self.rotary_emb = get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=config.max_position_embeddings,
                base=rope_theta,
                rope_scaling=rope_parameters or None,
                is_neox_style=True,
            )
        else:
            self.rotary_emb = None

        layer_type = config.layer_types[layer_id]
        sliding_window = (
            config.sliding_window - 1 if layer_type == "sliding_attention" else -1
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=sliding_window,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if not self._hero_weights_checked:
            required = {}
            if self.per_head_gains:
                required.update(q_norm=self.q_norm.weight, k_norm=self.k_norm.weight)
            if self.scalable_softmax:
                required["ssmax_scale"] = self.ssmax_scale
            check_head_weights_loaded(required)
            self._hero_weights_checked = True
        qkv = self.qkv_proj(hidden_states)[0]
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.per_head_gains:
            q = self.q_norm(q.reshape(-1, self.num_heads, self.head_dim)).reshape_as(q)
            k = self.k_norm(k.reshape(-1, self.num_kv_heads, self.head_dim)).reshape_as(
                k
            )
        else:
            q = self.q_norm(q.reshape(-1, self.head_dim)).reshape_as(q)
            k = self.k_norm(k.reshape(-1, self.head_dim)).reshape_as(k)
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)

        if self.scalable_softmax:
            q = scale_attention_queries(
                q.reshape(-1, self.num_heads, self.head_dim),
                positions,
                self.ssmax_scale,
            ).reshape_as(q)
        attention_output = self.attn(q, k, v, forward_batch)
        if self.g_proj is not None:
            gate = self.g_proj(hidden_states)[0]
            attention_output = attention_output * torch.sigmoid(gate.float()).to(
                attention_output.dtype
            )
        return self.o_proj(attention_output)[0]


class Olmo3MoeDecoderLayer(nn.Module):
    """Olmo peri-LN decoder block."""

    def __init__(
        self,
        config: PretrainedConfig,
        *,
        layer_id: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        attention_class = (
            Olmo3MoeKDAAttention
            if config.layer_types[layer_id] == "linear_attention"
            else Olmo3MoeAttention
        )
        self.self_attn = attention_class(
            config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = (
            Olmo3MoeDenseMLP(
                config.hidden_size,
                config.dense_mlp_intermediate_size,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
            if layer_id in config.dense_layers_indices
            else Olmo3MoeSparseMLP(
                config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        )
        self.pre_attention_layernorm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.use_peri_ln
            else None
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.use_peri_ln
            else None
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        residual = hidden_states
        attention_inputs = (
            self.pre_attention_layernorm(hidden_states)
            if self.pre_attention_layernorm is not None
            else hidden_states
        )
        attention_output = self.self_attn(positions, attention_inputs, forward_batch)
        hidden_states = residual + self.post_attention_layernorm(attention_output)

        residual = hidden_states
        mlp_inputs = (
            self.pre_feedforward_layernorm(hidden_states)
            if self.pre_feedforward_layernorm is not None
            else hidden_states
        )
        mlp_output = self.mlp(mlp_inputs)
        return residual + self.post_feedforward_layernorm(mlp_output)


class Olmo3MoeModel(nn.Module):
    """Olmo3MoE decoder backbone."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        validate_olmo3_moe_config(config)
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.embed_norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if getattr(config, "embed_norm", False)
            else None
        )
        self.embed_scale = float(getattr(config, "embed_scale", 1.0))
        self.layers = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: Olmo3MoeDecoderLayer(
                config, layer_id=idx, quant_config=quant_config, prefix=prefix
            ),
            prefix=add_prefix("layers", prefix),
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = (
            self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        )
        hidden_states = hidden_states * self.embed_scale
        if self.embed_norm is not None:
            hidden_states = self.embed_norm(hidden_states)
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states, forward_batch)
        return self.norm(hidden_states)


class Olmo3MoeForCausalLM(nn.Module):
    """SGLang entry point for ``Olmo3MoeForCausalLM`` checkpoints."""

    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = Olmo3MoeModel(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )
        self.logits_processor = LogitsProcessor(config)

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load HF-layout weights into SGLang fused and tensor-parallel parameters.

        Name resolution is constant-time per tensor and memoized across calls.
        Online weight updates replay every checkpoint name on each publication,
        and a per-expert scan over ``3 * n_routed_experts`` patterns costs about
        a second per publication at 512 experts.
        """
        resolver = self._weight_targets
        if resolver is None:
            resolver = self._weight_targets = WeightTargets(
                dict(self.named_parameters()), self.config.n_routed_experts
            )
        loaded_params: set[str] = set()
        for name, loaded_weight, *rest in weights:
            loader_kwargs = rest[0] if rest else {}
            target = resolver.resolve(name)
            if target is None:
                continue
            kind, mapped_name, shard_id, expert_id = target
            param = resolver.params[mapped_name]
            if kind == "stacked":
                param.weight_loader(param, loaded_weight, shard_id)
            elif kind == "expert":
                param.weight_loader(
                    param,
                    loaded_weight,
                    mapped_name,
                    shard_id=shard_id,
                    expert_id=expert_id,
                )
            else:
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, **loader_kwargs)
            loaded_params.add(mapped_name)
        return loaded_params

    _weight_targets: "WeightTargets | None" = None


STACKED_PARAMS_MAPPING = (
    (".qkv_proj", ".q_proj", "q"),
    (".qkv_proj", ".k_proj", "k"),
    (".qkv_proj", ".v_proj", "v"),
    (".qkv_conv1d", ".q_conv1d", 0),
    (".qkv_conv1d", ".k_conv1d", 1),
    (".qkv_conv1d", ".v_conv1d", 2),
    (".gate_up_proj", ".gate_proj", 0),
    (".gate_up_proj", ".up_proj", 1),
)
_EXPERT_SHARDS = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}
_EXPERT_WEIGHT = re.compile(r"\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.")


class WeightTargets:
    """Resolve HF checkpoint names to SGLang parameters without scanning mappings.

    ``resolve`` returns ``(kind, parameter_name, shard_id, expert_id)`` or
    ``None`` for names that are skipped. Results are memoized per checkpoint
    name; unknown names raise ``KeyError`` exactly as the scanning loader did.
    """

    def __init__(self, params: dict[str, torch.nn.Parameter], num_experts: int):
        self.params = params
        self.num_experts = num_experts
        self._cache: dict[str, tuple[str, str, Any, int | None] | None] = {}

    def resolve(self, name: str) -> tuple[str, str, Any, int | None] | None:
        try:
            return self._cache[name]
        except KeyError:
            target = self._cache[name] = self._resolve(name)
            return target

    def _resolve(self, name: str) -> tuple[str, str, Any, int | None] | None:
        if "rotary_emb.inv_freq" in name:
            return None
        name = name.replace(".linear_attn.", ".self_attn.")
        if ".mlp.experts." not in name:
            for param_name, weight_name, shard_id in STACKED_PARAMS_MAPPING:
                if weight_name not in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                if mapped_name in self.params:
                    return ("stacked", mapped_name, shard_id, None)
        match = _EXPERT_WEIGHT.search(name)
        if match is not None and int(match.group(1)) < self.num_experts:
            expert_id, projection = int(match.group(1)), match.group(2)
            prefix = "experts.w2_" if projection == "down_proj" else "experts.w13_"
            mapped_name = name.replace(f"experts.{expert_id}.{projection}.", prefix)
            if mapped_name in self.params:
                return ("expert", mapped_name, _EXPERT_SHARDS[projection], expert_id)
        if name in self.params:
            return ("direct", name, None, None)
        if name.endswith(".bias"):
            return None
        raise KeyError(f"No SGLang parameter for checkpoint weight {name}")


EntryClass = Olmo3MoeForCausalLM
