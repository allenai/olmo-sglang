# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference-only Olmo3MoE implementation for SGLang.

This module is loaded through ``SGLANG_EXTERNAL_MODEL_PACKAGE=olmo_sglang.models``.
It deliberately starts with the full/sliding-attention Olmo3MoE reference
architecture. The production KDA variant shares the surrounding model, peri-LN,
latent-MoE, and weight-loading contracts implemented here, but its recurrent
attention layer remains a separate follow-up.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import partial

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
from olmo_sglang.config import validate_olmo3_moe_config
from olmo_sglang.kda_layer import Olmo3MoeKDAAttention
from olmo_sglang.routing import fp32_router_logits, olmo3_moe_topk


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
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

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
        qkv = self.qkv_proj(hidden_states)[0]
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.reshape(-1, self.head_dim)).reshape_as(q)
        k = self.k_norm(k.reshape(-1, self.head_dim)).reshape_as(k)
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)

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
                config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
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
        """Load HF-layout weights into SGLang fused and tensor-parallel parameters."""

        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".qkv_conv1d", ".q_conv1d", 0),
            (".qkv_conv1d", ".k_conv1d", 1),
            (".qkv_conv1d", ".v_conv1d", 2),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
        )

        params = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight, *rest in weights:
            loader_kwargs = rest[0] if rest else {}
            if "rotary_emb.inv_freq" in name:
                continue

            name = name.replace(".linear_attn.", ".self_attn.")
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if ".mlp.experts." in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                if mapped_name not in params:
                    continue
                param = params[mapped_name]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped_name)
                break
            else:
                for (
                    param_name,
                    weight_name,
                    expert_id,
                    shard_id,
                ) in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    mapped_name = name.replace(weight_name, param_name)
                    if mapped_name not in params:
                        continue
                    param = params[mapped_name]
                    param.weight_loader(
                        param,
                        loaded_weight,
                        mapped_name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    loaded_params.add(mapped_name)
                    break
                else:
                    if name.endswith(".bias") and name not in params:
                        continue
                    if name not in params:
                        raise KeyError(
                            f"No SGLang parameter for checkpoint weight {name}"
                        )
                    param = params[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, **loader_kwargs)
                    loaded_params.add(name)
        return loaded_params


EntryClass = Olmo3MoeForCausalLM
