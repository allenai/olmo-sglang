# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OLMo KDA model layer built on SGLang's linear-attention scheduler."""

from __future__ import annotations

import torch
from sglang.kernels.ops.attention.fla.fused_norm_gate import FusedRMSNormGated
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix
from torch import nn
from transformers import PretrainedConfig


class Olmo3MoeKDAAttention(nn.Module):
    """KDA layer matching the Megatron-converted OLMo checkpoint contract."""

    def __init__(
        self,
        config: PretrainedConfig,
        *,
        layer_id: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        parallel = get_parallel()
        if parallel.tp_size != 1 or parallel.attn_tp_size != 1:
            raise NotImplementedError("The initial OLMo KDA path requires TP=1")

        self.layer_id = layer_id
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.gate_dim = self.num_v_heads * self.head_k_dim
        self.conv_size = config.linear_conv_kernel_dim

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_k_dim,
            self.num_k_heads,
            self.num_k_heads,
            bias=False,
            quant_config=quant_config,
            v_head_size=self.head_v_dim,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.f_proj_1 = ReplicatedLinear(
            config.hidden_size,
            self.head_v_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("f_proj_1", prefix),
        )
        self.f_proj_2 = ColumnParallelLinear(
            self.head_v_dim,
            self.gate_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("f_proj_2", prefix),
        )
        self.beta_proj = ColumnParallelLinear(
            config.hidden_size,
            self.num_v_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("beta_proj", prefix),
        )
        self.g_proj_1 = ReplicatedLinear(
            config.hidden_size,
            self.head_v_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("g_proj_1", prefix),
        )
        self.g_proj_2 = ColumnParallelLinear(
            self.head_v_dim,
            self.value_dim,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("g_proj_2", prefix),
        )

        self.qkv_conv1d = MergedColumnParallelLinear(
            input_size=self.conv_size,
            output_sizes=[self.key_dim, self.key_dim, self.value_dim],
            bias=False,
            params_dtype=torch.float32,
            prefix=add_prefix("qkv_conv1d", prefix),
        )
        self.qkv_conv1d.weight.data = self.qkv_conv1d.weight.data.unsqueeze(1)

        self.A_log = nn.Parameter(torch.empty(self.num_v_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(self.gate_dim, dtype=torch.float32))
        self.o_norm = FusedRMSNormGated(
            self.head_v_dim,
            eps=config.linear_norm_eps,
            activation="sigmoid",
        )
        self.o_proj = RowParallelLinear(
            self.value_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        self.attn = RadixLinearAttention(
            layer_id=layer_id,
            num_q_heads=self.num_k_heads,
            num_k_heads=self.num_k_heads,
            num_v_heads=self.num_v_heads,
            head_q_dim=self.head_k_dim,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            conv_weights=self.qkv_conv1d.weight.squeeze(1),
            bias=self.qkv_conv1d.bias,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        del positions
        mixed_qkv = self.qkv_proj(hidden_states)[0]
        raw_gate = self.f_proj_2(self.f_proj_1(hidden_states)[0])[0]
        raw_beta = self.beta_proj(hidden_states)[0]
        output_gate = self.g_proj_2(self.g_proj_1(hidden_states)[0])[0]

        if not forward_batch.forward_mode.is_decode():
            raw_gate = raw_gate.unflatten(-1, (self.num_v_heads, self.head_k_dim)).unsqueeze(0)
        raw_beta = raw_beta.unsqueeze(0)

        output = self.attn(
            forward_batch,
            mixed_qkv=mixed_qkv,
            a=raw_gate,
            b=raw_beta,
        )
        output_gate = output_gate.unflatten(-1, (self.num_v_heads, self.head_v_dim))
        output = self.o_norm(output, output_gate)
        output = output.squeeze(0).flatten(-2)
        return self.o_proj(output)[0]
