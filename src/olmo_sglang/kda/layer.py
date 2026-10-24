# SPDX-License-Identifier: Apache-2.0

"""OLMo KDA model layer built on SGLang's linear-attention scheduler."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from sglang.kernels.ops.attention.fla.fused_norm_gate import FusedRMSNormGated
from sglang.srt.distributed.utils import divide
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
from sglang.srt.model_loader.weight_utils import sharded_weight_loader
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix, set_weight_attrs
from torch import nn
from transformers import PretrainedConfig

from olmo_sglang.kda import gate_projections, projections


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
        if parallel.tp_size != parallel.attn_tp_size:
            raise NotImplementedError(
                "OLMo KDA currently requires standard TP with matching model and attention groups"
            )
        attn_tp_size = parallel.attn_tp_size
        attn_tp_rank = parallel.attn_tp_rank

        self.layer_id = layer_id
        self.total_num_k_heads = config.linear_num_key_heads
        self.total_num_v_heads = config.linear_num_value_heads
        if self.total_num_k_heads % attn_tp_size:
            raise ValueError("linear_num_key_heads must be divisible by TP size")
        if self.total_num_v_heads % attn_tp_size:
            raise ValueError("linear_num_value_heads must be divisible by TP size")
        self.num_k_heads = divide(self.total_num_k_heads, attn_tp_size)
        self.num_v_heads = divide(self.total_num_v_heads, attn_tp_size)
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.gate_dim = self.num_v_heads * self.head_k_dim
        total_key_dim = self.total_num_k_heads * self.head_k_dim
        total_value_dim = self.total_num_v_heads * self.head_v_dim
        total_gate_dim = self.total_num_v_heads * self.head_k_dim
        self.conv_size = config.linear_conv_kernel_dim

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_k_dim,
            self.total_num_k_heads,
            self.total_num_k_heads,
            bias=False,
            quant_config=quant_config,
            v_head_size=self.head_v_dim,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
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
            total_gate_dim,
            bias=False,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("f_proj_2", prefix),
        )
        self.beta_proj = ColumnParallelLinear(
            config.hidden_size,
            self.total_num_v_heads,
            bias=False,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
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
            total_value_dim,
            bias=True,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("g_proj_2", prefix),
        )
        self._group_kda_gates = (
            gate_projections.projection_mode() == "grouped"
            and attn_tp_size == 1
            and quant_config is None
        )

        self._pack_kda_projections = (
            projections.projection_mode() == "packed"
            and attn_tp_size == 1
            and quant_config is None
        )
        self.register_buffer("_packed_input_weight", None, persistent=False)
        if self._pack_kda_projections:
            self._repack_input_weights()

        self.qkv_conv1d = MergedColumnParallelLinear(
            input_size=self.conv_size,
            output_sizes=[total_key_dim, total_key_dim, total_value_dim],
            bias=False,
            params_dtype=torch.float32,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_conv1d", prefix),
        )
        self.qkv_conv1d.weight.data = self.qkv_conv1d.weight.data.unsqueeze(1)

        self.A_log = nn.Parameter(torch.empty(self.num_v_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(self.gate_dim, dtype=torch.float32))
        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(0)})
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})
        self.o_norm = FusedRMSNormGated(
            self.head_v_dim,
            eps=config.linear_norm_eps,
            activation="sigmoid",
        )
        self.o_proj = RowParallelLinear(
            total_value_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
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

    def _repack_input_weights(self):
        modules = [getattr(self, name) for name in projections.PROJECTION_NAMES]
        self._packed_input_sizes = [module.weight.shape[0] for module in modules]
        self._packed_input_weight = projections.pack_weights(modules)
        # UnquantizedLinearMethod consumes weight and keeps the selected GEMM
        # backend (Torch or the pinned runtime's supported alternative).
        self._packed_input_layer = SimpleNamespace(weight=self._packed_input_weight)

    def _project_inputs(self, hidden_states):
        output = self.qkv_proj.quant_method.apply(
            self._packed_input_layer, hidden_states
        )
        return output.split(self._packed_input_sizes, dim=-1)

    def _project_decode(self, hidden_states):
        if self._pack_kda_projections:
            mixed_qkv, first_gate, raw_beta, first_output_gate = self._project_inputs(
                hidden_states
            )
        else:
            mixed_qkv = self.qkv_proj(hidden_states)[0]
            first_gate = self.f_proj_1(hidden_states)[0]
            raw_beta = self.beta_proj(hidden_states)[0]
            first_output_gate = self.g_proj_1(hidden_states)[0]
        if self._group_kda_gates:
            raw_gate, output_gate = gate_projections.project(
                first_gate,
                first_output_gate,
                self.f_proj_2.weight,
                self.g_proj_2.weight,
                self.g_proj_2.bias,
            )
        else:
            raw_gate = self.f_proj_2(first_gate)[0]
            output_gate = self.g_proj_2(first_output_gate)[0]
        return mixed_qkv, raw_gate, raw_beta, output_gate

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        if self._pack_kda_projections:
            self._repack_input_weights()
        return self

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        del positions
        if (
            (self._pack_kda_projections or self._group_kda_gates)
            and forward_batch.forward_mode.is_decode()
            and hidden_states.is_cuda
            and hidden_states.dtype == torch.bfloat16
        ):
            mixed_qkv, raw_gate, raw_beta, output_gate = self._project_decode(
                hidden_states
            )
        else:
            mixed_qkv = self.qkv_proj(hidden_states)[0]
            raw_gate = self.f_proj_2(self.f_proj_1(hidden_states)[0])[0]
            raw_beta = self.beta_proj(hidden_states)[0]
            output_gate = self.g_proj_2(self.g_proj_1(hidden_states)[0])[0]

        if not forward_batch.forward_mode.is_decode():
            raw_gate = raw_gate.unflatten(
                -1, (self.num_v_heads, self.head_k_dim)
            ).unsqueeze(0)
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
