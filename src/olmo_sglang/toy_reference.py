# SPDX-License-Identifier: Apache-2.0

"""Small, independent PyTorch reference for local OLMo-SGLang parity work."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn


class RMSNorm(nn.Module):
    """Reference RMS normalization."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = inputs.float() * torch.rsqrt(
            inputs.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return (normalized * self.weight.float()).to(inputs.dtype)


class RMSNormGated(RMSNorm):
    """Reference FLA RMSNormGated with sigmoid activation."""

    def forward(self, inputs: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return super().forward(inputs) * gate.float().sigmoid().to(inputs.dtype)


class DenseMLP(nn.Module):
    """Reference SwiGLU MLP with HF-compatible parameter names."""

    def __init__(
        self, input_size: int, intermediate_size: int, output_size: int
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(input_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(input_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, output_size, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(inputs)) * self.up_proj(inputs))


class SparseMLP(nn.Module):
    """Reference latent MoE and shared expert."""

    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.config = config
        self.router = nn.Module()
        self.router.gate = nn.Linear(
            config.hidden_size, config.n_routed_experts, bias=False
        )
        self.latent_down_proj = nn.Linear(
            config.hidden_size, config.latent_moe_dim, bias=False
        )
        self.experts = nn.ModuleList(
            [
                DenseMLP(
                    config.latent_moe_dim,
                    config.moe_intermediate_size,
                    config.latent_moe_dim,
                )
                for _ in range(config.n_routed_experts)
            ]
        )
        self.latent_up_proj = nn.Linear(
            config.latent_moe_dim, config.hidden_size, bias=False
        )
        self.shared_expert = DenseMLP(
            config.hidden_size,
            config.shared_expert_intermediate_size,
            config.hidden_size,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        trace: dict[str, torch.Tensor] | None = None,
        prefix: str,
    ) -> torch.Tensor:
        router_logits = F.linear(inputs.float(), self.router.gate.weight.float())
        scores = router_logits.softmax(dim=-1)
        weights, indices = scores.topk(self.config.num_experts_per_tok, dim=-1)
        normalize = self.config.normalize_expert_weights
        if normalize is not None:
            weights = weights / torch.linalg.vector_norm(
                weights,
                ord=normalize,
                dim=-1,
                keepdim=True,
            )
        if self.config.restore_weight_scale:
            weights = weights * self.config.num_experts_per_tok
        original_topk = self.config.original_num_experts_per_tok
        if (
            original_topk is not None
            and original_topk != self.config.num_experts_per_tok
        ):
            weights = weights * (original_topk / self.config.num_experts_per_tok) ** 0.5

        latent = self.latent_down_proj(inputs)
        flat = latent.reshape(-1, self.config.latent_moe_dim)
        flat_indices = indices.reshape(-1, self.config.num_experts_per_tok)
        flat_weights = weights.reshape(-1, self.config.num_experts_per_tok).to(
            flat.dtype
        )
        routed = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            token_ids, slots = (flat_indices == expert_id).nonzero(as_tuple=True)
            if token_ids.numel() == 0:
                continue
            expert_output = expert(flat.index_select(0, token_ids))
            routed.index_add_(
                0,
                token_ids,
                expert_output * flat_weights[token_ids, slots].unsqueeze(-1),
            )
        output = self.latent_up_proj(routed.view_as(latent)) + self.shared_expert(
            inputs
        )
        if trace is not None:
            trace[f"{prefix}.router_logits"] = router_logits
            trace[f"{prefix}.router_indices"] = indices
            trace[f"{prefix}.output"] = output
        return output


class KDAAttention(nn.Module):
    """Torch recurrence matching the OLMo negative-eigenvalue KDA contract."""

    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.config = config
        key_dim = config.linear_num_key_heads * config.linear_key_head_dim
        value_dim = config.linear_num_value_heads * config.linear_value_head_dim
        gate_hidden = config.linear_value_head_dim
        gate_dim = config.linear_num_value_heads * config.linear_key_head_dim

        self.q_proj = nn.Linear(config.hidden_size, key_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, key_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, value_dim, bias=False)
        self.f_proj_1 = nn.Linear(config.hidden_size, gate_hidden, bias=False)
        self.f_proj_2 = nn.Linear(gate_hidden, gate_dim, bias=False)
        self.beta_proj = nn.Linear(
            config.hidden_size, config.linear_num_value_heads, bias=False
        )
        self.g_proj_1 = nn.Linear(config.hidden_size, gate_hidden, bias=False)
        self.g_proj_2 = nn.Linear(gate_hidden, value_dim, bias=True)
        self.q_conv1d = nn.Conv1d(
            key_dim,
            key_dim,
            config.linear_conv_kernel_dim,
            groups=key_dim,
            bias=False,
        )
        self.k_conv1d = nn.Conv1d(
            key_dim,
            key_dim,
            config.linear_conv_kernel_dim,
            groups=key_dim,
            bias=False,
        )
        self.v_conv1d = nn.Conv1d(
            value_dim,
            value_dim,
            config.linear_conv_kernel_dim,
            groups=value_dim,
            bias=False,
        )
        self.A_log = nn.Parameter(
            torch.empty(config.linear_num_value_heads, dtype=torch.float32)
        )
        self.dt_bias = nn.Parameter(torch.empty(gate_dim, dtype=torch.float32))
        self.o_norm = RMSNormGated(config.linear_value_head_dim, config.linear_norm_eps)
        self.o_proj = nn.Linear(value_dim, config.hidden_size, bias=False)

    @staticmethod
    def _causal_conv(inputs: torch.Tensor, convolution: nn.Conv1d) -> torch.Tensor:
        width = convolution.kernel_size[0]
        transposed = inputs.transpose(1, 2)
        output = F.conv1d(
            transposed,
            convolution.weight.to(inputs.dtype),
            groups=transposed.shape[1],
            padding=width - 1,
        )
        return F.silu(output[..., : inputs.shape[1]]).transpose(1, 2)

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        trace: dict[str, torch.Tensor] | None = None,
        prefix: str,
    ) -> torch.Tensor:
        config = self.config
        batch_size, sequence_length, _ = inputs.shape
        num_heads = config.linear_num_value_heads
        key_head_dim = config.linear_key_head_dim
        value_head_dim = config.linear_value_head_dim

        q = self._causal_conv(self.q_proj(inputs), self.q_conv1d).view(
            batch_size, sequence_length, num_heads, key_head_dim
        )
        k = self._causal_conv(self.k_proj(inputs), self.k_conv1d).view(
            batch_size, sequence_length, num_heads, key_head_dim
        )
        v = self._causal_conv(self.v_proj(inputs), self.v_conv1d).view(
            batch_size, sequence_length, num_heads, value_head_dim
        )
        raw_gate = self.f_proj_2(self.f_proj_1(inputs)).view(
            batch_size, sequence_length, num_heads, key_head_dim
        )
        beta = self.beta_proj(inputs).float().sigmoid()
        if config.linear_allow_neg_eigval:
            beta = beta * 2.0

        q_float = q.float()
        k_float = k.float()
        q_float = (
            q_float
            * torch.rsqrt(q_float.pow(2).sum(-1, keepdim=True) + 1e-6)
            * key_head_dim**-0.5
        )
        k_float = k_float * torch.rsqrt(k_float.pow(2).sum(-1, keepdim=True) + 1e-6)
        decay = -self.A_log.float().exp().view(1, 1, num_heads, 1) * F.softplus(
            raw_gate.float() + self.dt_bias.float().view(1, 1, num_heads, key_head_dim)
        )

        state = torch.zeros(
            batch_size,
            num_heads,
            value_head_dim,
            key_head_dim,
            dtype=torch.float32,
            device=inputs.device,
        )
        outputs = []
        for token_index in range(sequence_length):
            state = state * decay[:, token_index].exp().unsqueeze(-2)
            predicted = torch.einsum("bhvk,bhk->bhv", state, k_float[:, token_index])
            residual = (v[:, token_index].float() - predicted) * beta[
                :, token_index
            ].unsqueeze(-1)
            state = state + residual.unsqueeze(-1) * k_float[:, token_index].unsqueeze(
                -2
            )
            outputs.append(
                torch.einsum("bhvk,bhk->bhv", state, q_float[:, token_index])
            )
        output = torch.stack(outputs, dim=1).to(inputs.dtype)

        output_gate = self.g_proj_2(self.g_proj_1(inputs)).view(
            batch_size, sequence_length, num_heads, value_head_dim
        )
        output = self.o_norm(output, output_gate).flatten(-2)
        output = self.o_proj(output)
        if trace is not None:
            for name, value in (
                ("q", q),
                ("k", k),
                ("v", v),
                ("raw_gate", raw_gate),
                ("beta", beta),
                ("output", output),
            ):
                trace[f"{prefix}.{name}"] = value
        return output


class FullAttention(nn.Module):
    """Reference eager causal attention for the NoPE OLMo variant."""

    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.config = config
        head_dim = config.head_dim
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * head_dim, bias=False
        )
        self.q_norm = RMSNorm(head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, config.rms_norm_eps)
        self.g_proj = (
            nn.Linear(
                config.hidden_size, config.num_attention_heads * head_dim, bias=False
            )
            if config.attention_gate_type == "elementwise"
            else None
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * head_dim, config.hidden_size, bias=False
        )

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        trace: dict[str, torch.Tensor] | None = None,
        prefix: str,
    ) -> torch.Tensor:
        config = self.config
        batch_size, sequence_length, _ = inputs.shape
        q = self.q_proj(inputs).view(
            batch_size, sequence_length, config.num_attention_heads, config.head_dim
        )
        k = self.k_proj(inputs).view(
            batch_size, sequence_length, config.num_key_value_heads, config.head_dim
        )
        v = self.v_proj(inputs).view(
            batch_size, sequence_length, config.num_key_value_heads, config.head_dim
        )
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        repeat = config.num_attention_heads // config.num_key_value_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
        scores = torch.matmul(q, k.transpose(-1, -2)) * config.head_dim**-0.5
        causal_mask = torch.triu(
            torch.full(
                (sequence_length, sequence_length),
                float("-inf"),
                device=inputs.device,
            ),
            diagonal=1,
        )
        probabilities = torch.softmax(scores.float() + causal_mask, dim=-1).to(
            inputs.dtype
        )
        output = (
            torch.matmul(probabilities, v)
            .transpose(1, 2)
            .reshape(batch_size, sequence_length, -1)
        )
        if self.g_proj is not None:
            output = output * self.g_proj(inputs).float().sigmoid().to(output.dtype)
        output = self.o_proj(output)
        if trace is not None:
            trace[f"{prefix}.q"] = q
            trace[f"{prefix}.k"] = k
            trace[f"{prefix}.v"] = v
            trace[f"{prefix}.output"] = output
        return output


class DecoderLayer(nn.Module):
    """Reference peri-LN decoder layer."""

    def __init__(self, config: SimpleNamespace, layer_id: int) -> None:
        super().__init__()
        self.self_attn = (
            KDAAttention(config)
            if config.layer_types[layer_id] == "linear_attention"
            else FullAttention(config)
        )
        self.mlp = (
            DenseMLP(
                config.hidden_size,
                config.dense_mlp_intermediate_size,
                config.hidden_size,
            )
            if layer_id in config.dense_layers_indices
            else SparseMLP(config)
        )
        self.pre_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        trace: dict[str, torch.Tensor] | None,
        prefix: str,
    ) -> torch.Tensor:
        residual = inputs
        attention_inputs = self.pre_attention_layernorm(inputs)
        if trace is not None:
            trace[f"{prefix}.attention_input"] = attention_inputs
        attention_output = self.self_attn(
            attention_inputs,
            trace=trace,
            prefix=f"{prefix}.self_attn",
        )
        hidden_states = residual + self.post_attention_layernorm(attention_output)

        residual = hidden_states
        mlp_inputs = self.pre_feedforward_layernorm(hidden_states)
        if trace is not None:
            trace[f"{prefix}.mlp_input"] = mlp_inputs
        if isinstance(self.mlp, SparseMLP):
            mlp_output = self.mlp(mlp_inputs, trace=trace, prefix=f"{prefix}.mlp")
        else:
            mlp_output = self.mlp(mlp_inputs)
            if trace is not None:
                trace[f"{prefix}.mlp.output"] = mlp_output
        hidden_states = residual + self.post_feedforward_layernorm(mlp_output)
        if trace is not None:
            trace[f"{prefix}.output"] = hidden_states
        return hidden_states


class ToyReferenceForCausalLM(nn.Module):
    """Tiny independent model that consumes the same HF-layout weights as SGLang."""

    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.config = config
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.model.embed_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.model.layers = nn.ModuleList(
            [
                DecoderLayer(config, layer_id)
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.model.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        return_trace: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        trace: dict[str, torch.Tensor] | None = {} if return_trace else None
        hidden_states = self.model.embed_tokens(input_ids) * self.config.embed_scale
        hidden_states = self.model.embed_norm(hidden_states)
        if trace is not None:
            trace["model.embedding"] = hidden_states
        for layer_id, layer in enumerate(self.model.layers):
            hidden_states = layer(
                hidden_states,
                trace=trace,
                prefix=f"model.layers.{layer_id}",
            )
        hidden_states = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states).float()
        if trace is not None:
            trace["model.final_norm"] = hidden_states
            trace["logits"] = logits
        return logits, trace

    @classmethod
    def from_pretrained(
        cls, model_path: Path, *, device: torch.device
    ) -> ToyReferenceForCausalLM:
        """Load a generated local parity checkpoint strictly."""

        config_data = json.loads(
            (model_path / "config.json").read_text(encoding="utf-8")
        )
        config = SimpleNamespace(**config_data)
        model = cls(config)
        state_dict = load_file(model_path / "model.safetensors")
        model.load_state_dict(state_dict, strict=True)
        return model.to(device).eval()

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, *, max_new_tokens: int) -> list[int]:
        """Greedily generate by recomputing the full sequence each step."""

        generated: list[int] = []
        sequence = input_ids
        for _ in range(max_new_tokens):
            logits, _ = self(sequence)
            next_token = int(logits[0, -1].argmax().item())
            generated.append(next_token)
            sequence = torch.cat(
                (
                    sequence,
                    torch.tensor(
                        [[next_token]], dtype=torch.long, device=sequence.device
                    ),
                ),
                dim=1,
            )
        return generated


def trace_summary(trace: dict[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    """Convert reference tensors into compact JSON diagnostics."""

    summary: dict[str, dict[str, Any]] = {}
    for name, tensor in trace.items():
        values = tensor.detach().float()
        summary[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "finite": bool(torch.isfinite(values).all().item()),
            "mean": float(values.mean().item()),
            "std": float(values.std().item()) if values.numel() > 1 else 0.0,
            "max_abs": float(values.abs().max().item()),
        }
    return summary
