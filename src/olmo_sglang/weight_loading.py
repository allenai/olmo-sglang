# SPDX-License-Identifier: Apache-2.0

"""Portable checkpoint-name resolution for native packed decoder parameters."""

import re
from typing import Any

import torch

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
_FUSED_EXPERT_WEIGHT = re.compile(r"\.mlp\.experts\.(gate_up_proj|down_proj)\.")
_FUSED_TARGETS = {
    "gate_up_proj": ("experts.w13_", "w13"),
    "down_proj": ("experts.w2_", "w2"),
}


class WeightTargets:
    """Resolve HF checkpoint names to SGLang parameters without scanning mappings.

    ``resolve`` returns ``(kind, parameter_name, shard_id, expert_id)`` or
    ``None`` for names that are skipped. Results are memoized per checkpoint
    name; unknown names raise ``KeyError`` exactly as the scanning loader did.
    """

    def __init__(
        self,
        params: dict[str, torch.nn.Parameter],
        num_experts: int,
        fused_layers: dict[str, Any] | None = None,
    ):
        self.params = params
        self.num_experts = num_experts
        self.fused_layers = fused_layers or {}
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
        fused = _FUSED_EXPERT_WEIGHT.search(name)
        if fused is not None:
            prefix, shard_id = _FUSED_TARGETS[fused.group(1)]
            mapped_name = name.replace(f"experts.{fused.group(1)}.", prefix)
            if mapped_name in self.fused_layers and mapped_name in self.params:
                return ("fused", mapped_name, shard_id, None)
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
