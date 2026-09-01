# SPDX-License-Identifier: Apache-2.0

"""Create a deterministic tiny OLMo hybrid checkpoint for validation."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from safetensors.torch import save_file

LOGGER = logging.getLogger(__name__)


def _randn(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    return (
        torch.randn(shape, generator=generator, dtype=torch.float32).mul_(0.02).half()
    )


def build_checkpoint(output_dir: Path) -> None:
    """Write a tiny mixed KDA/full-attention checkpoint in HF layout."""

    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "architectures": ["Olmo3MoeForCausalLM"],
        "attention_bias": False,
        "attention_gate_type": None,
        "attention_hidden_size": 32,
        "bos_token_id": None,
        "dense_layers_indices": [0, 1],
        "dense_mlp_intermediate_size": 48,
        "dtype": "float16",
        "embed_norm": True,
        "embed_scale": 0.5,
        "eos_token_id": 63,
        "gating_function": "softmax",
        "head_dim": 8,
        "hidden_act": "silu",
        "hidden_size": 32,
        "intermediate_size": 48,
        "layer_types": ["linear_attention", "full_attention"],
        "linear_allow_neg_eigval": True,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 8,
        "linear_norm_eps": 1e-5,
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 4,
        "linear_value_head_dim": 16,
        "max_position_embeddings": 32,
        "model_type": "llama",
        "moe_intermediate_size": 16,
        "n_routed_experts": 4,
        "normalize_expert_weights": 1.0,
        "num_attention_heads": 4,
        "num_experts_per_tok": 2,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "original_num_experts_per_tok": None,
        "pad_token_id": 1,
        "restore_weight_scale": True,
        "rms_norm_eps": 1e-5,
        "shared_expert_intermediate_size": None,
        "sliding_window": 8,
        "tie_word_embeddings": False,
        "use_cache": True,
        "use_head_qk_norm": True,
        "use_peri_ln": True,
        "use_rope": False,
        "vocab_size": 64,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "generation_config.json").write_text(
        json.dumps({"eos_token_id": 63, "pad_token_id": 1}, indent=2) + "\n",
        encoding="utf-8",
    )

    generator = torch.Generator().manual_seed(2026)
    weights: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": _randn((64, 32), generator),
        "model.embed_norm.weight": torch.ones(32, dtype=torch.float16),
        "model.norm.weight": torch.ones(32, dtype=torch.float16),
        "lm_head.weight": _randn((64, 32), generator),
    }
    for layer_id in range(2):
        prefix = f"model.layers.{layer_id}"
        for norm_name in (
            "pre_attention_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            weights[f"{prefix}.{norm_name}.weight"] = torch.ones(
                32, dtype=torch.float16
            )
        weights[f"{prefix}.mlp.gate_proj.weight"] = _randn((48, 32), generator)
        weights[f"{prefix}.mlp.up_proj.weight"] = _randn((48, 32), generator)
        weights[f"{prefix}.mlp.down_proj.weight"] = _randn((32, 48), generator)

    prefix = "model.layers.0.self_attn"
    weights.update(
        {
            f"{prefix}.q_proj.weight": _randn((32, 32), generator),
            f"{prefix}.k_proj.weight": _randn((32, 32), generator),
            f"{prefix}.v_proj.weight": _randn((64, 32), generator),
            f"{prefix}.f_proj_1.weight": _randn((16, 32), generator),
            f"{prefix}.f_proj_2.weight": _randn((32, 16), generator),
            f"{prefix}.beta_proj.weight": _randn((4, 32), generator),
            f"{prefix}.g_proj_1.weight": _randn((16, 32), generator),
            f"{prefix}.g_proj_2.weight": _randn((64, 16), generator),
            f"{prefix}.g_proj_2.bias": torch.zeros(64, dtype=torch.float16),
            f"{prefix}.q_conv1d.weight": _randn((32, 1, 4), generator).float(),
            f"{prefix}.k_conv1d.weight": _randn((32, 1, 4), generator).float(),
            f"{prefix}.v_conv1d.weight": _randn((64, 1, 4), generator).float(),
            f"{prefix}.A_log": torch.linspace(0.0, 1.0, 4),
            f"{prefix}.dt_bias": torch.zeros(32),
            f"{prefix}.o_norm.weight": torch.ones(16, dtype=torch.float16),
            f"{prefix}.o_proj.weight": _randn((32, 64), generator),
        }
    )

    prefix = "model.layers.1.self_attn"
    weights.update(
        {
            f"{prefix}.q_proj.weight": _randn((32, 32), generator),
            f"{prefix}.k_proj.weight": _randn((16, 32), generator),
            f"{prefix}.v_proj.weight": _randn((16, 32), generator),
            f"{prefix}.q_norm.weight": torch.ones(8, dtype=torch.float16),
            f"{prefix}.k_norm.weight": torch.ones(8, dtype=torch.float16),
            f"{prefix}.o_proj.weight": _randn((32, 32), generator),
        }
    )
    save_file(weights, output_dir / "model.safetensors")
    LOGGER.info("%s", output_dir)


def main() -> None:
    """Create the deterministic checkpoint at the requested path."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build_checkpoint(args.output_dir)


if __name__ == "__main__":
    main()
