# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create deterministic, production-shaped local OLMo parity checkpoints."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from tokenizers import AddedToken, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

LOGGER = logging.getLogger(__name__)
PROFILES = (
    "attention-dense",
    "kda-dense",
    "hybrid-moe",
    "production-shape",
)
CHAT_TEMPLATE = """{%- for message in messages -%}
{{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' -}}
{%- if loop.last and add_generation_prompt -%}
{{- '<|im_start|>assistant\n' -}}
{%- endif -%}
{%- endfor -%}"""


def _randn(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    return (
        torch.randn(shape, generator=generator, dtype=torch.float32)
        .mul_(0.08)
        .bfloat16()
    )


def _config(
    profile: str,
    tokenizer: PreTrainedTokenizerFast,
    max_position_embeddings: int = 64,
) -> dict[str, Any]:
    layer_types = {
        "attention-dense": ["full_attention"],
        "kda-dense": ["linear_attention"],
        "hybrid-moe": ["linear_attention", "full_attention"],
        "production-shape": [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
    }[profile]
    dense_layers = (
        list(range(len(layer_types)))
        if profile not in {"hybrid-moe", "production-shape"}
        else []
    )
    config = {
        "architectures": ["Olmo3MoeForCausalLM"],
        "attention_bias": False,
        "attention_gate_type": "elementwise",
        "attention_hidden_size": 32,
        "bos_token_id": None,
        "dense_layers_indices": dense_layers,
        "dense_mlp_intermediate_size": 48,
        "dtype": "bfloat16",
        "embed_norm": True,
        "embed_scale": 0.5,
        "eos_token_id": tokenizer.eos_token_id,
        "gating_function": "softmax",
        "head_dim": 8,
        "hidden_act": "silu",
        "hidden_size": 32,
        "intermediate_size": 48,
        "latent_moe_dim": 24,
        "layer_types": layer_types,
        "linear_allow_neg_eigval": True,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 8,
        "linear_norm_eps": 1e-5,
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 4,
        "linear_value_head_dim": 16,
        "max_position_embeddings": max_position_embeddings,
        "model_type": "llama",
        "moe_intermediate_size": 20,
        "n_routed_experts": 4,
        "normalize_expert_weights": 1.0,
        "num_attention_heads": 4,
        "num_experts_per_tok": 2,
        "num_hidden_layers": len(layer_types),
        "num_key_value_heads": 2,
        "original_num_experts_per_tok": None,
        "pad_token_id": tokenizer.pad_token_id,
        "restore_weight_scale": True,
        "rms_norm_eps": 1e-5,
        "shared_expert_intermediate_size": 48,
        "sliding_window": 8,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "use_head_qk_norm": True,
        "use_peri_ln": True,
        "use_rope": False,
        "vocab_size": len(tokenizer),
    }
    if profile == "production-shape":
        config.update(
            {
                # Exact production widths and first five-layer attention pattern,
                # with 32 experts so the fixture remains practical locally.
                "attention_hidden_size": 2048,
                "dense_layers_indices": [0],
                "dense_mlp_intermediate_size": 8568,
                "embed_scale": 35.77708763999664,
                "head_dim": 128,
                "hidden_size": 1280,
                "latent_moe_dim": 640,
                "linear_key_head_dim": 128,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 16,
                "linear_value_head_dim": 256,
                "moe_intermediate_size": 952,
                "n_routed_experts": 32,
                "num_attention_heads": 16,
                "num_experts_per_tok": 16,
                "num_key_value_heads": 8,
                "rms_norm_eps": 1e-6,
                "shared_expert_intermediate_size": 952,
            }
        )
    return config


def _build_tokenizer(output_dir: Path) -> PreTrainedTokenizerFast:
    vocabulary = [
        "<unk>",
        "<pad>",
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "system",
        "user",
        "assistant",
        "You",
        "are",
        "a",
        "helpful",
        "AI",
        ".",
        "What",
        "is",
        "the",
        "capital",
        "of",
        "France",
        "?",
        "Paris",
        "Janet",
        "has",
        "3",
        "bags",
        "with",
        "4",
        "apples",
        "in",
        "each",
        "bag",
        "She",
        "gives",
        "away",
        "5",
        "How",
        "many",
        "remain",
    ]
    tokenizer_backend = Tokenizer(
        WordLevel(
            {token: index for index, token in enumerate(vocabulary)}, unk_token="<unk>"
        )
    )
    tokenizer_backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<|endoftext|>",
        additional_special_tokens=[
            AddedToken("<|im_start|>", special=True),
            AddedToken("<|im_end|>", special=True),
        ],
        chat_template=CHAT_TEMPLATE,
    )
    tokenizer.save_pretrained(output_dir)
    (output_dir / "chat_template.jinja").write_text(
        CHAT_TEMPLATE + "\n", encoding="utf-8"
    )
    return tokenizer


def _add_dense_mlp(
    weights: dict[str, torch.Tensor],
    *,
    prefix: str,
    input_size: int,
    intermediate_size: int,
    output_size: int,
    generator: torch.Generator,
) -> None:
    weights[f"{prefix}.gate_proj.weight"] = _randn(
        (intermediate_size, input_size), generator
    )
    weights[f"{prefix}.up_proj.weight"] = _randn(
        (intermediate_size, input_size), generator
    )
    weights[f"{prefix}.down_proj.weight"] = _randn(
        (output_size, intermediate_size), generator
    )


def _add_sparse_mlp(
    weights: dict[str, torch.Tensor],
    *,
    prefix: str,
    config: dict[str, Any],
    generator: torch.Generator,
) -> None:
    hidden_size = config["hidden_size"]
    latent_size = config["latent_moe_dim"]
    weights[f"{prefix}.router.gate.weight"] = _randn(
        (config["n_routed_experts"], hidden_size), generator
    )
    weights[f"{prefix}.latent_down_proj.weight"] = _randn(
        (latent_size, hidden_size), generator
    )
    weights[f"{prefix}.latent_up_proj.weight"] = _randn(
        (hidden_size, latent_size), generator
    )
    for expert_id in range(config["n_routed_experts"]):
        _add_dense_mlp(
            weights,
            prefix=f"{prefix}.experts.{expert_id}",
            input_size=latent_size,
            intermediate_size=config["moe_intermediate_size"],
            output_size=latent_size,
            generator=generator,
        )
    _add_dense_mlp(
        weights,
        prefix=f"{prefix}.shared_expert",
        input_size=hidden_size,
        intermediate_size=config["shared_expert_intermediate_size"],
        output_size=hidden_size,
        generator=generator,
    )


def _add_kda(
    weights: dict[str, torch.Tensor],
    *,
    prefix: str,
    config: dict[str, Any],
    generator: torch.Generator,
) -> None:
    hidden_size = config["hidden_size"]
    key_dim = config["linear_num_key_heads"] * config["linear_key_head_dim"]
    value_dim = config["linear_num_value_heads"] * config["linear_value_head_dim"]
    gate_hidden = config["linear_value_head_dim"]
    gate_dim = config["linear_num_value_heads"] * config["linear_key_head_dim"]
    conv_kernel = config["linear_conv_kernel_dim"]
    weights.update(
        {
            f"{prefix}.q_proj.weight": _randn((key_dim, hidden_size), generator),
            f"{prefix}.k_proj.weight": _randn((key_dim, hidden_size), generator),
            f"{prefix}.v_proj.weight": _randn((value_dim, hidden_size), generator),
            f"{prefix}.f_proj_1.weight": _randn((gate_hidden, hidden_size), generator),
            f"{prefix}.f_proj_2.weight": _randn((gate_dim, gate_hidden), generator),
            f"{prefix}.beta_proj.weight": _randn(
                (config["linear_num_value_heads"], hidden_size), generator
            ),
            f"{prefix}.g_proj_1.weight": _randn((gate_hidden, hidden_size), generator),
            f"{prefix}.g_proj_2.weight": _randn((value_dim, gate_hidden), generator),
            f"{prefix}.g_proj_2.bias": _randn((value_dim,), generator),
            f"{prefix}.q_conv1d.weight": _randn(
                (key_dim, 1, conv_kernel), generator
            ).float(),
            f"{prefix}.k_conv1d.weight": _randn(
                (key_dim, 1, conv_kernel), generator
            ).float(),
            f"{prefix}.v_conv1d.weight": _randn(
                (value_dim, 1, conv_kernel), generator
            ).float(),
            f"{prefix}.A_log": torch.linspace(
                0.0, 1.0, config["linear_num_value_heads"]
            ),
            f"{prefix}.dt_bias": torch.linspace(-0.25, 0.25, gate_dim),
            f"{prefix}.o_norm.weight": torch.linspace(
                0.9, 1.1, config["linear_value_head_dim"], dtype=torch.bfloat16
            ),
            f"{prefix}.o_proj.weight": _randn((hidden_size, value_dim), generator),
        }
    )


def _add_full_attention(
    weights: dict[str, torch.Tensor],
    *,
    prefix: str,
    config: dict[str, Any],
    generator: torch.Generator,
) -> None:
    hidden_size = config["hidden_size"]
    head_dim = config["head_dim"]
    weights.update(
        {
            f"{prefix}.q_proj.weight": _randn(
                (config["num_attention_heads"] * head_dim, hidden_size), generator
            ),
            f"{prefix}.k_proj.weight": _randn(
                (config["num_key_value_heads"] * head_dim, hidden_size), generator
            ),
            f"{prefix}.v_proj.weight": _randn(
                (config["num_key_value_heads"] * head_dim, hidden_size), generator
            ),
            f"{prefix}.q_norm.weight": torch.linspace(
                0.9, 1.1, head_dim, dtype=torch.bfloat16
            ),
            f"{prefix}.k_norm.weight": torch.linspace(
                1.1, 0.9, head_dim, dtype=torch.bfloat16
            ),
            f"{prefix}.g_proj.weight": _randn(
                (config["num_attention_heads"] * head_dim, hidden_size), generator
            ),
            f"{prefix}.o_proj.weight": _randn(
                (hidden_size, config["num_attention_heads"] * head_dim), generator
            ),
        }
    )


def build_checkpoint(
    output_dir: Path, *, profile: str, max_position_embeddings: int = 64
) -> None:
    """Write one strict HF-layout checkpoint and its tokenizer contract."""

    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}")
    if max_position_embeddings <= 0:
        raise ValueError("max_position_embeddings must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = _build_tokenizer(output_dir)
    config = _config(profile, tokenizer, max_position_embeddings)
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "generation_config.json").write_text(
        json.dumps(
            {
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    generator = torch.Generator().manual_seed(2026)
    hidden_size = config["hidden_size"]
    weights: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": _randn(
            (config["vocab_size"], hidden_size), generator
        ),
        "model.embed_norm.weight": torch.linspace(
            0.8, 1.2, hidden_size, dtype=torch.bfloat16
        ),
        "model.norm.weight": torch.linspace(
            1.2, 0.8, hidden_size, dtype=torch.bfloat16
        ),
        "lm_head.weight": _randn((config["vocab_size"], hidden_size), generator),
    }
    for layer_id, layer_type in enumerate(config["layer_types"]):
        prefix = f"model.layers.{layer_id}"
        for norm_index, norm_name in enumerate(
            (
                "pre_attention_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            )
        ):
            weights[f"{prefix}.{norm_name}.weight"] = torch.linspace(
                0.9 + norm_index * 0.01,
                1.1 + norm_index * 0.01,
                hidden_size,
                dtype=torch.bfloat16,
            )
        if layer_type == "linear_attention":
            _add_kda(
                weights,
                prefix=f"{prefix}.self_attn",
                config=config,
                generator=generator,
            )
        else:
            _add_full_attention(
                weights,
                prefix=f"{prefix}.self_attn",
                config=config,
                generator=generator,
            )
        if layer_id in config["dense_layers_indices"]:
            _add_dense_mlp(
                weights,
                prefix=f"{prefix}.mlp",
                input_size=hidden_size,
                intermediate_size=config["dense_mlp_intermediate_size"],
                output_size=hidden_size,
                generator=generator,
            )
        else:
            _add_sparse_mlp(
                weights,
                prefix=f"{prefix}.mlp",
                config=config,
                generator=generator,
            )
    save_file(weights, output_dir / "model.safetensors")
    LOGGER.info("profile=%s output=%s tensors=%d", profile, output_dir, len(weights))


def main() -> None:
    """Create the requested local parity checkpoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--profile", choices=PROFILES, default="hybrid-moe")
    parser.add_argument("--max-position-embeddings", type=int, default=64)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build_checkpoint(
        args.output_dir,
        profile=args.profile,
        max_position_embeddings=args.max_position_embeddings,
    )


if __name__ == "__main__":
    main()
