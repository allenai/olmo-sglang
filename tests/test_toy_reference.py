# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoTokenizer

from olmo_sglang.validation.reference import (
    FullAttention,
    ToyReferenceForCausalLM,
    trace_summary,
)

GENERATOR_PATH = (
    Path(__file__).parents[1] / "tools" / "create_tiny_parity_checkpoint.py"
)


def _generator_module():
    spec = importlib.util.spec_from_file_location(
        "create_tiny_parity_checkpoint", GENERATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("profile", "expected_layers", "expects_router"),
    (
        ("attention-dense", 1, False),
        ("kda-dense", 1, False),
        ("hybrid-moe", 2, True),
        ("scaled-attention-hybrid-moe", 2, True),
        ("biased-sliding-hybrid-moe", 2, True),
    ),
)
def test_toy_reference_strictly_loads_each_profile(
    tmp_path: Path, profile: str, expected_layers: int, expects_router: bool
) -> None:
    checkpoint = tmp_path / profile
    _generator_module().build_checkpoint(checkpoint, profile=profile)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is the capital of France?"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = tokenizer(
        rendered, add_special_tokens=False, return_tensors="pt"
    ).input_ids
    model = ToyReferenceForCausalLM.from_pretrained(
        checkpoint, device=torch.device("cpu")
    )
    logits, trace = model(input_ids, return_trace=True)

    assert len(model.model.layers) == expected_layers
    assert logits.shape == (1, input_ids.shape[1], len(tokenizer))
    assert trace is not None
    assert trace_summary(trace)["logits"]["finite"]
    assert any("router_indices" in name for name in trace) is expects_router


def test_toy_chat_template_and_token_ids_are_stable(tmp_path: Path) -> None:
    checkpoint = tmp_path / "attention"
    _generator_module().build_checkpoint(checkpoint, profile="attention-dense")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is the capital of France?"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]

    assert rendered == (
        "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert input_ids[0] == tokenizer.convert_tokens_to_ids("<|im_start|>")
    assert input_ids[-2] == tokenizer.convert_tokens_to_ids("<|im_start|>")
    assert input_ids[-1] == tokenizer.convert_tokens_to_ids("assistant")


def test_sliding_reference_excludes_tokens_outside_the_window(tmp_path: Path) -> None:
    generator = _generator_module()
    tokenizer = generator._build_tokenizer(tmp_path)
    config = generator._config("attention-dense", tokenizer)
    config.update(layer_types=["sliding_attention"], attention_bias=True)
    torch.manual_seed(13)
    model = FullAttention(SimpleNamespace(**config))
    inputs = torch.randn(1, 16, config["hidden_size"])
    changed = inputs.clone()
    changed[:, :8] += 10
    expected = model(inputs, prefix="attention")
    actual = model(changed, prefix="attention")
    torch.testing.assert_close(actual[:, -1], expected[:, -1], atol=0, rtol=0)
    assert not torch.equal(actual[:, 0], expected[:, 0])


def test_production_shape_profile_preserves_critical_dimensions(tmp_path: Path) -> None:
    tokenizer = _generator_module()._build_tokenizer(tmp_path)
    config = _generator_module()._config("production-shape", tokenizer)

    assert config["hidden_size"] == 1280
    assert config["layer_types"] == ["linear_attention"] * 4 + ["full_attention"]
    assert config["dense_layers_indices"] == [0]
    assert config["attention_hidden_size"] == 2048
    assert config["num_attention_heads"] == 16
    assert config["num_key_value_heads"] == 8
    assert config["head_dim"] == 128
    assert config["linear_num_key_heads"] == 16
    assert config["linear_num_value_heads"] == 16
    assert config["linear_key_head_dim"] == 128
    assert config["linear_value_head_dim"] == 256
    assert config["dense_mlp_intermediate_size"] == 8568
    assert config["moe_intermediate_size"] == 952
    assert config["latent_moe_dim"] == 640
    assert config["num_experts_per_tok"] == 16
