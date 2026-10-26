"""CPU regressions for real export failure modes and Core vision weight layout."""

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from olmo_sglang.vision.config import (
    IMAGE_TOKENS,
    text_config,
    validate_image_tokenizer,
    validate_vision_config,
)

checkpoint = pytest.importorskip("olmo_sglang.vision.checkpoint", exc_type=ImportError)
vision_model = pytest.importorskip("olmo_sglang.vision.model", exc_type=ImportError)
audit_checkpoint = checkpoint.audit_checkpoint
restore_vocabulary = checkpoint.restore_vocabulary
OlmoVisionStack = vision_model.OlmoVisionStack


@pytest.fixture
def config():
    return {
        "model_type": "olmo3moe_vision",
        "text_config": {"model_type": "olmo3moe", "vocab_size": 16, "hidden_size": 8},
        "image_patch_token_id": 6,
        "vit_layers": [1, 0],
        "vision_config": {
            "name": "siglip",
            "image_num_layers": 2,
            "image_emb_dim": 8,
            "image_num_heads": 2,
            "image_num_key_value_heads": 2,
            "image_head_dim": 4,
            "image_mlp_dim": 16,
            "image_patch_size": 2,
            "image_default_input_size": [4, 4],
            "image_num_pos": 4,
            "patch_embedding_bias": True,
            "use_cls_token": False,
            "use_pre_ln": False,
        },
        "connector_config": {
            "image_emb_dim": 8,
            "image_num_heads": 2,
            "image_num_key_value_heads": 2,
            "image_head_dim": 4,
            "output_dim": 8,
            "num_input_layers": 2,
            "pooling_type": "attention_meanq",
            "pooling_attention_mask": True,
            "projector_type": "mlp",
            "mlp_hidden_size": 16,
        },
    }


def test_real_export_vocab_failure_precedes_allocation(config):
    config["text_config"]["vocab_size"] = 100278
    config["image_patch_token_id"] = 100280
    with pytest.raises(ValueError, match="Restore them"):
        validate_vision_config(config)


@pytest.mark.parametrize(
    "update,match",
    [
        ({"vit_layers": [2, 0]}, "vit_layers"),
        ({"vit_layers": [1]}, "num_input_layers"),
        ({"image_patch_token_id": True}, "outside"),
    ],
)
def test_invalid_layout(config, update, match):
    config.update(update)
    with pytest.raises(ValueError, match=match):
        validate_vision_config(config)


def test_tokenizer_requires_exact_image_ids(config):
    vocab = {name: i + 4 for i, name in enumerate(IMAGE_TOKENS)}
    tokenizer = SimpleNamespace(get_vocab=lambda: vocab)
    assert validate_image_tokenizer(config, tokenizer)["<im_patch>"] == 6
    vocab["<im_patch>"] = 12
    with pytest.raises(ValueError, match="differs"):
        validate_image_tokenizer(config, tokenizer)
    del vocab["<im_patch>"]
    with pytest.raises(ValueError, match="missing"):
        validate_image_tokenizer(config, tokenizer)


def test_text_config_is_not_changed():
    config = SimpleNamespace(model_type="olmo3moe")
    assert text_config(config) is config
    assert (
        text_config(SimpleNamespace(model_type="olmo3moe_vision", text_config=config))
        is config
    )


def test_core_weights_and_features_match(config):
    torch.manual_seed(17)
    reference = OlmoVisionStack(config).eval()
    for p in reference.parameters():
        torch.nn.init.normal_(p, std=0.02)
    actual = OlmoVisionStack(config).eval()
    core_weights = [
        (k.replace("vision_tower.", "vision.", 1), v.clone())
        for k, v in reference.state_dict().items()
    ]
    loaded = actual.load_weights(core_weights)
    assert loaded == set(actual.state_dict())
    images = torch.randn(2, 2, 4, 12)
    pools = torch.tensor([[[0, 1], [2, -1], [4, 6]], [[1, 3], [5, -1], [6, 7]]])
    with torch.no_grad():
        hidden = reference.vision_tower(images.reshape(4, 4, 12))
        features = torch.cat([hidden[1], hidden[0]], -1).reshape(2, 8, 16)
        expected = reference.connector(features, pools)
        assert torch.equal(actual.encode_images(images, pools), expected)
    with pytest.raises(ValueError, match="Missing"):
        actual.load_weights(core_weights[:-1])
    with pytest.raises(ValueError, match="Duplicate"):
        actual.load_weights(core_weights + core_weights[:1])
    with pytest.raises(ValueError, match="Unexpected"):
        actual.load_weights([("lm.unexpected", torch.ones(1))])
    bad = [(core_weights[0][0], torch.zeros(1))]
    with pytest.raises(ValueError, match="Shape mismatch"):
        actual.load_weights(bad)
    with pytest.raises(ValueError, match="outside"):
        actual.encode_images(images, pools + 10)


@pytest.fixture
def export_pair(tmp_path, config):
    source, core = tmp_path / "hf", tmp_path / "core"
    source.mkdir()
    core.mkdir()
    tok = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, "<|im_end|>": 1, "hello": 2, "world": 3}, unk_token="[UNK]"
        )
    )
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]").save_pretrained(
        source
    )
    original = {
        "lm": {"vocab_size": 16},
        "vision": config["vision_config"],
        "connector": config["connector_config"],
        "vit_layers": config["vit_layers"],
        "image_patch_token_id": 6,
    }
    (core / "olmo_core_config.json").write_text(json.dumps({"model": original}))
    truncated = copy.deepcopy(config)
    truncated["text_config"]["vocab_size"] = 4
    (source / "config.json").write_text(json.dumps(truncated))
    embeddings = torch.arange(128).reshape(16, 8).float()
    head = embeddings + 1
    save_file(
        {"lm.embeddings.weight": embeddings, "lm.lm_head.w_out.weight": head},
        core / "model.safetensors",
    )
    save_file(
        {
            "language_model.model.embed_tokens.weight": embeddings[:4].clone(),
            "language_model.lm_head.weight": head[:4].clone(),
            "language_model.model.layers.0.example": torch.ones(2),
            "vision_tower.example": torch.ones(2),
            "connector.example": torch.ones(2),
        },
        source / "model.safetensors",
    )
    return source, core, tmp_path / "restored"


def test_restore_preserves_originals_and_trained_rows(export_pair):
    source, core, output = export_pair
    before = {p: p.read_bytes() for root in (source, core) for p in root.iterdir()}
    with pytest.raises(ValueError, match="outside"):
        audit_checkpoint(source)
    receipt = restore_vocabulary(source, core, output)
    assert receipt["audit"]["vocab_size"] == 16
    assert receipt["image_tokens"]["<im_patch>"] == 6
    assert receipt["output_vocab_policy"] == "full_core_vocabulary"
    assert all(p.read_bytes() == content for p, content in before.items())
    actual = load_file(output / "restored-vocabulary.safetensors")
    expected = load_file(core / "model.safetensors")
    assert torch.equal(
        actual["language_model.model.embed_tokens.weight"],
        expected["lm.embeddings.weight"],
    )
    assert torch.equal(
        actual["language_model.lm_head.weight"], expected["lm.lm_head.w_out.weight"]
    )
    # Exercise the serving iterator's actual semantics: every tensor in every
    # shard, not per-key index filtering. No stale vocabulary key may survive.
    all_keys = []
    for shard in output.glob("*.safetensors"):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            all_keys.extend(handle.keys())
    assert len(all_keys) == len(set(all_keys)) == 5
    assert torch.equal(
        load_file(output / "source-00000.safetensors")["vision_tower.example"],
        torch.ones(2),
    )
    with pytest.raises(FileExistsError):
        restore_vocabulary(source, core, output)


def test_restore_rejects_unrelated_weights_before_writing(export_pair):
    source, core, output = export_pair
    save_file(
        {
            "lm.embeddings.weight": torch.zeros(16, 8),
            "lm.lm_head.w_out.weight": torch.zeros(16, 8),
        },
        core / "model.safetensors",
    )
    with pytest.raises(ValueError, match="exact prefix"):
        restore_vocabulary(source, core, output)
    assert not output.exists()


def test_serving_wrapper_streams_all_components_and_rejects_images(config, monkeypatch):
    # Substitute only the GPU-only text executor. The wrapper, Core vision
    # construction, tensors and streaming loader under test are real CPU code.
    class TextExecutor(torch.nn.Module):
        def __init__(self, cfg, prefix):
            super().__init__()
            self.config = SimpleNamespace(n_routed_experts=4)
            self.proj = torch.nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

        def load_weights(self, weights):
            state = dict(weights)
            self.load_state_dict(state, strict=True)
            return set(state)

        def forward(self, ids, positions, batch):
            return ids

    substitute = ModuleType("olmo_sglang.models.olmo3_moe")
    substitute.Olmo3MoeForCausalLM = TextExecutor
    monkeypatch.setitem(sys.modules, substitute.__name__, substitute)
    path = Path(__file__).parents[1] / "src/olmo_sglang/models/olmo3_moe_vision.py"
    spec = importlib.util.spec_from_file_location("vision_wrapper_cpu_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = SimpleNamespace(
        **{**config, "text_config": SimpleNamespace(**config["text_config"])}
    )
    model = module.Olmo3MoeVisionForConditionalGeneration(cfg)
    weights = [(name, torch.randn_like(p)) for name, p in model.named_parameters()]
    assert model.load_weights(iter(weights)) == {name for name, _ in weights}
    assert all(
        torch.equal(dict(model.named_parameters())[name], tensor)
        for name, tensor in weights
    )
    with pytest.raises(ValueError, match="Duplicate"):
        model.load_weights(iter(weights + weights[:1]))
    missing_visual = [(n, t) for n, t in weights if n != weights[0][0]]
    with pytest.raises(ValueError, match="Missing vision"):
        model.load_weights(iter(missing_visual))
    with pytest.raises(ValueError, match="Unexpected"):
        model.load_weights(iter(weights + [("unknown.weight", torch.ones(1))]))
    ids = torch.tensor([1, 2])
    assert torch.equal(model(ids, ids, SimpleNamespace(mm_inputs=None)), ids)
    with pytest.raises(NotImplementedError, match="#65"):
        model(ids, ids, SimpleNamespace(mm_inputs=[object()]))
    with pytest.raises(NotImplementedError, match="#65"):
        model(ids, ids, SimpleNamespace(mm_inputs=None), input_embeds=torch.ones(2, 8))
