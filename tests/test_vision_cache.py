"""Portable regressions for model-owned image features across weight reloads."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


class TextExecutor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(n_routed_experts=4)
        self.embed_tokens = torch.nn.Embedding(8, 4)
        self.embed_norm = torch.nn.LayerNorm(4, bias=False)

    def get_normalized_embeddings(self, ids):
        return self.embed_norm(self.embed_tokens(ids))

    def load_weights(self, weights):
        state = dict(weights)
        self.load_state_dict(state, strict=True)
        return set(state)


@pytest.fixture
def model_and_cache(monkeypatch):
    text_module = ModuleType("olmo_sglang.models.olmo3_moe")
    text_module.Olmo3MoeForCausalLM = TextExecutor
    monkeypatch.setitem(sys.modules, text_module.__name__, text_module)
    # The cache lifecycle is independent of Core's encoder arithmetic. Substitute
    # small CPU layers while exercising the actual wrapper feature and loader paths.
    vision_module = ModuleType("olmo_sglang.vision.model")
    calls = []

    def encode_images(model, pixels, pools):
        calls.append(True)
        return model.connector(model.vision_tower(pixels))

    vision_module.OlmoVisionStack = SimpleNamespace(encode_images=encode_images)
    monkeypatch.setitem(sys.modules, vision_module.__name__, vision_module)
    schedule = ModuleType("sglang.srt.managers.mm_schedule")
    schedule.embedding_cache = {}
    monkeypatch.setitem(sys.modules, schedule.__name__, schedule)
    path = Path(__file__).parents[1] / "src/olmo_sglang/models/olmo3_moe_vision.py"
    spec = importlib.util.spec_from_file_location("vision_cache_test_wrapper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = module.Olmo3MoeVisionForConditionalGeneration
    model = cls.__new__(cls)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(image_patch_token_id=2)
    with torch.random.fork_rng():
        torch.manual_seed(29)
        model.vision_tower = torch.nn.Linear(4, 4, bias=False)
        model.connector = torch.nn.Linear(4, 4, bias=False)
        model.language_model = TextExecutor()
    model.eval()
    item = SimpleNamespace(
        hash=123,
        feature=torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        pooling_indices=torch.tensor([[0]]),
        offsets=[(0, 0)],
        precomputed_embeddings=None,
    )
    return model, schedule, item, calls


def cached_feature(model, cache, item):
    if item.hash not in cache:
        cache[item.hash] = model.get_image_feature([item]).detach().clone()
    return cache[item.hash]


@pytest.mark.parametrize(
    "changed",
    [
        "vision_tower.weight",
        "connector.weight",
        "language_model.embed_tokens.weight",
        "language_model.embed_norm.weight",
    ],
)
def test_weight_reload_recomputes_cached_image_features(model_and_cache, changed):
    model, schedule, item, calls = model_and_cache
    before = cached_feature(model, schedule.embedding_cache, item).clone()
    assert torch.equal(cached_feature(model, schedule.embedding_cache, item), before)
    assert len(calls) == 1, "The unchanged model must reuse its image feature"
    weights = {name: p.detach().clone() for name, p in model.named_parameters()}
    if changed.endswith("embed_tokens.weight"):
        weights[changed][model.config.image_patch_token_id, 0] += 3
    else:
        weights[changed] += 0.5
    model.load_weights(weights.items())
    actual = cached_feature(model, schedule.embedding_cache, item)
    fresh = model.get_image_feature([item]).detach()
    assert not torch.equal(before, fresh), "The intervention must change features"
    assert torch.equal(actual, fresh), "Image cache retained features from old weights"
    assert len(calls) == 3
    assert torch.equal(cached_feature(model, schedule.embedding_cache, item), fresh)
    assert len(calls) == 3, "Cache reuse should resume after a successful reload"


def test_failed_partial_reload_also_invalidates_features(model_and_cache):
    model, schedule, item, _ = model_and_cache
    before = cached_feature(model, schedule.embedding_cache, item).clone()
    replacement = model.vision_tower.weight.detach().clone() + 0.5
    weights = [("vision_tower.weight", replacement), ("unknown.weight", torch.ones(1))]
    with pytest.raises(ValueError, match="Unexpected multimodal"):
        model.load_weights(weights)
    assert torch.equal(model.vision_tower.weight, replacement)
    fresh = model.get_image_feature([item]).detach()
    assert not torch.equal(before, fresh)
    assert torch.equal(cached_feature(model, schedule.embedding_cache, item), fresh)


@pytest.mark.parametrize("module_loaded", [False, True])
def test_initial_loading_without_an_initialized_cache(
    model_and_cache, monkeypatch, module_loaded
):
    model, schedule, _, _ = model_and_cache
    schedule.embedding_cache = None
    if not module_loaded:
        monkeypatch.delitem(sys.modules, schedule.__name__)
    weights = [(name, p.detach().clone()) for name, p in model.named_parameters()]
    assert model.load_weights(weights) == {name for name, _ in weights}
