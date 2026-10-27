"""A reused visual cache must not preserve a previous policy's patch embedding."""

from types import SimpleNamespace

import torch

from olmo_sglang.vision import embedding


def test_cached_features_use_current_patch_embedding_after_language_update():
    lookup = torch.nn.Embedding(16, 8)
    norm = torch.nn.LayerNorm(8)
    language = SimpleNamespace(
        get_normalized_embeddings=lambda ids: norm(lookup(ids) * 8)
    )
    cached_visual_and_text = torch.randn(5, 8)
    cache_before = cached_visual_and_text.clone()
    mask = torch.tensor([False, True, True, False, True])
    first = embedding.add_patch_embedding(cached_visual_and_text, mask, language, 6)
    expected = cached_visual_and_text.clone()
    expected[mask] += language.get_normalized_embeddings(torch.tensor([6]))
    torch.testing.assert_close(first, expected)
    with torch.no_grad():
        lookup.weight[6, 0] += 2
    second = embedding.add_patch_embedding(cached_visual_and_text, mask, language, 6)
    torch.testing.assert_close(first[~mask], second[~mask])
    assert not torch.allclose(first[mask], second[mask])
    torch.testing.assert_close(cached_visual_and_text, cache_before)
