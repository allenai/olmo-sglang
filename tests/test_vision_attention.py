"""Portable checks for Core's causal image-attention contract."""

import pytest

from olmo_sglang.vision.config import validate_causal_image_attention


@pytest.mark.parametrize("document_mode", [None, False, True])
@pytest.mark.parametrize("has_recurrent_mixers", [False, True])
def test_image_attention_policy(document_mode, has_recurrent_mixers):
    if has_recurrent_mixers or document_mode is True:
        validate_causal_image_attention(
            document_mode, has_recurrent_mixers=has_recurrent_mixers
        )
    else:
        with pytest.raises(ValueError, match="Bidirectional image attention"):
            validate_causal_image_attention(
                document_mode, has_recurrent_mixers=has_recurrent_mixers
            )


@pytest.mark.parametrize("document_mode", ["false", 0, 1])
def test_image_attention_rejects_malformed_mode(document_mode):
    with pytest.raises(ValueError, match="boolean or None"):
        validate_causal_image_attention(document_mode, has_recurrent_mixers=True)
