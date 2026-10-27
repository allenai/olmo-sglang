"""Keep cached visual features independent of trainable language embeddings."""

import torch


def add_patch_embedding(embeddings, patch_mask, language_model, patch_token_id):
    """Add the current normalized patch embedding after cached visual placement.

    The mask must be captured before SGLang clamps hashed image placeholder IDs.
    A single embedding lookup is broadcast over patch positions, avoiding a
    dynamically sized CUDA gather for the image-token count.
    """
    if patch_mask.shape != embeddings.shape[:-1] or patch_mask.dtype != torch.bool:
        raise ValueError("Patch mask must match the embedding sequence")
    ids = torch.tensor([patch_token_id], device=embeddings.device, dtype=torch.long)
    base = language_model.get_normalized_embeddings(ids).to(embeddings.dtype)
    return embeddings + patch_mask.unsqueeze(-1).to(embeddings.dtype) * base
