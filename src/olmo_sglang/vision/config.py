"""Validate the supported Core SigLIP/connector checkpoint contract."""

from collections.abc import Mapping
from typing import Any

VISION_ARCHITECTURE = "Olmo3MoeVisionForConditionalGeneration"
VISION_MODEL_TYPE = "olmo3moe_vision"
IMAGE_TOKENS = (
    "<im_start>",
    "<im_end>",
    "<im_patch>",
    "<im_col>",
    "<low_res_im_start>",
    "<|image|>",
)


def field(config: Any, name: str, default: Any = None) -> Any:
    """Read either a serialized configuration or a HF configuration object."""
    return (
        config.get(name, default)
        if isinstance(config, Mapping)
        else getattr(config, name, default)
    )


def text_config(config: Any) -> Any:
    """Unwrap only the known Olmo vision architecture, preserving text configs."""
    if field(config, "model_type") == VISION_MODEL_TYPE:
        return field(config, "text_config")
    return config


def validate_vision_config(config: Any) -> None:
    """Reject unsupported layouts and image IDs before allocating model weights."""
    if field(config, "model_type") != VISION_MODEL_TYPE:
        raise ValueError("Expected model_type=olmo3moe_vision")
    text = field(config, "text_config")
    if field(text, "model_type") != "olmo3moe":
        raise ValueError("text_config.model_type must be olmo3moe")
    vocab = field(text, "vocab_size")
    patch = field(config, "image_patch_token_id")
    if (
        type(vocab) is not int
        or vocab <= 0
        or type(patch) is not int
        or not 0 <= patch < vocab
    ):
        raise ValueError(
            f"image_patch_token_id={patch} is outside text vocabulary {vocab}; "
            "language-only exports may omit trained image-token rows. Restore them "
            "from the original checkpoint; do not initialize or remap them."
        )
    vision = field(config, "vision_config", {})
    connector = field(config, "connector_config", {})
    if not isinstance(vision, Mapping) or not isinstance(connector, Mapping):
        raise ValueError(
            "vision_config and connector_config must be Core config dictionaries"
        )
    if vision.get("name") not in ("siglip", "siglip2"):
        raise ValueError("Only the Core SigLIP vision layout is supported")
    if vision.get("use_cls_token", True) or vision.get("use_pre_ln", True):
        raise ValueError("SigLIP checkpoint must disable CLS tokens and pre-LN")
    layers = field(config, "vit_layers")
    depth = vision.get("image_num_layers")
    if (
        type(depth) is not int
        or depth <= 0
        or not isinstance(layers, (list, tuple))
        or not layers
        or any(type(i) is not int or not -depth <= i < depth for i in layers)
    ):
        raise ValueError("vit_layers must select valid vision hidden-state layers")
    if connector.get("num_input_layers") != len(layers):
        raise ValueError("connector.num_input_layers must match vit_layers")
    if connector.get("image_emb_dim") != vision.get("image_emb_dim"):
        raise ValueError("Vision and connector feature dimensions differ")
    if connector.get("output_dim") != field(text, "hidden_size"):
        raise ValueError("Connector output_dim must match text hidden_size")
    if (
        connector.get("pooling_type") != "attention_meanq"
        or connector.get("projector_type") != "mlp"
    ):
        raise ValueError(
            "Only attention_meanq pooling with an MLP connector is supported"
        )
    if not connector.get("pooling_attention_mask"):
        raise ValueError("The supported connector requires pooling_attention_mask")


def validate_image_tokenizer(config: Any, tokenizer: Any) -> dict[str, int]:
    """Verify actual token IDs, never accepting the unknown-token fallback."""
    validate_vision_config(config)
    vocab = tokenizer.get_vocab()
    missing = set(IMAGE_TOKENS) - vocab.keys()
    if missing:
        raise ValueError(f"Tokenizer is missing image tokens: {sorted(missing)}")
    ids = {name: vocab[name] for name in IMAGE_TOKENS}
    limit = field(field(config, "text_config"), "vocab_size")
    if len(set(ids.values())) != len(ids) or any(
        type(i) is not int or not 0 <= i < limit for i in ids.values()
    ):
        raise ValueError(
            "Image token IDs must be unique and within the embedding table"
        )
    if ids["<im_patch>"] != field(config, "image_patch_token_id"):
        raise ValueError(
            "Tokenizer <im_patch> ID differs from checkpoint image_patch_token_id"
        )
    if max(vocab.values()) >= limit:
        raise ValueError("Tokenizer vocabulary exceeds the checkpoint embedding table")
    return ids
