"""Core image preprocessing and token expansion, independent of SGLang/CUDA."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from olmo_core.nn.vision.molmo2_image_processor import preprocess_image_molmo2
from olmo_core.nn.vision.molmo2_tokens import Molmo2TokenIds, build_image_token_ids
from PIL import Image
from transformers import AutoTokenizer, PretrainedConfig, ProcessorMixin

from olmo_sglang.vision.config import field, validate_image_tokenizer


@dataclass
class PreparedImage:
    patches: torch.Tensor
    pooling: torch.Tensor
    token_ids: list[int]
    offsets: list[tuple[int, int]]
    content_hash: int


def patch_offsets(ids, patch_id, start=0):
    """Inclusive contiguous patch-token runs; structural tokens remain text."""
    offsets = []
    for index, token in enumerate(ids):
        if token != patch_id:
            continue
        position = start + index
        if offsets and offsets[-1][1] == position - 1:
            offsets[-1] = (offsets[-1][0], position)
        else:
            offsets.append((position, position))
    return offsets


class OlmoVisionProcessor(ProcessorMixin):
    """Tokenizer plus the pinned Core SigLIP multi-crop image recipe.

    Accepts one explicit <|image|> placeholder per PIL image. Existing tokenized
    prompts are expanded directly, never decoded and re-tokenized. This first
    implementation supports the inspected 378/14 SigLIP checkpoints only.
    """

    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"

    def __init__(self, config, tokenizer):
        super().__init__(tokenizer=tokenizer)
        self.config = config
        ids = validate_image_tokenizer(config, tokenizer)
        end_turn = tokenizer.get_vocab().get("<|im_end|>")
        if end_turn is None:
            raise ValueError("Tokenizer is missing required <|im_end|> token")
        self.token_ids = Molmo2TokenIds(
            im_start_id=ids["<im_start>"],
            im_end_id=ids["<im_end>"],
            im_patch_id=ids["<im_patch>"],
            im_col_id=ids["<im_col>"],
            low_res_im_start_id=ids["<low_res_im_start>"],
            image_placeholder_id=ids["<|image|>"],
            im_end_turn_id=end_turn,
        )
        vision = field(config, "vision_config")
        if (
            vision.get("image_default_input_size") != [378, 378]
            or vision.get("image_patch_size") != 14
            or vision.get("image_num_pos") != 729
        ):
            raise ValueError(
                "Image processing currently requires the qualified 378/14 SigLIP layout"
            )
        recipe = field(config, "olmo_core_config", {}).get("recipe", {})
        self.max_crops = recipe.get("max_crops")
        if type(self.max_crops) is not int or not 1 <= self.max_crops <= 24:
            raise ValueError("Checkpoint must specify recipe.max_crops in [1, 24]")

    @classmethod
    def from_pretrained(cls, path, *args, **kwargs):
        # Vision checkpoints currently live in local/WEKA directories. Reuse the
        # trained tokenizer without invoking a text-only remote model processor.
        config = json.loads((Path(path) / "config.json").read_text())
        tokenizer = AutoTokenizer.from_pretrained(
            path,
            config=PretrainedConfig(),
            local_files_only=True,
            trust_remote_code=False,
        )
        return cls(config, tokenizer)

    def document_prompt_ids(self, text, num_images=1):
        """Core stage-1 document layout: EOS, image placeholders, plain task text."""
        if type(num_images) is not int or num_images < 0:
            raise ValueError("num_images must be a nonnegative integer")
        if self.tokenizer.eos_token_id is None:
            raise ValueError("The document layout requires an EOS token")
        return [
            self.tokenizer.eos_token_id,
            *([self.token_ids.image_placeholder_id] * num_images),
            *self.tokenizer.encode(text, add_special_tokens=False),
        ]

    def prepare(self, prompt, images):
        ids = (
            list(prompt) if isinstance(prompt, list) else self.tokenizer.encode(prompt)
        )
        placeholder = self.token_ids.image_placeholder_id
        if ids.count(placeholder) != len(images):
            raise ValueError("Supply exactly one <|image|> placeholder per image")
        structural = self.token_ids.image_token_ids
        if any(token in structural for token in ids):
            raise ValueError(
                "Provide image placeholders, not manually expanded structural image tokens"
            )
        result, prepared = [], []
        image_iter = iter(images)
        for token in ids:
            if token != placeholder:
                result.append(token)
                continue
            image = next(image_iter)
            if not isinstance(image, Image.Image):
                raise NotImplementedError(
                    "Only decoded still images are supported; precomputed inputs are not qualified"
                )
            patches, pooling, grid = preprocess_image_molmo2(
                image,
                dtype=torch.float32,
                device=torch.device("cpu"),
                max_crops=self.max_crops,
                is_training=False,
            )
            image_ids = build_image_token_ids(*map(int, grid), token_ids=self.token_ids)
            count = image_ids.count(self.token_ids.im_patch_id)
            if count != pooling.shape[1] or not (pooling >= 0).any(-1).all():
                raise ValueError(
                    "Core image token count and valid pooling rows disagree"
                )
            offsets = patch_offsets(image_ids, self.token_ids.im_patch_id, len(result))
            digest = hashlib.sha256()
            # Hash pooling/layout as well as pixels; both affect image features.
            for tensor in (
                patches,
                pooling,
                torch.tensor(image_ids, dtype=torch.int64),
            ):
                digest.update(str((tensor.shape, tensor.dtype)).encode())
                digest.update(tensor.contiguous().numpy().tobytes())
            prepared.append(
                PreparedImage(
                    patches,
                    pooling,
                    image_ids,
                    offsets,
                    int.from_bytes(digest.digest()[:8], "little"),
                )
            )
            result.extend(image_ids)
        return result, prepared
