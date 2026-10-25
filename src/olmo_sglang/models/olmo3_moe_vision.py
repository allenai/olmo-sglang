"""Checkpoint-loading entry point for Olmo's Core vision + KDA MoE exports.

Image request processing and feature injection are deliberately fail-closed
until the serving integration is qualified (oe-planning #65).
"""

import importlib
from collections.abc import Iterable

import torch
from torch import nn

from olmo_sglang.models.olmo3_moe import Olmo3MoeForCausalLM
from olmo_sglang.vision.config import validate_vision_config
from olmo_sglang.vision.weights import validate_language_coverage
from olmo_sglang.weight_loading import STACKED_PARAMS_MAPPING, WeightTargets


class Olmo3MoeVisionForConditionalGeneration(nn.Module):
    """Load all three model components, including the trained visual parameters."""

    fall_back_to_pt_during_load = False

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        validate_vision_config(config)
        if quant_config is not None:
            raise NotImplementedError(
                "Olmo vision checkpoint loading is not qualified for quantized weights"
            )
        self.config = config
        # Keep Core optional for the existing text-only model registry. SGLang
        # imports every model module during discovery, including unused models.
        vision_module = importlib.import_module("olmo_sglang.vision.model")
        stack = vision_module.OlmoVisionStack(
            config, device=torch.empty(0).device, dtype=torch.get_default_dtype()
        )
        self.vision_tower = stack.vision_tower
        self.connector = stack.connector
        lm_prefix = f"{prefix}.language_model" if prefix else "language_model"
        self.language_model = Olmo3MoeForCausalLM(config.text_config, prefix=lm_prefix)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_attention_sliding_window_size(self):
        return self.language_model.get_attention_sliding_window_size()

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        if input_embeds is not None or any(
            getattr(forward_batch, "mm_inputs", None) or []
        ):
            raise NotImplementedError(
                "Image request processing and post-normalization feature injection "
                "are not yet qualified; see oe-planning #65"
            )
        return self.language_model(input_ids, positions, forward_batch)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Stream LM tensors to its native loader; load all vision tensors strictly."""
        visual = {
            name: p
            for name, p in self.named_parameters()
            if name.startswith(("vision_tower.", "connector."))
        }
        seen, loaded_visual = set(), set()
        language_parameters = dict(self.language_model.named_parameters())
        resolver = WeightTargets(
            language_parameters,
            self.language_model.config.n_routed_experts,
            fused_layers={
                name: None
                for name in language_parameters
                if name.endswith(("experts.w13_weight", "experts.w2_weight"))
            },
        )
        language_targets = set()

        def language_weights():
            for item in weights:
                name, tensor, *extra = item
                if name in seen:
                    raise ValueError(f"Duplicate multimodal checkpoint tensor: {name}")
                seen.add(name)
                if name.startswith("language_model."):
                    local_name = name.removeprefix("language_model.")
                    target = resolver.resolve(local_name)
                    if target is None:
                        raise ValueError(
                            f"Unexpected language checkpoint tensor: {name}"
                        )
                    if target in language_targets:
                        raise ValueError(
                            f"Overlapping language checkpoint writes: {name}"
                        )
                    language_targets.add(target)
                    yield (local_name, tensor, *extra)
                elif name in visual:
                    parameter = visual[name]
                    if (
                        extra
                        or parameter.shape != tensor.shape
                        or not tensor.is_floating_point()
                    ):
                        raise ValueError(
                            f"Invalid visual tensor shape or shard metadata: {name}"
                        )
                    with torch.no_grad():
                        parameter.copy_(tensor)
                    loaded_visual.add(name)
                else:
                    raise ValueError(f"Unexpected multimodal checkpoint tensor: {name}")

        loaded_lm = self.language_model.load_weights(language_weights())
        validate_language_coverage(
            language_parameters,
            language_targets,
            STACKED_PARAMS_MAPPING,
            self.language_model.config.n_routed_experts,
        )
        expected_lm = set(dict(self.language_model.named_parameters()))
        if missing_lm := expected_lm - loaded_lm:
            raise ValueError(
                f"Missing language checkpoint parameters: {sorted(missing_lm)}"
            )
        if missing_visual := visual.keys() - loaded_visual:
            raise ValueError(
                f"Missing vision checkpoint parameters: {sorted(missing_visual)}"
            )
        return loaded_visual | {"language_model." + name for name in loaded_lm}


EntryClass = Olmo3MoeVisionForConditionalGeneration
