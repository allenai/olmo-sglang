"""Checkpoint-loading entry point for Olmo's Core vision + KDA MoE exports.

Image preprocessing uses Core; feature injection preserves its post-normalization
addition and SGLang handles hashed image tokens and chunked feature placement.
"""

import importlib
import sys
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
        self.vit_layers = stack.vit_layers
        lm_prefix = f"{prefix}.language_model" if prefix else "language_model"
        self.language_model = Olmo3MoeForCausalLM(config.text_config, prefix=lm_prefix)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_attention_sliding_window_size(self):
        return self.language_model.get_attention_sliding_window_size()

    @classmethod
    def get_placeholder_str(cls, modality, i):
        if modality != "image":
            raise NotImplementedError("Olmo vision supports still images only")
        return "<|image|>"

    def pad_input_ids(self, input_ids, mm_inputs):
        utils = importlib.import_module("sglang.srt.managers.mm_utils")
        return utils.MultiModalityDataPaddingPatternMultimodalTokens().pad_input_tokens(
            input_ids, mm_inputs
        )

    def get_image_feature(self, items):
        vision_module = importlib.import_module("olmo_sglang.vision.model")
        parameter = next(self.vision_tower.parameters())
        features = []
        for item in items:
            if item.precomputed_embeddings is not None:
                raise NotImplementedError(
                    "Precomputed image embeddings are not qualified"
                )
            pixels = item.feature.to(device=parameter.device, dtype=parameter.dtype)
            pools = item.pooling_indices.to(parameter.device)
            with torch.profiler.record_function("olmo.vision_encoder_connector"):
                encoded = vision_module.OlmoVisionStack.encode_images(
                    self, pixels, pools
                )
            encoded = encoded.reshape(-1, encoded.shape[-1])
            count = sum(end - start + 1 for start, end in item.offsets)
            if encoded.shape[0] != count:
                raise ValueError("Image features and patch-token positions disagree")
            patch_ids = torch.full(
                (count,),
                self.config.image_patch_token_id,
                dtype=torch.long,
                device=parameter.device,
            )
            base = self.language_model.get_normalized_embeddings(patch_ids)
            # Core: embedding -> scale -> norm -> add, including the trained
            # patch-token embedding. Generic replacement alone would drop it.
            features.append(base + encoded.to(base.dtype))
        return torch.cat(features, dim=0)

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        if input_embeds is not None:
            raise NotImplementedError(
                "Externally supplied vision embeddings are not qualified"
            )
        mm_inputs = getattr(forward_batch, "mm_inputs", None)
        if mm_inputs and any(mm_inputs) and not forward_batch.forward_mode.is_decode():
            if forward_batch.forward_mode.is_target_verify():
                raise NotImplementedError(
                    "Speculative image inference is not qualified"
                )
            utils = importlib.import_module("sglang.srt.managers.mm_utils")
            selected = [i for i, value in enumerate(mm_inputs) if value is not None]
            with torch.profiler.record_function("olmo.image_embedding_and_placement"):
                embeds, _ = utils.embed_mm_inputs(
                    mm_inputs_list=[mm_inputs[i] for i in selected],
                    extend_prefix_lens=[
                        forward_batch.extend_prefix_lens_cpu[i] for i in selected
                    ],
                    extend_seq_lens=[
                        forward_batch.extend_seq_lens_cpu[i] for i in selected
                    ],
                    input_ids=input_ids,
                    input_embedding=_NormalizedEmbedding(self.language_model),
                    multimodal_model=self,
                )
            # Retain a CPU fallback for future chunks if the embedding cache evicts.
            for i in selected:
                for item in mm_inputs[i].mm_items:
                    if isinstance(item.feature, torch.Tensor) and item.feature.is_cuda:
                        item.feature = item.feature.to("cpu", non_blocking=True)
            forward_batch.mm_inputs = None
            forward_batch.mm_input_embeds = embeds
            return self.language_model(
                input_ids,
                positions,
                forward_batch,
                input_embeds=embeds,
                input_embeds_normalized=True,
            )
        return self.language_model(input_ids, positions, forward_batch)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Stream LM tensors to its native loader; load all vision tensors strictly."""
        # Image features also contain learned patch embeddings and their norm.
        # Their content-only cache keys survive SGLang's KV/KDA flush. Invalidate
        # before any writes, including reloads that fail after a partial update.
        # Initial loading can precede cache initialization; avoid importing the
        # scheduler (and its GPU dependencies) just to discover an absent cache.
        schedule = sys.modules.get("sglang.srt.managers.mm_schedule")
        cache = getattr(schedule, "embedding_cache", None)
        if cache is not None:
            cache.clear()
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


class _NormalizedEmbedding:
    """SGLang's embedding interface without re-registering shared LM parameters."""

    def __init__(self, language_model):
        self.language_model = language_model
        self.num_embeddings = language_model.get_input_embeddings().num_embeddings

    def __call__(self, input_ids):
        return self.language_model.get_normalized_embeddings(input_ids)


EntryClass = Olmo3MoeVisionForConditionalGeneration
