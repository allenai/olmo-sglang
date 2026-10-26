"""SGLang transport adapter for the Core-compatible Olmo image processor."""

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.multimodal.customized_mm_processor_utils import (
    register_customized_processor,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from transformers import PretrainedConfig

from olmo_sglang.models.olmo3_moe_vision import Olmo3MoeVisionForConditionalGeneration
from olmo_sglang.vision.processing import OlmoVisionProcessor


@register_customized_processor(OlmoVisionProcessor)
class _ProcessorConfig(PretrainedConfig):
    model_type = "olmo3moe_vision"


class OlmoImageProcessor(BaseMultimodalProcessor):
    models = [Olmo3MoeVisionForConditionalGeneration]
    gpu_image_decode = False
    preserve_processor_input_ids = True

    def __init__(self, hf_config, server_args, processor, *args, **kwargs):
        if server_args.tp_size != 1 or server_args.speculative_algorithm:
            raise NotImplementedError(
                "Initial Olmo image inference requires TP=1 and no speculation"
            )
        super().__init__(hf_config, server_args, processor, *args, **kwargs)
        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<|image|>",
            image_token_id=processor.token_ids.image_placeholder_id,
        ).build(processor)

    async def process_mm_data_async(
        self,
        image_data,
        input_text,
        request_obj,
        audio_data=None,
        video_data=None,
        **kwargs,
    ):
        if audio_data or video_data:
            raise NotImplementedError(
                "Olmo vision currently supports still images only"
            )
        # Reject mismatched placeholders before the base loader can auto-insert one.
        ids = (
            list(input_text)
            if isinstance(input_text, list)
            else self._tokenizer.encode(input_text)
        )
        if ids.count(self._processor.token_ids.image_placeholder_id) != len(
            image_data or []
        ):
            raise ValueError("Supply exactly one <|image|> placeholder per image")
        # The base loader needs text only to match media to placeholders. Give
        # it that inventory, preserving the original IDs exclusively for prepare.
        # Passing the tokenized prompt would decode it unnecessarily (and setting
        # return_text=False alone asserts in the pinned SGLang loader).
        loaded = await self.load_mm_data(
            prompt=self.mm_tokens.image_token * len(image_data or []),
            multimodal_tokens=self.mm_tokens,
            image_data=image_data,
            return_text=False,
        )
        input_ids, images = self._processor.prepare(ids, loaded.images)
        items = []
        for image in images:
            item = MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=image.patches,
                offsets=image.offsets,
                model_specific_data={"pooling_indices": image.pooling},
            )
            item.set_hash(image.content_hash)
            items.append(item)
        return MultimodalProcessorOutput(
            mm_items=items,
            input_ids=input_ids,
            im_token_id=self._processor.token_ids.im_patch_id,
        )
