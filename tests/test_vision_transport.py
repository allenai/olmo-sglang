"""Exercise the adapter's prompt/media boundary without a GPU runtime."""

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.mark.parametrize("pretokenized", [False, True])
def test_media_loading_preserves_prompt_ids_without_decoding(monkeypatch, pretokenized):
    modules = {
        "sglang.srt.managers.schedule_batch": {
            "Modality": SimpleNamespace(IMAGE="image"),
            "MultimodalDataItem": SimpleNamespace,
            "MultimodalProcessorOutput": SimpleNamespace,
        },
        "sglang.srt.multimodal.customized_mm_processor_utils": {
            "register_customized_processor": lambda _: lambda cls: cls,
        },
        "sglang.srt.multimodal.processors.base_processor": {
            "BaseMultimodalProcessor": object,
            "MultimodalSpecialTokens": SimpleNamespace,
        },
        "olmo_sglang.models.olmo3_moe_vision": {
            "Olmo3MoeVisionForConditionalGeneration": object,
        },
        "olmo_sglang.vision.processing": {"OlmoVisionProcessor": object},
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).parents[1] / "src/olmo_sglang/processors/olmo_vision.py"
    spec = importlib.util.spec_from_file_location("vision_transport_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    processor = module.OlmoImageProcessor.__new__(module.OlmoImageProcessor)
    original_ids = [7, 99, 8, 99, 9]
    media = [object(), object()]

    def encode(text):
        assert not pretokenized
        assert text == "original prompt"
        return original_ids.copy()

    def decode(_):
        pytest.fail("Tokenized prompts must not be decoded to load images")

    async def load_mm_data(*, prompt, multimodal_tokens, image_data, return_text):
        # The pinned loader requires a string when return_text=False. It counts
        # placeholders before loading media, preserving their input ordering.
        assert isinstance(prompt, str)
        assert prompt.count(multimodal_tokens.image_token) == len(image_data)
        assert image_data == media
        assert return_text is False
        return SimpleNamespace(images=media)

    def prepare(ids, images):
        assert ids == original_ids
        assert images == media
        return ids, []

    processor._tokenizer = SimpleNamespace(encode=encode, decode=decode)
    processor._processor = SimpleNamespace(
        token_ids=SimpleNamespace(image_placeholder_id=99, im_patch_id=100),
        prepare=prepare,
    )
    processor.mm_tokens = SimpleNamespace(image_token="<|image|>")
    processor.load_mm_data = load_mm_data
    prompt = original_ids.copy() if pretokenized else "original prompt"
    output = asyncio.run(processor.process_mm_data_async(media, prompt, None))
    assert output.input_ids == original_ids
    with pytest.raises(ValueError, match="exactly one"):
        asyncio.run(processor.process_mm_data_async(media[:1], prompt, None))
