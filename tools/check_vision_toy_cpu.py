"""Real SigLIP + connector, native MM placement, and a tiny CPU hybrid MoE.

Only the CUDA text executor is substituted. The actual vision wrapper's feature
and forward methods and SGLang's hashing/chunk-placement/cache code run on CPU.
The two-layer decoder is synthetic; this is plumbing validation, not accuracy.
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
from olmo_core.nn.vision.molmo2_tokens import prepare_molmo2_tokenizer
from PIL import Image, ImageDraw
from safetensors import safe_open
from sglang.srt.managers import mm_schedule
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.runtime_context import get_parallel
from transformers import AutoTokenizer, PretrainedConfig

from olmo_sglang.validation.reference import ToyReferenceForCausalLM
from olmo_sglang.vision.model import OlmoVisionStack
from olmo_sglang.vision.processing import OlmoVisionProcessor
from tools.create_tiny_parity_checkpoint import _config


class ToyExecutor(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.reference = ToyReferenceForCausalLM(SimpleNamespace(**config)).eval()
        for parameter in self.reference.parameters():
            torch.nn.init.normal_(parameter, std=0.02)
        self.config = self.reference.config
        self.last_embeddings = None

    def get_input_embeddings(self):
        return self.reference.model.embed_tokens

    def get_normalized_embeddings(self, ids):
        model = self.reference.model
        return model.embed_norm(model.embed_tokens(ids) * self.config.embed_scale)

    def forward(
        self, ids, positions, batch, input_embeds=None, input_embeds_normalized=False
    ):
        hidden = (
            self.get_normalized_embeddings(ids)
            if input_embeds is None
            else input_embeds
        )
        assert input_embeds is None or input_embeds_normalized
        self.last_embeddings = hidden.clone()
        hidden = hidden.unsqueeze(0)
        for i, layer in enumerate(self.reference.model.layers):
            hidden = layer(hidden, trace=None, prefix=f"model.layers.{i}")
        return self.reference.lm_head(self.reference.model.norm(hidden[:, -1:])).float()


def load_wrapper_class():
    # Scope the substitution to this test module import; normal serving imports
    # the real native decoder. All image-side wrapper methods are unchanged.
    substitute = ModuleType("olmo_sglang.models.olmo3_moe")
    substitute.Olmo3MoeForCausalLM = ToyExecutor
    path = Path(__file__).parents[1] / "src/olmo_sglang/models/olmo3_moe_vision.py"
    spec = importlib.util.spec_from_file_location("olmo_vision_cpu_probe", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {substitute.__name__: substitute}):
        spec.loader.exec_module(module)
    return module.Olmo3MoeVisionForConditionalGeneration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(65)
    started = time.perf_counter()
    root = Path(args.fixture)
    config = json.loads((root / "hf-step5000-alt/config.json").read_text())
    config["text_config"]["vocab_size"] = 100352
    tokenizer = AutoTokenizer.from_pretrained(
        root / "hf-step5000-alt",
        config=PretrainedConfig(),
        trust_remote_code=False,
        local_files_only=True,
    )
    prepare_molmo2_tokenizer(tokenizer, model_vocab_size=100352)
    processor = OlmoVisionProcessor(config, tokenizer)
    visual = OlmoVisionStack(config).eval()
    tiny = _config("hybrid-moe", tokenizer, 2048)
    tiny.update(hidden_size=1024, vocab_size=100352, embed_scale=32.0)
    language = ToyExecutor(tiny)
    with safe_open(
        root / "vision_and_embeddings.safetensors", framework="pt", device="cpu"
    ) as handle:
        loaded = visual.load_weights(
            (name, handle.get_tensor(name))
            for name in handle.keys()
            if name.startswith(("vision.", "connector."))
        )
        with torch.no_grad():
            language.reference.model.embed_tokens.weight.copy_(
                handle.get_tensor("lm.embeddings.weight")
            )
            language.reference.lm_head.weight.copy_(
                handle.get_tensor("lm.lm_head.w_out.weight")
            )
            language.reference.model.embed_norm.weight.copy_(
                handle.get_tensor("lm.embedding_norm.weight")
            )
    cls = load_wrapper_class()
    wrapper = cls.__new__(cls)
    torch.nn.Module.__init__(wrapper)
    wrapper.config = SimpleNamespace(**config)
    wrapper.vision_tower, wrapper.connector = visual.vision_tower, visual.connector
    wrapper.vit_layers, wrapper.language_model = visual.vit_layers, language
    mm_schedule.init_mm_embedding_cache(128 * 1024 * 1024)
    mode = SimpleNamespace(is_decode=lambda: False, is_target_verify=lambda: False)
    prompt = [
        tokenizer.eos_token_id,
        processor.token_ids.image_placeholder_id,
        *tokenizer.encode("olmocr:", add_special_tokens=False),
    ]
    records, outputs = [], []
    with torch.inference_mode(), get_parallel().override(attn_tp_rank=0, tp_size=1):
        for color in ("white", "black"):
            image = Image.new("RGB", (220, 180), color)
            ImageDraw.Draw(image).text((20, 60), "VISION 65", fill="red")
            ids, prepared = processor.prepare(prompt, [image])
            item = prepared[0]
            mm_item = MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=item.patches,
                offsets=item.offsets,
                model_specific_data={"pooling_indices": item.pooling},
            )
            mm_item.set_hash(item.content_hash)
            mm = MultimodalInputs(
                mm_items=[mm_item], im_token_id=processor.token_ids.im_patch_id
            )
            padded = wrapper.pad_input_ids(ids, mm)
            expected = language.get_normalized_embeddings(torch.tensor(ids))
            feature = visual.encode_images(item.patches, item.pooling).reshape(-1, 1024)
            expected[torch.tensor(ids) == processor.token_ids.im_patch_id] += feature
            batch = SimpleNamespace(
                mm_inputs=[mm],
                forward_mode=mode,
                extend_prefix_lens_cpu=[0],
                extend_seq_lens_cpu=[len(ids)],
            )
            start = time.perf_counter()
            logits = wrapper(torch.tensor(padded), torch.arange(len(ids)), batch)
            full = language.last_embeddings
            assert torch.equal(full, expected)
            outputs.append(logits)
            # Exercise actual upstream chunk slicing and its embedding-cache hit.
            chunks = []
            for offset in range(0, len(ids), 97):
                n = min(97, len(ids) - offset)
                batch = SimpleNamespace(
                    mm_inputs=[mm],
                    forward_mode=mode,
                    extend_prefix_lens_cpu=[offset],
                    extend_seq_lens_cpu=[n],
                )
                wrapper(
                    torch.tensor(padded[offset : offset + n]),
                    torch.arange(offset, offset + n),
                    batch,
                )
                chunks.append(language.last_embeddings)
            assert torch.equal(torch.cat(chunks), expected)
            records.append(
                {
                    "image": color,
                    "hash": item.content_hash,
                    "tokens": len(ids),
                    "crops": item.patches.shape[1],
                    "max_embedding_error": float((full - expected).abs().max()),
                    "chunked_embeddings_exact": True,
                    "seconds": time.perf_counter() - start,
                }
            )
    assert not torch.equal(outputs[0], outputs[1])
    report = {
        "passed": True,
        "real_visual_tensors": len(loaded),
        "records": records,
        "image_swap_changes_toy_logits": True,
        "seconds": time.perf_counter() - started,
        "scope": "Real vision/connector and embedding weights; synthetic two-layer KDA/MoE CPU decoder; native SGLang image placement and chunk/cache machinery",
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
