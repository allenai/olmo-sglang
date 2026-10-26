"""Reuse Core's vision implementation and exact parameter layout.

This module is deliberately independent of SGLang so loading and small feature
comparisons can run on a CPU. Install the qualified Core vision revision first.
"""

import copy
from collections.abc import Iterable

import torch
from olmo_core.nn.vision.config import VisionEncoderConfig
from olmo_core.nn.vision.connector import VisionConnectorConfig
from torch import nn

from olmo_sglang.vision.config import field, validate_vision_config


class OlmoVisionStack(nn.Module):
    """The trained SigLIP tower and connector, under their HF export names."""

    def __init__(self, config, *, device="cpu", dtype=torch.float32):
        super().__init__()
        validate_vision_config(config)
        self.config = config
        self.vit_layers = tuple(field(config, "vit_layers"))
        vision = copy.deepcopy(field(config, "vision_config"))
        connector = copy.deepcopy(field(config, "connector_config"))
        vision["dtype"] = connector["dtype"] = str(dtype).removeprefix("torch.")
        self.vision_tower = VisionEncoderConfig.from_dict(vision).build(
            init_device=device
        )
        self.connector = VisionConnectorConfig.from_dict(connector).build(
            init_device=device
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load just the vision stack, rejecting missing/duplicate/extra tensors.

        Core ``vision.*`` and HF ``vision_tower.*`` are equivalent namespaces.
        Language-model tensors must be explicitly separated by the caller.
        """
        parameters = dict(self.named_parameters())
        loaded = set()
        for name, tensor in weights:
            if name.startswith("vision."):
                name = "vision_tower." + name.removeprefix("vision.")
            if name not in parameters:
                raise ValueError(f"Unexpected vision checkpoint tensor: {name}")
            if name in loaded:
                raise ValueError(f"Duplicate vision checkpoint tensor: {name}")
            parameter = parameters[name]
            if not tensor.is_floating_point():
                raise ValueError(
                    f"Vision checkpoint tensor must be floating point: {name}"
                )
            if parameter.shape != tensor.shape:
                raise ValueError(
                    f"Shape mismatch for {name}: {tensor.shape} != {parameter.shape}"
                )
            if parameter.is_meta:
                raise ValueError("Materialize the vision stack before loading weights")
            with torch.no_grad():
                parameter.copy_(tensor)
            loaded.add(name)
        missing = parameters.keys() - loaded
        if missing:
            raise ValueError(f"Missing vision checkpoint tensors: {sorted(missing)}")
        return loaded

    def encode_images(self, images, pooled_patches_idx):
        """Map pre-patchified crops [B,T,N,P] to LM features [B,pooled,D].

        Raw image preprocessing and serving request integration belong to #65.
        Padded pooling rows are preserved, matching Core's connector contract.
        """
        if images.ndim != 4 or pooled_patches_idx.ndim != 3:
            raise ValueError(
                "Expected images [B,T,N,P] and pooling indices [B,pooled,K]"
            )
        b, crops, patches, width = images.shape
        if not b or not crops or not patches or pooled_patches_idx.shape[0] != b:
            raise ValueError("Image and pooling batch sizes must agree and be nonempty")
        if pooled_patches_idx.dtype != torch.long:
            raise ValueError("Pooling indices must have dtype int64")
        if bool(
            ((pooled_patches_idx < -1) | (pooled_patches_idx >= crops * patches)).any()
        ):
            raise ValueError("Pooling index outside the crop/patch range")
        hidden = self.vision_tower(images.reshape(b * crops, patches, width))
        selected = [hidden[i] for i in self.vit_layers]
        features = torch.cat(selected, dim=-1) if len(selected) > 1 else selected[0]
        features = features.reshape(b, crops * patches, features.shape[-1])
        return self.connector(features, pooled_patches_idx)
