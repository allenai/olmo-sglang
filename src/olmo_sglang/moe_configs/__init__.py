# SPDX-License-Identifier: Apache-2.0

"""Tuned SGLang fused-MoE Triton kernel configurations for OLMo expert shapes.

SGLang looks for ``E={experts},N={intermediate},device_name={gpu}.json`` under
``$SGLANG_MOE_CONFIG_DIR/configs/triton_{version}/`` (falling back to the newest
version directory present) and otherwise uses an untuned default tile. The files
here were produced with SGLang's own tuner; ``tools/moe_tuning`` records how.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

ENVIRONMENT_VARIABLE = "SGLANG_MOE_CONFIG_DIR"
CONFIG_ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger(__name__)
_ANNOUNCED = False


def shipped_config_files() -> list[Path]:
    """Return every tuned configuration file packaged with the overlay."""

    return sorted((CONFIG_ROOT / "configs").glob("triton_*/E=*.json"))


def install_config_dir(
    config: Any,
    *,
    device_name: str,
    tp_size: int,
    ep_size: int,
    quantized: bool = False,
) -> str | None:
    """Point SGLang at the packaged configurations unless the caller chose a directory.

    SGLang replaces its lookup root with the variable rather than merging, so this
    is only installed for the qualified 512-expert, 1024-intermediate,
    512-input shape on B300 with TP1/EP1 and no quantization. Other layouts
    retain their upstream lookup root, including tuning for local TP dimensions.
    Run during model construction after device/parallel setup, before expert
    construction or the first cached MoE lookup.
    """

    global _ANNOUNCED
    existing = os.environ.get(ENVIRONMENT_VARIABLE)
    if existing:
        return None
    if (
        device_name != "NVIDIA B300 SXM6 AC"
        or tp_size != 1
        or ep_size != 1
        or quantized
    ):
        return None
    expert_hidden_size = getattr(config, "latent_moe_dim", None)
    if expert_hidden_size is None:
        expert_hidden_size = getattr(config, "hidden_size", None)
    shape = (
        getattr(config, "n_routed_experts", None),
        getattr(config, "moe_intermediate_size", None),
        expert_hidden_size,
    )
    if shape != (512, 1024, 512):
        return None
    os.environ[ENVIRONMENT_VARIABLE] = str(CONFIG_ROOT)
    if not _ANNOUNCED:
        LOGGER.info(
            "Using packaged fused-MoE kernel configurations from %s", CONFIG_ROOT
        )
        _ANNOUNCED = True
    return str(CONFIG_ROOT)
