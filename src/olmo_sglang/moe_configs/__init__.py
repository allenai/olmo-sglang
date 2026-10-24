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


def install_config_dir(config: Any) -> str | None:
    """Point SGLang at the packaged configurations unless the caller chose a directory.

    SGLang replaces its lookup root with the variable rather than merging, so this
    is only installed for the qualified 512-expert, 1024-intermediate,
    512-input shape. Other OLMo shapes retain their upstream lookup root.
    The lookup is cached on first use, so this must run before the first MoE forward;
    the OLMo configuration hook runs in every process when the model config is built.
    """

    global _ANNOUNCED
    existing = os.environ.get(ENVIRONMENT_VARIABLE)
    if existing:
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
