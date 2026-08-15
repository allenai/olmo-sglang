# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public registration API for SGLang model discovery."""

from __future__ import annotations

import os


MODEL_ARCHITECTURE = "Olmo3MoeForCausalLM"
MODEL_PACKAGE = "olmo_sglang.models"
_EXTERNAL_PACKAGE_ENV = "SGLANG_EXTERNAL_MODEL_PACKAGE"


def register() -> None:
    """Register the OLMo model in this process and spawned SGLang workers.

    Raises:
        RuntimeError: If the environment already selects a different external
            SGLang model package.
    """

    configured_package = os.environ.get(_EXTERNAL_PACKAGE_ENV)
    if configured_package not in (None, "", MODEL_PACKAGE):
        raise RuntimeError(
            f"{_EXTERNAL_PACKAGE_ENV} already selects {configured_package!r}; cannot also select {MODEL_PACKAGE!r}"
        )
    os.environ[_EXTERNAL_PACKAGE_ENV] = MODEL_PACKAGE

    try:
        from sglang.srt.models.registry import ModelRegistry
    except ImportError as error:
        raise RuntimeError(
            "SGLang is required to register olmo-sglang models. Install the "
            "SGLang runtime before calling olmo_sglang.register()."
        ) from error

    from olmo_sglang.kda_backend import register_olmo_kda_backend

    register_olmo_kda_backend()
    ModelRegistry.register(MODEL_PACKAGE, overwrite=True, strict=True)
