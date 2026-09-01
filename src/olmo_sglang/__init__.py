# SPDX-License-Identifier: Apache-2.0

"""Out-of-tree SGLang support for Olmo models."""

from olmo_sglang.registration import MODEL_ARCHITECTURE, MODEL_PACKAGE, register

__version__ = "0.1.0"

__all__ = ["MODEL_ARCHITECTURE", "MODEL_PACKAGE", "__version__", "register"]
