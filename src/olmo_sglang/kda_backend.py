"""Compatibility import for the historical KDA backend module path.

The canonical implementation lives in :mod:`olmo_sglang.kda.backend`. This
module remains because olmo-miles preflight imports it directly.
"""

from olmo_sglang.kda.backend import (
    OlmoFLAKDAKernel,
    OlmoKDAAttnBackend,
    OlmoKDACacheParams,
    OlmoKDAStateShape,
    OlmoPackedKDAKernel,
    register_olmo_kda_backend,
    require_fla_0_5_2,
)

__all__ = [
    "OlmoFLAKDAKernel",
    "OlmoKDAAttnBackend",
    "OlmoKDACacheParams",
    "OlmoKDAStateShape",
    "OlmoPackedKDAKernel",
    "register_olmo_kda_backend",
    "require_fla_0_5_2",
]
