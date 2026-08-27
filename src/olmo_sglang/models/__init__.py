# SPDX-License-Identifier: Apache-2.0

"""External model modules discovered by SGLang's model registry."""

from olmo_sglang.activations import install_sglang_moe_activation_fallback
from olmo_sglang.kda_backend import register_olmo_kda_backend

register_olmo_kda_backend()
install_sglang_moe_activation_fallback()
