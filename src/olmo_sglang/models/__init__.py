# SPDX-License-Identifier: Apache-2.0

"""External model modules discovered by SGLang's model registry."""

from olmo_sglang.activations import install_sglang_moe_activation_fallback
from olmo_sglang.kda.backend import register_olmo_kda_backend
from olmo_sglang.policy_cache import register as register_policy_cache

register_olmo_kda_backend()
install_sglang_moe_activation_fallback()

register_policy_cache()
