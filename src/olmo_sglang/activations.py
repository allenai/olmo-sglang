# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Activation adapters for OLMo shapes unsupported by older SGLang kernels."""

import torch
import torch.nn.functional as F


def native_silu_and_mul(gate_up: torch.Tensor) -> torch.Tensor:
    """Apply SwiGLU without SGLang's vector-width-constrained JIT kernel."""

    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up
