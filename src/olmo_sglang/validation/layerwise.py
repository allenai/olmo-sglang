"""Diagnostic activation capture; never changes model math or qualification gates."""

from __future__ import annotations

from contextlib import contextmanager

import torch

# Both implementations use peri-norm; post-FFN norm inputs expose combined MLP
# outputs even though Core's fused MoE does not call a single MLP module.
BOUNDARIES = {
    "attention": ("self_attn", "attention"),
    "attention_pre_norm": ("pre_attention_layernorm", "attention_input_norm"),
    "attention_post_norm": ("post_attention_layernorm", "attention_norm"),
    "ffn_pre_norm": ("pre_feedforward_layernorm", "feed_forward_input_norm"),
    "ffn_post_norm": ("post_feedforward_layernorm", "feed_forward_norm"),
}


def tensor_value(value):
    if isinstance(value, (tuple, list)):
        value = value[0]
    if not isinstance(value, torch.Tensor):
        raise TypeError("Expected a tensor activation")
    return value.detach().cpu().clone()


@contextmanager
def capture_block(block, *, backend):
    if backend not in ("hf", "core"):
        raise ValueError("backend must be hf or core")
    captures, handles = {}, []

    def attach(module, name):
        def before(_module, args, kwargs):
            value = args[0] if args else kwargs["hidden_states"]
            captures[name + ".input"] = tensor_value(value)

        def after(_module, _args, output):
            captures[name + ".output"] = tensor_value(output)

        handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
        handles.append(module.register_forward_hook(after))

    attach(block, "block")
    for label, names in BOUNDARIES.items():
        module = getattr(block, names[backend == "core"], None)
        if module is None:
            raise ValueError(f"Missing required peri-norm boundary: {label}")
        attach(module, label)
    try:
        yield captures
    finally:
        for handle in handles:
            handle.remove()


def tensor_error(actual, expected):
    if actual.shape != expected.shape:
        raise ValueError(
            f"Activation shape mismatch: {actual.shape} != {expected.shape}"
        )
    a, b = actual.float(), expected.float()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Non-finite activation")
    error = (a - b).abs()
    return {
        "shape": list(a.shape),
        "actual_dtype": str(actual.dtype),
        "reference_dtype": str(expected.dtype),
        "max_abs": float(error.max()),
        "mean_abs": float(error.mean()),
        "relative_l2": float(torch.linalg.vector_norm(a - b))
        / max(float(torch.linalg.vector_norm(b)), 1e-30),
        "reference_rms": float(b.square().mean().sqrt()),
        "equal_fraction": float((a == b).float().mean()),
    }


def compare_captures(actual, expected):
    if actual.keys() != expected.keys():
        raise ValueError("Captured boundaries differ; refusing partial comparison")
    return {name: tensor_error(actual[name], value) for name, value in expected.items()}
