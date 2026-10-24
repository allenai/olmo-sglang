"""Default-off simple sampler optimization for the pinned SGLang runtime."""

import os
from importlib import import_module, metadata

import torch

SAMPLER_VARIABLE = "OLMO_SGLANG_SAMPLER"
_SUPPORTED_VERSION = "0.5.19.dev49+g3145136"


def sampler_mode():
    value = os.environ.get(SAMPLER_VARIABLE, "torch").lower().strip()
    if value not in {"torch", "fused"}:
        raise ValueError(f"Invalid {SAMPLER_VARIABLE}: {value!r}")
    return value


def supported(probabilities):
    return (
        probabilities.is_cuda
        and probabilities.dtype == torch.float32
        and probabilities.ndim == 2
        and probabilities.is_contiguous()
        and probabilities.shape[0] > 0
        and probabilities.shape[1] > 0
    )


def sample(probabilities):
    """Keep Torch's exponential RNG, then combine division, selection and validation."""
    if not supported(probabilities):
        return torch.multinomial(probabilities, num_samples=1).view(-1).to(torch.int32)
    kernels = import_module("olmo_sglang.sampling_kernels")
    noise = torch.empty_like(probabilities).exponential_(1)
    output, valid = kernels.select_with_noise(probabilities, noise)
    torch._assert_async(
        valid.all(), "probability tensor contains invalid values or a zero-sum row"
    )
    return output


def _sample_from_probs(self, probs, sampling_info, positions, simple_sampling_case):
    if (
        simple_sampling_case
        and sampling_info.sampling_seed is None
        and supported(probs)
    ):
        return sample(probs)
    return self._olmo_original_sample_from_probs(
        probs, sampling_info, positions, simple_sampling_case
    )


def install(*, tp_size, backend):
    """Install in an Olmo model worker before serving; restart to change modes."""
    if sampler_mode() != "fused" or tp_size != 1 or backend != "pytorch":
        return False
    if metadata.version("sglang") != _SUPPORTED_VERSION:
        raise RuntimeError(f"Fused sampler requires pinned SGLang {_SUPPORTED_VERSION}")
    sampler = import_module("sglang.srt.layers.sampler")
    if not hasattr(sampler.Sampler, "_olmo_original_sample_from_probs"):
        sampler.Sampler._olmo_original_sample_from_probs = (
            sampler.Sampler._sample_from_probs
        )
        sampler.Sampler._sample_from_probs = _sample_from_probs
    return True
