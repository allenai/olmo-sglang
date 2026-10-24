"""Sampler option gates, seeded fallbacks, RNG consumption and argmax semantics."""

from importlib import import_module
from types import SimpleNamespace

import pytest
import torch

from olmo_sglang import sampling


def test_sampler_mode_validation(monkeypatch):
    monkeypatch.delenv(sampling.SAMPLER_VARIABLE, raising=False)
    assert sampling.sampler_mode() == "torch"
    monkeypatch.setenv(sampling.SAMPLER_VARIABLE, " Fused ")
    assert sampling.sampler_mode() == "fused"
    monkeypatch.setenv(sampling.SAMPLER_VARIABLE, "unknown")
    with pytest.raises(ValueError, match=sampling.SAMPLER_VARIABLE):
        sampling.sampler_mode()


def test_cpu_sampling_retains_torch_rng_and_samples():
    probabilities = torch.tensor([[0.1, 0.2, 0.7], [0.8, 0.0, 0.2]])
    state = torch.get_rng_state()
    expected = torch.multinomial(probabilities, 1).view(-1).int()
    expected_state = torch.get_rng_state()
    torch.set_rng_state(state)
    actual = sampling.sample(probabilities)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.get_rng_state(), expected_state)


def test_sampler_install_scope_and_fallbacks(monkeypatch):
    class Sampler:
        def _sample_from_probs(self, *args):
            return "original"

    monkeypatch.setattr(
        sampling.metadata, "version", lambda name: sampling._SUPPORTED_VERSION
    )
    monkeypatch.setattr(
        sampling, "import_module", lambda name: SimpleNamespace(Sampler=Sampler)
    )
    monkeypatch.setenv(sampling.SAMPLER_VARIABLE, "torch")
    assert not sampling.install(tp_size=1, backend="pytorch")
    monkeypatch.setenv(sampling.SAMPLER_VARIABLE, "fused")
    assert not sampling.install(tp_size=2, backend="pytorch")
    assert not sampling.install(tp_size=1, backend="flashinfer")
    assert sampling.install(tp_size=1, backend="pytorch")
    instance = Sampler()
    monkeypatch.setattr(sampling, "supported", lambda value: True)
    monkeypatch.setattr(sampling, "sample", lambda value: "fused")
    assert (
        instance._sample_from_probs(
            None, SimpleNamespace(sampling_seed=None), None, True
        )
        == "fused"
    )
    assert (
        instance._sample_from_probs(None, SimpleNamespace(sampling_seed=1), None, True)
        == "original"
    )
    assert (
        instance._sample_from_probs(
            None, SimpleNamespace(sampling_seed=None), None, False
        )
        == "original"
    )
    monkeypatch.setattr(sampling, "supported", lambda value: False)
    assert (
        instance._sample_from_probs(
            None, SimpleNamespace(sampling_seed=None), None, True
        )
        == "original"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows", [1, 16, 64])
@pytest.mark.parametrize("vocab", [2, 33, 512, 100278])
def test_sampler_matches_torch_samples_and_rng_state(rows, vocab):
    torch.manual_seed(731)
    probabilities = torch.randn(rows, vocab, device="cuda").softmax(-1)
    for _ in range(5):
        state = torch.cuda.get_rng_state()
        expected = torch.multinomial(probabilities, 1).view(-1).int()
        expected_state = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(state)
        actual = sampling.sample(probabilities)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.equal(torch.cuda.get_rng_state(), expected_state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sampler_partial_reduction_ties_infinities_and_nans():
    kernels = import_module("olmo_sglang.sampling_kernels")
    probabilities = torch.ones(4, 8193, device="cuda")
    noise = torch.ones_like(probabilities)
    noise[0, 4096] = 0
    noise[1, 0] = noise[1, 4096] = 0
    probabilities[2, 0] = noise[2, 0] = 0
    noise[2, 4096] = 0
    actual, valid = kernels.select_with_noise(probabilities, noise)
    expected = (probabilities / noise).argmax(-1).int()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert valid.all()
    invalid = torch.tensor(
        [[0.0, 0.0], [-1.0, 2.0], [float("nan"), 1.0], [float("inf"), 1.0]],
        device="cuda",
    )
    _, valid = kernels.select_with_noise(invalid, torch.ones_like(invalid))
    assert not valid.any()
