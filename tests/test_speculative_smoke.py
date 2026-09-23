import sys
from pathlib import Path
from types import ModuleType

import pytest

from olmo_sglang.validation import speculative as speculative_smoke


@pytest.mark.parametrize("speculative", [False, True])
def test_engine_uses_matched_backends_that_support_tiny_attention_heads(
    monkeypatch, speculative
):
    captured = {}
    sglang = ModuleType("sglang")
    sglang.Engine = lambda **kwargs: captured.update(kwargs)
    monkeypatch.setitem(sys.modules, "sglang", sglang)
    speculative_smoke._create_engine(
        Path("model"),
        context_length=256,
        speculative=speculative,
        mem_fraction_static=0.25,
        cuda_graph_backend_decode="full",
    )
    assert captured["attention_backend"] == "triton"
    assert captured["sampling_backend"] == "pytorch"
    assert captured["cuda_graph_backend_decode"] == "full"
    assert captured.get("speculative_algorithm") == ("NGRAM" if speculative else None)


class _FakeEngine:
    def __init__(self, output):
        self.output = output
        self.shutdown_called = False

    def generate(self, *, input_ids, sampling_params):
        assert input_ids == [5, 6, 7, 8]
        assert sampling_params == {
            "temperature": 0,
            "max_new_tokens": 3,
            "ignore_eos": True,
        }
        return self.output

    def shutdown(self):
        self.shutdown_called = True


def _install_fake_engines(monkeypatch, speculative_output_ids):
    baseline = _FakeEngine({"output_ids": [9, 10, 11], "meta_info": {}})
    speculative = _FakeEngine(
        {
            "output_ids": speculative_output_ids,
            "meta_info": {
                "spec_verify_ct": 2,
                "spec_num_proposed_drafts": 6,
                "spec_num_correct_drafts": 3,
                "spec_accept_rate": 0.5,
                "spec_accept_length": 2.5,
            },
        }
    )
    engines = iter((baseline, speculative))
    monkeypatch.setattr(speculative_smoke, "register", lambda: None)
    monkeypatch.setattr(
        speculative_smoke, "_create_engine", lambda *args, **kwargs: next(engines)
    )
    return baseline, speculative


def test_speculative_smoke_requires_and_reports_exact_parity(monkeypatch):
    baseline, speculative = _install_fake_engines(monkeypatch, [9, 10, 11])

    report = speculative_smoke.run_speculative_smoke(
        Path("model"), [5, 6, 7, 8], 3, context_length=16
    )

    assert report["output_ids"] == [9, 10, 11]
    assert report["spec_verify_ct"] == 2
    assert report["spec_num_proposed_drafts"] == 6
    assert report["spec_num_correct_drafts"] == 3
    assert report["spec_accept_rate"] == 0.5
    assert report["spec_accept_length"] == 2.5
    assert baseline.shutdown_called
    assert speculative.shutdown_called


def test_speculative_smoke_rejects_output_divergence(monkeypatch):
    baseline, speculative = _install_fake_engines(monkeypatch, [9, 12, 11])

    with pytest.raises(AssertionError, match="changed greedy output"):
        speculative_smoke.run_speculative_smoke(
            Path("model"), [5, 6, 7, 8], 3, context_length=16
        )

    assert baseline.shutdown_called
    assert speculative.shutdown_called
