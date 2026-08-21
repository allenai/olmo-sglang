from pathlib import Path

import pytest
import sglang

from olmo_sglang import tp_smoke


class _FakeEngine:
    def __init__(self, output_ids, logprobs):
        self.output_ids = output_ids
        self.logprobs = logprobs
        self.shutdown_called = False

    def generate(self, *, input_ids, sampling_params, return_logprob):
        assert input_ids == [5, 6, 7, 8]
        assert sampling_params["temperature"] == 0
        assert sampling_params["max_new_tokens"] == 3
        assert sampling_params["ignore_eos"]
        assert return_logprob
        output_token_logprobs = [
            [logprob, token_id, None]
            for logprob, token_id in zip(self.logprobs, self.output_ids)
        ]
        return {
            "output_ids": self.output_ids,
            "meta_info": {"output_token_logprobs": output_token_logprobs},
        }

    def shutdown(self):
        self.shutdown_called = True


def test_create_engine_pins_miles_execution_contract(monkeypatch):
    captured = {}

    def fake_engine(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(sglang, "Engine", fake_engine)

    tp_smoke._create_engine(
        Path("model"), tp_size=2, context_length=128, mem_fraction_static=0.25
    )

    assert captured["tp_size"] == 2
    assert captured["attention_backend"] == "triton"
    assert captured["page_size"] == 1
    assert captured["disable_radix_cache"]
    assert captured["cuda_graph_backend_decode"] == "disabled"
    assert captured["cuda_graph_backend_prefill"] == "disabled"


def _install_fake_engines(
    monkeypatch,
    sharded_output_ids,
    *,
    sharded_logprobs=(-0.102, -0.201, -0.3),
):
    baseline = _FakeEngine([9, 10, 11], [-0.1, -0.2, -0.3])
    sharded = _FakeEngine(sharded_output_ids, sharded_logprobs)
    engines = iter((baseline, sharded))
    monkeypatch.setattr(tp_smoke, "register", lambda: None)
    monkeypatch.setattr(
        tp_smoke, "_create_engine", lambda *args, **kwargs: next(engines)
    )
    return baseline, sharded


def test_tp_smoke_requires_and_reports_exact_parity(monkeypatch):
    baseline, sharded = _install_fake_engines(monkeypatch, [9, 10, 11])

    report = tp_smoke.run_tp_smoke(
        Path("model"), [5, 6, 7, 8], 3, tp_size=2, context_length=16
    )

    assert report["output_ids"] == [9, 10, 11]
    assert report["baseline_tp_size"] == 1
    assert report["comparison_tp_size"] == 2
    assert report["token_parity"]
    assert report["chosen_token_logprob_max_abs_diff"] == pytest.approx(0.002)
    assert report["chosen_token_logprob_mean_abs_diff"] == pytest.approx(0.001)
    assert baseline.shutdown_called
    assert sharded.shutdown_called


def test_tp_smoke_rejects_output_divergence(monkeypatch):
    baseline, sharded = _install_fake_engines(monkeypatch, [9, 12, 11])

    with pytest.raises(AssertionError, match="tensor parallelism changed"):
        tp_smoke.run_tp_smoke(
            Path("model"), [5, 6, 7, 8], 3, tp_size=2, context_length=16
        )

    assert baseline.shutdown_called
    assert sharded.shutdown_called


def test_tp_smoke_requires_tp_greater_than_one():
    with pytest.raises(ValueError, match="greater than one"):
        tp_smoke.run_tp_smoke(
            Path("model"), [5, 6, 7, 8], 3, tp_size=1, context_length=16
        )


def test_tp_smoke_rejects_missing_output_logprobs(monkeypatch):
    baseline, _ = _install_fake_engines(monkeypatch, [9, 10, 11])
    baseline.logprobs = []

    with pytest.raises(
        AssertionError, match="number of output token log probabilities"
    ):
        tp_smoke.run_tp_smoke(
            Path("model"), [5, 6, 7, 8], 3, tp_size=2, context_length=16
        )
