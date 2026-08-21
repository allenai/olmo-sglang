from pathlib import Path

import pytest

from olmo_sglang import tp_smoke


class _FakeEngine:
    def __init__(self, output_ids):
        self.output_ids = output_ids
        self.shutdown_called = False

    def generate(self, *, input_ids, sampling_params):
        assert input_ids == [5, 6, 7, 8]
        assert sampling_params["temperature"] == 0
        assert sampling_params["max_new_tokens"] == 3
        assert sampling_params["ignore_eos"]
        return {"output_ids": self.output_ids, "meta_info": {}}

    def shutdown(self):
        self.shutdown_called = True


def _install_fake_engines(monkeypatch, sharded_output_ids):
    baseline = _FakeEngine([9, 10, 11])
    sharded = _FakeEngine(sharded_output_ids)
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
