from pathlib import Path

import pytest

from olmo_sglang import speculative_smoke


class _RecordingEngine:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []
        self.shutdown_called = False

    def generate(self, *, input_ids, sampling_params):
        self.calls.append((input_ids, sampling_params))
        return next(self.outputs)

    def shutdown(self):
        self.shutdown_called = True


def test_branching_smoke_primes_corpus_before_matched_request(monkeypatch):
    metadata = {
        "spec_verify_ct": 2,
        "spec_num_proposed_drafts": 6,
        "spec_num_correct_drafts": 1,
        "spec_accept_rate": 1 / 6,
        "spec_accept_length": 1.5,
    }
    baseline = _RecordingEngine([{"output_ids": [9, 10, 11], "meta_info": {}}])
    speculative = _RecordingEngine(
        [
            {"output_ids": [4], "meta_info": {}},
            {"output_ids": [8], "meta_info": {}},
            {"output_ids": [9, 10, 11], "meta_info": metadata},
        ]
    )
    engines = iter((baseline, speculative))
    engine_kwargs = []
    monkeypatch.setattr(speculative_smoke, "register", lambda: None)
    monkeypatch.setattr(
        speculative_smoke, "_count_ngram_leaf_paths", lambda *args, **kwargs: 2
    )

    def create_engine(*args, **kwargs):
        engine_kwargs.append(kwargs)
        return next(engines)

    monkeypatch.setattr(speculative_smoke, "_create_engine", create_engine)
    corpus_prompts = [[1, 2, 3, 4], [1, 2, 3, 8]]

    report = speculative_smoke.run_speculative_smoke(
        Path("model"),
        [1, 2, 3],
        3,
        context_length=16,
        ngram_breadth=2,
        corpus_prompts=corpus_prompts,
        cuda_graph_backend_decode="full",
    )

    assert engine_kwargs[1]["ngram_breadth"] == 2
    assert engine_kwargs[0]["cuda_graph_backend_decode"] == "full"
    assert engine_kwargs[1]["cuda_graph_backend_decode"] == "full"
    assert [call[0] for call in speculative.calls] == [
        *corpus_prompts,
        [1, 2, 3],
    ]
    assert [call[1]["max_new_tokens"] for call in speculative.calls] == [1, 1, 3]
    assert report["ngram_breadth"] == 2
    assert report["corpus_prompt_count"] == 2
    assert report["branch_leaf_paths"] == 2
    assert report["cuda_graph_backend_decode"] == "full"
    assert baseline.shutdown_called
    assert speculative.shutdown_called


def test_speculative_smoke_rejects_nonpositive_breadth():
    with pytest.raises(ValueError, match="ngram_breadth must be positive"):
        speculative_smoke.run_speculative_smoke(
            Path("model"),
            [1, 2, 3],
            3,
            context_length=16,
            ngram_breadth=0,
        )
