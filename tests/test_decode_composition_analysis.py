"""Reject invalid comparisons rather than quietly reporting a favorable row."""

import copy

import pytest

from tools.decode_composition.analyze import (
    equality,
    incremental_throughput,
    provenance,
)


def test_provenance_rejects_wrong_source_option_and_codec():
    options = {"SGLANG_USE_PICKLE_IPC": "false", "OLMO_SGLANG_ROUTING_KERNELS": "torch"}
    modes = dict(options)
    modes.update(
        {
            name: "<unset>"
            for name in [
                "OLMO_SGLANG_CORE_COMPAT",
                "OLMO_SGLANG_ROUNDING_KERNELS",
                "SGLANG_MOE_CONFIG_DIR",
            ]
        }
    )
    report = {
        "model_code_provenance": {
            "source_files": {"model.py": "known"},
            "configuration_files": {},
            "requested_modes": modes,
            "actual_use_pickle_ipc": False,
        }
    }
    arm = {"source": "performance", "options": options}
    manifest = {"source_files": {"performance": {"src/olmo_sglang/model.py": "known"}}}
    provenance(report, arm, manifest)
    for kind in ["source", "mode", "codec", "override"]:
        changed = copy.deepcopy(report)
        p = changed["model_code_provenance"]
        if kind == "source":
            p["source_files"]["model.py"] = "wrong"
        elif kind == "mode":
            p["requested_modes"]["OLMO_SGLANG_ROUTING_KERNELS"] = "fused"
        elif kind == "codec":
            p["actual_use_pickle_ipc"] = True
        else:
            p["requested_modes"]["SGLANG_MOE_CONFIG_DIR"] = "/other/configs"
        with pytest.raises(ValueError):
            provenance(changed, arm, manifest)


def test_changed_tokens_are_not_reported_as_exact_logprob_agreement():
    before = [
        {
            "request": {"id": "a"},
            "result": {"output_ids": [1, 2], "logprobs": [-1.0, -2.0]},
        }
    ]
    after = copy.deepcopy(before)
    assert equality(before, after)["exact"]
    after[0]["result"]["output_ids"][1] = 3
    result = equality(before, after)
    assert not result["exact"]
    assert result["trajectories_with_changed_ids"] == 1
    after = copy.deepcopy(before)
    after[0]["result"]["logprobs"][1] -= 0.001
    result = equality(before, after)
    assert not result["exact"]
    assert result["changed_logprobs_on_identical_trajectories"] == 1


def test_incremental_pair_rejects_changed_work_and_reports_time_ratio():
    report = {
        "waves": [
            {
                "panel": panel,
                "repeat": repeat,
                "concurrency": 64,
                "requests_sha256": f"{panel}-{repeat}",
                "generated_tokens": 640,
                "seconds": float(repeat + 10),
            }
            for panel in ["fixed", "mixed"]
            for repeat in range(3)
        ]
    }
    after = copy.deepcopy(report)
    for wave in after["waves"]:
        wave["seconds"] /= 1.05
    result = incremental_throughput(report, after)
    assert all(
        row["throughput_change_fraction"] == pytest.approx(0.05)
        for row in result["cells"]
    )
    after["waves"][0]["generated_tokens"] -= 1
    with pytest.raises(ValueError, match="workload mismatch"):
        incremental_throughput(report, after)
