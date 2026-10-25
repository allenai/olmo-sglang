"""Audit the unchanged-source B300 projection-ablation follow-up."""

import argparse
import json
from pathlib import Path

from .analyze import (
    equality,
    incremental_throughput,
    provenance,
    read,
    require,
    routes,
    sha,
    vectors,
)

ABLATIONS = {
    "native-kda-gates": "OLMO_SGLANG_KDA_GATE_PROJECTIONS",
    "native-kda-first-stage": "OLMO_SGLANG_KDA_PROJECTIONS",
    "native-moe-projections": "OLMO_SGLANG_MOE_PROJECTIONS",
}


def analyze(root, payload, previous, runner, output):
    manifest = read(payload / "composition-manifest.json")
    require(
        manifest
        == read(root / "composition-manifest.json")
        == read(previous / "composition-manifest.json"),
        "Changed composition payload",
    )
    require(
        read(root / "complete.json")["complete"]
        and read(previous / "complete.json")["complete"],
        "Incomplete experiment",
    )
    hardware = (root / "hardware.txt").read_text().strip()
    old_hardware = (previous / "hardware.txt").read_text().strip()
    fields = [value.strip() for value in hardware.split(",")]
    old_fields = [value.strip() for value in old_hardware.split(",")]
    require(
        len(fields) == len(old_fields) == 4
        and "B300" in fields[0]
        and fields[0] == old_fields[0]
        and fields[2:] == old_fields[2:],
        "Changed GPU model, driver or memory geometry",
    )
    expected = {
        "fresh-main": manifest["arms"]["baseline"],
        "full-performance": manifest["arms"]["performance"],
    }
    performance = expected["full-performance"]
    for name, variable in ABLATIONS.items():
        expected[name] = {
            "source": performance["source"],
            "options": {**performance["options"], variable: "torch"},
        }
    actual = read(root / "attribution-manifest.json")
    require(actual["arms"] == expected, "Changed ablation matrix")
    require(actual["runner_sha256"] == sha(runner), "Changed attribution runner")
    phases = read(root / "status.json")
    require(
        [item["name"] for item in phases] == list(expected)
        and all(item["code"] == 0 for item in phases),
        "Missing, reordered or failed phase",
    )
    reports, outputs, hashes = {}, {}, {}
    reference = read(previous / "baseline-before/report.json")
    for name, arm in expected.items():
        report = read(root / name / "report.json")
        require(report["complete"], "Incomplete benchmark: " + name)
        provenance(report, arm, manifest)
        for key in [
            "model_files",
            "engine_config",
            "packages",
            "frozen_workload_sha256",
        ]:
            require(report[key] == reference[key], "Changed " + key)
        require(
            report["frozen_workload_sha256"] == sha(payload / "workload.json"),
            "Wrong workload",
        )
        for wave, old in zip(report["waves"], reference["waves"], strict=True):
            for key in [
                "panel",
                "repeat",
                "concurrency",
                "requests_sha256",
                "generated_tokens",
            ]:
                require(wave[key] == old[key], "Changed frozen wave: " + key)
        reports[name] = report
        outputs[name] = vectors(root / name, report)
        hashes[name] = routes(root / name, report, outputs[name])
    cross_job = {}
    for name, old_name in [
        ("fresh-main", "baseline-before"),
        ("full-performance", "performance"),
    ]:
        old_report = read(previous / old_name / "report.json")
        old_outputs = vectors(previous / old_name, old_report)
        match = equality(old_outputs, outputs[name])
        match["routes_exact"] = (
            routes(previous / old_name, old_report, old_outputs) == hashes[name]
        )
        require(match["exact"] and match["routes_exact"], "Cross-job repeat failed")
        cross_job[name] = match
    result = {
        "hardware": hardware,
        "manifest_sha256": sha(root / "composition-manifest.json"),
        "runner_sha256": sha(runner),
        "arms": expected,
        "cross_job_controls": cross_job,
        "versus_fresh_main": {
            name: {
                **equality(outputs["fresh-main"], records),
                "routes_exact": hashes[name] == hashes["fresh-main"],
            }
            for name, records in outputs.items()
            if name != "fresh-main"
        },
        "versus_full_performance": {
            name: {
                **equality(outputs["full-performance"], outputs[name]),
                "routes_exact": hashes[name] == hashes["full-performance"],
            }
            for name in ABLATIONS
        },
        "unbracketed_full_gain_over_ablation": {
            name: incremental_throughput(reports[name], reports["full-performance"])[
                "cells"
            ]
            for name in ABLATIONS
        },
        "phase_seconds": phases,
        "report_sha256": {name: sha(root / name / "report.json") for name in expected},
        "limitation": (
            "Free-generation causal ablation only. No separate forced-token "
            "trainer scoring of the ablated arms, and no closing timing control. "
            "Use the full composition panels for throughput and trainer costs."
        ),
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ["root", "payload", "previous", "runner", "output"]:
        parser.add_argument("--" + argument, type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root, args.payload, args.previous, args.runner, args.output)
    print(json.dumps(result["versus_fresh_main"], indent=2))
