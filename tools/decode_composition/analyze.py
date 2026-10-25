"""Audit four composition arms, bracketed throughput and paired Core errors."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .analyze_trace import summarize as summarize_trace
from .compare_logprobs import compare
from .compare_panels import compare_panels

CORE_COMMIT = "e505356353aa7ce1f6ff83e24d6eb945f463714e"
CORE_CONTRACT = {
    "core_builder_sha256": "e5cf4e0803d82b6756725cc9a84aa8aa6958138a859353f21295f69d14e988d6",
    "attention_backend": "torch",
    "routing": "natural",
    "document_contract": "one complete request per document; fixed reference batches, no EP",
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def provenance(report, arm, manifest):
    actual = report["model_code_provenance"]
    files = {
        "src/olmo_sglang/" + name: digest
        for category in ["source_files", "configuration_files"]
        for name, digest in actual[category].items()
    }
    require(files == manifest["source_files"][arm["source"]], "Runtime source mismatch")
    for name, value in arm["options"].items():
        require(
            actual["requested_modes"].get(name) == value, "Option mismatch: " + name
        )
    for name in [
        "OLMO_SGLANG_CORE_COMPAT",
        "OLMO_SGLANG_ROUNDING_KERNELS",
        "SGLANG_MOE_CONFIG_DIR",
    ]:
        require(
            actual["requested_modes"].get(name) == "<unset>",
            "Unexpected inherited override: " + name,
        )
    require(
        actual["actual_use_pickle_ipc"]
        == (arm["options"]["SGLANG_USE_PICKLE_IPC"] == "true"),
        "Codec dispatch mismatch",
    )


def core_provenance(core, paths):
    metadata = read(core.parent / "provenance.json")
    require(
        metadata["core_commit"] == CORE_COMMIT and metadata["core_status"] == "",
        "Changed Core source revision or dirty tree",
    )
    require(read(core.parent / "status.json")["code"] == 0, "Core scoring failed")
    inputs = {Path(name).name: digest for name, digest in metadata["inputs"].items()}
    require(inputs == {path.name: sha(path) for path in paths}, "Changed Core inputs")
    runtime = read(core / "runtime-source.json")
    require(
        runtime["core_builder_sha256"] == CORE_CONTRACT["core_builder_sha256"],
        "Changed Core builder",
    )
    return {
        "commit": metadata["core_commit"],
        "contract": CORE_CONTRACT,
        "provenance_sha256": sha(core.parent / "provenance.json"),
        "runtime_source_sha256": sha(core / "runtime-source.json"),
        "score_files_sha256": {
            "baseline-before-" + path.name: sha(core / ("baseline-before-" + path.name))
            for path in paths
        },
    }


def vectors(root, report):
    result = []
    for wave in report["waves"]:
        path = (
            root
            / f"tokens-{wave['panel']}-c{wave['concurrency']}-r{wave['repeat']}.json"
        )
        records = read(path)
        require(len(records) == wave["concurrency"] == 64, "Missing request records")
        requests = [record["request"] for record in records]
        require(
            hashlib.sha256(json.dumps(requests, sort_keys=True).encode()).hexdigest()
            == wave["requests_sha256"],
            "Request hash mismatch",
        )
        tokens = 0
        for record in records:
            output = record["result"]
            count = len(output["output_ids"])
            require(
                count == len(output["logprobs"]) == record["request"]["tokens"],
                "Token/logprob alignment mismatch",
            )
            require(np.isfinite(output["logprobs"]).all(), "Nonfinite logprobs")
            tokens += count
        require(tokens == wave["generated_tokens"], "Changed generated work")
        result.extend(records)
    require(len(result) == 384, "Require six complete waves")
    return result


def equality(before, after):
    changed_ids = changed_logprobs = 0
    maximum = 0.0
    for left, right in zip(before, after, strict=True):
        require(left["request"] == right["request"], "Different request identity")
        if left["result"]["output_ids"] != right["result"]["output_ids"]:
            changed_ids += 1
        else:
            a = np.asarray(left["result"]["logprobs"])
            b = np.asarray(right["result"]["logprobs"])
            changed_logprobs += int(np.count_nonzero(a != b))
            maximum = max(maximum, float(np.max(np.abs(a - b))))
    return {
        "requests": len(before),
        "tokens": sum(len(r["result"]["output_ids"]) for r in before),
        "trajectories_with_changed_ids": changed_ids,
        "changed_logprobs_on_identical_trajectories": changed_logprobs,
        "max_abs_logprob_change_on_identical_trajectories": maximum,
        "exact": changed_ids == changed_logprobs == 0,
    }


def routes(directory, report, records):
    entries = read(directory / "route-hashes.json")
    # generate() is called once for each cell's warmup, then three waves. The
    # repeated fixed r0/r2 requests share a digest; preserve occurrence order.
    timed = entries[1:4] + entries[5:8]
    require(len(timed) == 6 and len(entries) in [8, 9], "Route call count mismatch")
    flat = []
    for entry, wave in zip(timed, report["waves"], strict=True):
        require(
            entry["requests_sha256"] == wave["requests_sha256"], "Route wave mismatch"
        )
        require(len(entry["requests"]) == 64, "Incomplete route wave")
        flat.extend(entry["requests"])
    for route, record in zip(flat, records, strict=True):
        require(
            route["request_sha256"]
            == hashlib.sha256(
                json.dumps(record["request"], sort_keys=True).encode()
            ).hexdigest(),
            "Route request mismatch",
        )
        require(
            route["output_ids_sha256"]
            == hashlib.sha256(
                json.dumps(record["result"]["output_ids"]).encode()
            ).hexdigest(),
            "Route trajectory mismatch",
        )
        require(
            route["route_bytes"] == record["result"]["routed_experts_bytes"],
            "Route length mismatch",
        )
    return flat


def incremental_throughput(before, after):
    cells = []
    rng = np.random.default_rng(31)
    for panel in ["fixed", "mixed"]:
        b = [v for v in before["waves"] if v["panel"] == panel]
        a = [v for v in after["waves"] if v["panel"] == panel]
        require(len(b) == len(a) == 3, "Need three paired waves")
        for left, right in zip(b, a, strict=True):
            for key in ["repeat", "concurrency", "requests_sha256", "generated_tokens"]:
                require(left[key] == right[key], "Incremental workload mismatch")
        bt = np.asarray([v["seconds"] for v in b])
        at = np.asarray([v["seconds"] for v in a])
        indices = rng.integers(3, size=(2000, 3))
        cells.append(
            {
                "panel": panel,
                "throughput_change_fraction": float(bt.sum() / at.sum() - 1),
                "paired_wave_change_fractions": (bt / at - 1).tolist(),
                "paired_wave_bootstrap_95pct": np.quantile(
                    bt[indices].sum(1) / at[indices].sum(1) - 1, [0.025, 0.975]
                ).tolist(),
            }
        )
    return {
        "cells": cells,
        "limitation": "Direct paired treatment comparison, not separately bracketed; inspect the panel's repeated-main drift. Three waves do not characterize machine/day variance.",
    }


def profiles(root, output):
    result = {}
    for name in ["baseline-before", "performance", "aligned", "fused-routing"]:
        paths = list((root / name / "profile").glob("*DECODE.trace.json.gz"))
        require(len(paths) == 1, "Need one decode trace per profiled arm: " + name)
        trace = paths[0]
        summary = summarize_trace(trace)
        require(
            summary["steps"] == {"step[DECODE bs=64]": 64},
            "Profile must contain 64 C64 decode steps",
        )
        summary["trace"] = str(trace.relative_to(root))
        summary["trace_sha256"] = sha(trace)
        path = output / (name + "-profile.json")
        path.write_text(json.dumps(summary, indent=2) + "\n")
        result[name] = {
            key: summary[key]
            for key in [
                "trace",
                "trace_sha256",
                "step_count",
                "graph_kernel_count_per_step",
                "graph_kernel_ms_per_step",
                "all_gpu_active_union_ms_per_step",
                "all_gpu_span_ms_per_step",
                "family_ms_per_step",
            ]
        }
        result[name]["summary_sha256"] = sha(path)
    return result


def analyze(root, payload, core, output):
    output.mkdir(parents=True, exist_ok=True)
    manifest = read(root / "composition-manifest.json")
    require(
        manifest == read(payload / "composition-manifest.json"), "Manifest mismatch"
    )
    require(read(root / "complete.json")["complete"], "Run not complete")
    require(all(v["code"] == 0 for v in read(root / "status.json")), "Failed phase")
    reference = read(payload / "reference/report.json")
    reports, outputs, route_records = {}, {}, {}
    for name, arm_name in manifest["timing_order"]:
        report = read(root / name / "report.json")
        require(report["complete"], "Incomplete timed arm")
        provenance(report, manifest["arms"][arm_name], manifest)
        require(report["model_files"] == reference["model_files"], "Changed checkpoint")
        require(
            report["engine_config"] == reference["engine_config"],
            "Changed fixed engine settings",
        )
        require(
            report["frozen_workload_sha256"] == sha(payload / "workload.json"),
            "Changed workload",
        )
        require(
            report["packages"] == reference["packages"],
            "Changed pinned runtime packages",
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
        route_records[name] = routes(root / name, report, outputs[name])
    before, after = reports["baseline-before"], reports["baseline-after"]
    audit = {
        "hardware": read(root / "complete.json")["hardware"],
        "manifest_sha256": sha(root / "composition-manifest.json"),
        "throughput": {
            name: compare_panels(before, reports[name], after)
            for name in ["performance", "aligned", "fused-routing"]
        },
        "incremental_throughput": {
            f"{left}_to_{right}": incremental_throughput(reports[left], reports[right])
            for left, right in [
                ("performance", "aligned"),
                ("aligned", "fused-routing"),
            ]
        },
        "free_vectors": {
            name: equality(outputs["baseline-before"], records)
            for name, records in outputs.items()
            if name != "baseline-before"
        },
        "route_equality": {
            name: {
                "exact": records == route_records["baseline-before"],
                "bytes": sum(v["route_bytes"] for v in records),
            }
            for name, records in route_records.items()
        },
        "phase_seconds": read(root / "status.json"),
        "profiles": profiles(root, output),
    }
    replays = {}
    all_core = []
    paths = sorted((payload / "reference").glob("tokens-*.json"))
    require(len(paths) == 6, "Need six frozen reference waves")
    audit["core_provenance"] = core_provenance(core, paths)
    for path in paths:
        trainer = read(core / ("baseline-before-" + path.name))
        require(
            all(trainer[key] == value for key, value in CORE_CONTRACT.items()),
            "Changed Core scoring contract",
        )
        require(
            trainer["complete"] and trainer["input_sha256"] == sha(path),
            "Wrong Core reference input",
        )
        require(
            trainer["model_files"] == reference["model_files"], "Wrong Core checkpoint"
        )
        require(
            len(trainer["rows"]) == trainer["expected_requests"] == 64,
            "Incomplete Core rows",
        )
        all_core.extend(trainer["rows"])
    previous_report = None
    for name, arm in manifest["arms"].items():
        folder = root / ("replay-" + name)
        report = read(folder / "replay-report.json")
        require(
            report["complete"] and report["diagnostic_only"], "Incomplete forced replay"
        )
        provenance(report, arm, manifest)
        require(
            report["reference_report_sha256"] == sha(payload / "reference/report.json"),
            "Changed replay source",
        )
        require(
            report["model_files"] == reference["model_files"], "Changed replay model"
        )
        require(len(report["waves"]) == 6, "Incomplete replay waves")
        if previous_report:
            for key in ["engine_config", "cache_protocol"]:
                require(
                    report[key] == previous_report[key],
                    "Different replay protocol: " + key,
                )
        previous_report = report
        records = []
        for path in paths:
            wave = [v for v in report["waves"] if v["file"] == path.name]
            require(
                len(wave) == 1 and wave[0]["reference_file_sha256"] == sha(path),
                "Changed replay trajectories",
            )
            replay = read(folder / path.name)
            for old, new in zip(read(path), replay, strict=True):
                require(
                    old["request"] == new["request"]
                    and old["result"]["output_ids"] == new["result"]["output_ids"],
                    "Forced output token mismatch",
                )
            records.extend(replay)
        replays[name] = records
    audit["forced_vectors"] = {
        name: equality(replays["baseline"], records)
        for name, records in replays.items()
        if name != "baseline"
    }
    audit["trainer_comparisons"] = {}
    for left, right in [
        ("baseline", "performance"),
        ("baseline", "aligned"),
        ("baseline", "fused-routing"),
        ("performance", "aligned"),
        ("aligned", "fused-routing"),
    ]:
        result = compare(replays[left], replays[right], {"rows": all_core})
        require(
            result["requests"] == 384 and result["tokens"] == 1035090,
            "Incomplete numerical sample",
        )
        result["reference_contract"] = manifest["reference_contract"]
        name = left + "_to_" + right
        path = output / (name + "-trainer.json")
        path.write_text(json.dumps(result, indent=2) + "\n")
        audit["trainer_comparisons"][name] = {
            key: result[key]
            for key in [
                "requests",
                "tokens",
                "baseline",
                "candidate",
                "change",
                "additive_metric_change_prompt_bootstrap_95pct",
            ]
        }
        audit["trainer_comparisons"][name]["sha256"] = sha(path)
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["root", "payload", "core", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root, args.payload, args.core, args.output)
    print(
        json.dumps(
            {
                "hardware": result["hardware"],
                "free_vectors": result["free_vectors"],
                "trainer_comparisons": result["trainer_comparisons"],
            },
            indent=2,
        )
    )
