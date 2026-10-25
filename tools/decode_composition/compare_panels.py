"""Compare a PR to bracketing controls; reject changed work or engine settings."""

import argparse
import json
from pathlib import Path

import numpy as np


def compare_panels(before, candidate, after):
    for report in [before, candidate, after]:
        if not report.get("complete") or not report.get("frozen_workload_sha256"):
            raise ValueError("Require complete panels on a frozen workload")
    for field in ["gpu", "model_files", "engine_config", "frozen_workload_sha256"]:
        if before[field] != candidate[field] or before[field] != after[field]:
            raise ValueError(f"Changed comparison contract: {field}")
    cells = {}
    for b, c, a in zip(
        before["waves"], candidate["waves"], after["waves"], strict=True
    ):
        for field in [
            "panel",
            "concurrency",
            "repeat",
            "requests_sha256",
            "generated_tokens",
        ]:
            if b[field] != c[field] or b[field] != a[field]:
                raise ValueError(f"Changed wave identity: {field}")
        cells.setdefault((b["panel"], b["concurrency"]), []).append((b, c, a))
    summaries = []
    rng = np.random.default_rng(31)
    for (panel, concurrency), waves in cells.items():
        timings = np.array(
            [[b["seconds"], c["seconds"], a["seconds"]] for b, c, a in waves]
        )
        tokens = sum(b["generated_tokens"] for b, _, _ in waves)
        control = (timings[:, 0] + timings[:, 2]) / 2
        sample = rng.integers(len(waves), size=(2000, len(waves)))
        changes = control[sample].sum(axis=1) / timings[sample, 1].sum(axis=1) - 1
        summaries.append(
            {
                "panel": panel,
                "concurrency": concurrency,
                "waves": len(waves),
                "baseline_before_tps": tokens / timings[:, 0].sum(),
                "baseline_after_tps": tokens / timings[:, 2].sum(),
                "bracketed_baseline_tps": tokens / control.sum(),
                "candidate_tps": tokens / timings[:, 1].sum(),
                "throughput_change_fraction": control.sum() / timings[:, 1].sum() - 1,
                "paired_wave_change_fractions": (control / timings[:, 1] - 1).tolist(),
                "baseline_after_vs_before_fraction": timings[:, 0].sum()
                / timings[:, 2].sum()
                - 1,
                "paired_wave_bootstrap_95pct": np.quantile(
                    changes, [0.025, 0.975]
                ).tolist(),
            }
        )
    return {
        "workload_sha256": before["frozen_workload_sha256"],
        "cells": summaries,
        "startup_seconds": {
            name: r["startup_seconds"]
            for name, r in [
                ("before", before),
                ("candidate", candidate),
                ("after", after),
            ]
        },
        "warmup_seconds": {
            name: sum(w["seconds"] for w in r["warmups"])
            for name, r in [
                ("before", before),
                ("candidate", candidate),
                ("after", after),
            ]
        },
        "limitation": "Three waves give limited uncertainty estimates; inspect control drift and repeat if gains are small or inconsistent.",
        "decision": "Throughput evidence only; same-token numerical and real RL gates remain separate.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["before", "candidate", "after", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = compare_panels(
        *[
            json.loads(path.read_text())
            for path in [args.before, args.candidate, args.after]
        ]
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
