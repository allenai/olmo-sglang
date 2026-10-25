"""Paired same-token serving/Core error analysis; never compare different rollouts."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def trajectory_hash(record):
    payload = {
        "request": record["request"],
        "output_ids": record["result"]["output_ids"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def metrics(delta):
    absolute = np.abs(delta)
    return {
        "mean_abs": float(absolute.mean()),
        "signed_mean": float(delta.mean()),
        "p95_abs": float(np.quantile(absolute, 0.95)),
        "p99_abs": float(np.quantile(absolute, 0.99)),
        "max_abs": float(absolute.max()),
        # Same ratio direction as the Core scorer: exp(core - serving).
        "outside_20pct": float(ratio_outliers(delta).mean()),
    }


def ratio_outliers(delta):
    return (delta < -np.log(1.2)) | (delta > -np.log(0.8))


def compare(baseline, candidate, core):
    rows = []
    all_base, all_candidate = [], []
    for before, after, reference in zip(baseline, candidate, core["rows"], strict=True):
        digest = trajectory_hash(before)
        if digest != trajectory_hash(after) or digest != reference["trajectory_sha256"]:
            raise ValueError(
                "Comparison requires identical prompts and output tokens, including the Core reference"
            )
        b = np.asarray(before["result"]["logprobs"], dtype=np.float64)
        a = np.asarray(after["result"]["logprobs"], dtype=np.float64)
        c = np.asarray(reference["core_logprobs"], dtype=np.float64)
        if b.shape != a.shape or b.shape != c.shape or b.size == 0:
            raise ValueError("Empty or misaligned logprob vectors")
        if not all(np.isfinite(values).all() for values in [b, a, c]):
            raise ValueError("Nonfinite logprobs")
        b, a = b - c, a - c
        bm, am = metrics(b), metrics(a)
        rows.append(
            {
                "trajectory_sha256": digest,
                "id": before["request"]["id"],
                "domain": before["request"]["domain"],
                "tokens": len(b),
                "baseline": bm,
                "candidate": am,
                "change": {k: am[k] - bm[k] for k in bm},
            }
        )
        all_base.append(b)
        all_candidate.append(a)
    if not rows:
        raise ValueError("No paired trajectories")
    bm, am = metrics(np.concatenate(all_base)), metrics(np.concatenate(all_candidate))
    # Keep all tokens and repeated trajectories from one prompt together.
    # These additive statistics support a cheap clustered bootstrap; it does
    # not estimate uncertainty in the non-additive p95/p99/max tail metrics.
    rng = np.random.default_rng(31)
    clusters = {}
    for row, a, b in zip(rows, all_candidate, all_base, strict=True):
        cluster = clusters.setdefault(row["id"], [0.0, 0.0, 0.0, 0])
        cluster[0] += (np.abs(a) - np.abs(b)).sum()
        cluster[1] += (a - b).sum()
        cluster[2] += ratio_outliers(a).sum() - ratio_outliers(b).sum()
        cluster[3] += len(b)
    values = np.asarray(list(clusters.values()))
    counts = values[:, -1]
    indices = rng.integers(len(clusters), size=(2000, len(clusters)))
    intervals = {}
    for index, name in enumerate(["mean_abs", "signed_mean", "outside_20pct"]):
        samples = values[:, index][indices].sum(axis=1) / counts[indices].sum(axis=1)
        intervals[name] = np.quantile(samples, [0.025, 0.975]).tolist()
    by_domain = {}
    for domain in sorted({row["domain"] for row in rows}):
        selected = [i for i, row in enumerate(rows) if row["domain"] == domain]
        b = np.concatenate([all_base[i] for i in selected])
        a = np.concatenate([all_candidate[i] for i in selected])
        by_domain[domain] = {
            "requests": len(selected),
            "tokens": len(b),
            "baseline": metrics(b),
            "candidate": metrics(a),
        }
    by_position = {}
    for start, end in [
        (0, 128),
        (128, 1024),
        (1024, 4096),
        (4096, 8192),
        (8192, 32768),
    ]:
        b = np.concatenate([v[start:end] for v in all_base])
        a = np.concatenate([v[start:end] for v in all_candidate])
        if b.size:
            by_position[f"{start}:{end}"] = {
                "tokens": len(b),
                "baseline": metrics(b),
                "candidate": metrics(a),
            }
    return {
        "requests": len(rows),
        "prompt_clusters": len(clusters),
        "tokens": int(counts.sum()),
        "baseline": bm,
        "candidate": am,
        "change": {k: am[k] - bm[k] for k in bm},
        "mean_abs_change_request_bootstrap_95pct": intervals["mean_abs"],
        "additive_metric_change_prompt_bootstrap_95pct": intervals,
        "uncertainty_note": "Prompt-cluster percentile bootstrap keeps repeated samples together; small prompt counts and rare-tail estimates remain weak.",
        "decision": "Evidence only: compare against baseline repeat noise and the separate routing-replay RL gate.",
        "by_output_position": by_position,
        "by_domain": by_domain,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ["baseline", "candidate", "core", "output"]:
        parser.add_argument("--" + arg, type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        *[
            json.loads(path.read_text())
            for path in [args.baseline, args.candidate, args.core]
        ]
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
