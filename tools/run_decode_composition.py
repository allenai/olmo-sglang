"""Frozen C64 composition panel and same-token replay; run from a Beaker payload."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from run_accuracy_tuning import run_process

PAYLOAD = Path("/payload")
OUTPUT = Path("/output")
OPTIONS = {
    "OLMO_SGLANG_SHARED_ADD_NORM": "fused",
    "OLMO_SGLANG_ROUTING_TAIL": "fused",
    "OLMO_SGLANG_ROUTER_INPUT": "fused",
    "OLMO_SGLANG_SAMPLER": "fused",
    "OLMO_SGLANG_KDA_PROJECTIONS": "packed",
    "OLMO_SGLANG_MOE_PROJECTIONS": "packed",
    "OLMO_SGLANG_KDA_GATE_PROJECTIONS": "grouped",
    "SGLANG_USE_PICKLE_IPC": "false",
}


def environment(arm, hardware):
    env = os.environ.copy()
    for name in list(env):
        if name.startswith("OLMO_SGLANG_") or name == "SGLANG_MOE_CONFIG_DIR":
            env.pop(name)
    env.update(arm["options"])
    libraries = "/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64"
    env["LD_LIBRARY_PATH"] = (
        "/usr/local/cuda/compat:" + libraries if hardware == "H100" else libraries
    )
    env["PYTHONPATH"] = str(PAYLOAD / arm["source"] / "src")
    env["TRITON_CACHE_DIR"] = env["ISSUE31_ROOT"] + "/sglang-triton-cache"
    return env


def main():
    OUTPUT.mkdir(exist_ok=True)
    manifest = json.loads((PAYLOAD / "composition-manifest.json").read_text())
    hardware = os.environ["ISSUE31_HARDWARE"]
    actual = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    if hardware not in actual or len(actual.splitlines()) != 1:
        raise ValueError(f"Expected one {hardware}; found {actual}")
    (OUTPUT / "hardware.txt").write_text(actual + "\n")
    for relative, expected in manifest["payload_files"].items():
        if hashlib.sha256((PAYLOAD / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Payload identity mismatch: {relative}")
    (OUTPUT / "composition-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (OUTPUT / "workload.json").write_bytes((PAYLOAD / "workload.json").read_bytes())
    statuses = []

    def run(name, command, env, timeout):
        start = time.perf_counter()
        result = run_process(name, command, env, timeout)
        result["seconds"] = time.perf_counter() - start
        statuses.append(result)
        (OUTPUT / "status.json").write_text(json.dumps(statuses, indent=2) + "\n")
        if result["code"]:
            raise SystemExit(1)

    # Check both source variants because q/k arithmetic differs. The broad
    # component prechecks run once; the changed recurrence runs in both variants.
    for script in manifest["gpu_prechecks"]:
        run(
            "precheck-" + Path(script).stem,
            [sys.executable, str(PAYLOAD / script)],
            environment(manifest["arms"]["performance"], hardware),
            240,
        )
    run(
        "precheck-qk",
        [sys.executable, str(PAYLOAD / "check_kda_projections.py")],
        environment(manifest["arms"]["aligned"], hardware),
        240,
    )
    for name, arm_name in manifest["timing_order"]:
        arm = manifest["arms"][arm_name]
        command = [
            sys.executable,
            str(PAYLOAD / "benchmark.py"),
            "--backend",
            "sglang",
            "--root",
            os.environ["ISSUE31_ROOT"],
            "--output",
            str(OUTPUT / name),
            "--concurrency",
            "64",
            "--memory-fraction",
            "0.7",
            "--prefill-chunk",
            "8192",
            "--retain",
            "64",
            "--workload",
            str(OUTPUT / "workload.json"),
            "--sglang-rl-cache",
            "--sglang-routing-capture",
            "--sglang-return-routes",
        ]
        if name != "baseline-after":
            command.append("--profile")
        run(name, command, environment(arm, hardware), 900)

    # Every arm scores the same frozen H100 trajectories, on both GPU types.
    # This permits the same Core reference and matched cache history throughout.
    # Replay contains CPU synchronization and is never a throughput measurement.
    for arm_name, arm in manifest["arms"].items():
        run(
            "replay-" + arm_name,
            [
                sys.executable,
                str(PAYLOAD / "replay_suite.py"),
                "--root",
                os.environ["ISSUE31_ROOT"],
                "--reference",
                str(PAYLOAD / "reference"),
                "--output",
                str(OUTPUT / ("replay-" + arm_name)),
            ],
            environment(arm, hardware),
            900,
        )
    (OUTPUT / "complete.json").write_text(
        json.dumps({"complete": True, "hardware": hardware}) + "\n"
    )


if __name__ == "__main__":
    main()
