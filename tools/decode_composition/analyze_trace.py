"""Summarize a Kineto decode trace without confusing kernel sums with wall time."""

import argparse
import gzip
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def union_us(events):
    intervals = sorted((e["ts"], e["ts"] + e["dur"]) for e in events)
    if not intervals:
        return 0.0
    start, end = intervals[0]
    total = 0.0
    for left, right in intervals[1:]:
        if left > end:
            total += end - start
            start, end = left, right
        else:
            end = max(end, right)
    return total + end - start


def merged_intervals(events):
    result = []
    for start, end in sorted((e["ts"], e["ts"] + e["dur"]) for e in events):
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def intersection_us(left, right):
    i = j = 0
    result = 0.0
    while i < len(left) and j < len(right):
        result += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return result


def family(name):
    if name == "fused_moe_kernel":
        return "expert GEMMs"
    if name == "_grouped_gates":
        return "grouped second-stage KDA GEMMs"
    if name == "_split_latent_activate":
        return "latent split and shared activation"
    if name == "_routing_tail":
        return "routing-tail normalization"
    if "packed_kda_decode" in name:
        return "KDA recurrence"
    if any(
        word in name.lower() for word in ("nvjet", "gemm", "cutlass", "splitkreduce")
    ):
        return "other GEMMs and reductions"
    if any(
        word in name
        for word in (
            "gatherTopK",
            "bitonicSort",
            "softmax_warp",
            "NormOneOps",
            "moe_fused_gate",
        )
    ):
        return "router selection/normalization (partial)"
    if "moe_align" in name:
        return "expert alignment"
    if "_weighted_sum" in name:
        return "expert weighted reduction"
    if "_rms_norm" in name:
        return "RMS normalization"
    if "silu" in name.lower():
        return "SwiGLU/SiLU"
    if "copy" in name.lower():
        return "tensor copies/casts"
    return "other graph kernels"


def summarize(path):
    with gzip.open(path, "rt") if path.suffix == ".gz" else path.open() as f:
        data = json.load(f)
    events = data["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    graph = [e for e in kernels if e.get("args", {}).get("graph id", 0)]
    steps = Counter(
        e["name"]
        for e in events
        if e.get("cat") == "user_annotation" and e["name"].startswith("step[DECODE")
    )
    count = sum(steps.values())
    if not count:
        raise ValueError("No annotated decode steps")
    replays = defaultdict(list)
    for event in graph:
        replays[(event["args"]["graph id"], event["args"]["correlation"])].append(event)
    if len(replays) != count:
        raise ValueError(f"Graph replay/step mismatch: {len(replays)} vs {count}")
    groups = sorted(replays.values(), key=lambda group: min(e["ts"] for e in group))
    by_name = defaultdict(list)
    by_family = Counter()
    for event in graph:
        by_name[event["name"]].append(event)
        by_family[family(event["name"])] += event["dur"]
    divisor = 1000 * count
    graph_sum = sum(e["dur"] for e in graph)
    assert abs(sum(by_family.values()) - graph_sum) < 1e-5
    per_step = []
    per_layer = defaultdict(lambda: defaultdict(list))
    for index, group in enumerate(groups):
        group.sort(key=lambda e: e["ts"])
        moe = [e for e in group if e["name"] == "fused_moe_kernel"]
        if len(moe) != 30:
            raise ValueError(
                f"Expected two expert GEMMs in each of 15 sparse layers, got {len(moe)}"
            )
        for ordinal, event in enumerate(moe):
            projection = "up_gate" if ordinal % 2 == 0 else "down"
            per_layer[ordinal // 2 + 1][projection].append(event["dur"])
        per_step.append(
            {
                "index": index,
                "graph_kernel_sum_ms": sum(e["dur"] for e in group) / 1000,
                "graph_kernel_count": len(group),
                "graph_span_ms": (
                    max(e["ts"] + e["dur"] for e in group) - group[0]["ts"]
                )
                / 1000,
                "moe_up_gate_ms": sum(e["dur"] for e in moe[::2]) / 1000,
                "moe_down_ms": sum(e["dur"] for e in moe[1::2]) / 1000,
            }
        )
    gpu_work = [
        e for e in events if e.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}
    ]
    gpu_span = max(e["ts"] + e["dur"] for e in gpu_work) - min(
        e["ts"] for e in gpu_work
    )
    waits = [
        e
        for e in events
        if e.get("cat") == "cuda_runtime" and e["name"] == "cudaEventSynchronize"
    ]
    wait_us = union_us(waits)
    overlapping_wait_us = intersection_us(
        merged_intervals(waits), merged_intervals(gpu_work)
    )
    top = []
    for name, group in sorted(
        by_name.items(), key=lambda item: -sum(e["dur"] for e in item[1])
    ):
        top.append(
            {
                "name": name,
                "calls_per_step": len(group) / count,
                "ms_per_step": sum(e["dur"] for e in group) / divisor,
                "median_call_us": statistics.median(e["dur"] for e in group),
            }
        )
    return {
        "trace": str(path),
        "steps": dict(steps),
        "step_count": count,
        "graph_kernel_ms_per_step": graph_sum / divisor,
        "graph_kernel_count_per_step": len(graph) / count,
        "all_kernel_ms_per_step": sum(e["dur"] for e in kernels) / divisor,
        "all_gpu_active_union_ms_per_step": union_us(gpu_work) / divisor,
        "all_gpu_span_ms_per_step": gpu_span / divisor,
        "gpu_gaps_ms_per_step": (gpu_span - union_us(gpu_work)) / divisor,
        "cpu_event_wait_ms_per_step": wait_us / divisor,
        "cpu_wait_overlapping_gpu_work_ms_per_step": overlapping_wait_us / divisor,
        "cpu_wait_without_gpu_work_ms_per_step": (wait_us - overlapping_wait_us)
        / divisor,
        "family_ms_per_step": {k: v / divisor for k, v in by_family.most_common()},
        "top_graph_kernels": top,
        "moe_layer_mean_us": {
            str(layer): {k: statistics.mean(v) for k, v in values.items()}
            for layer, values in per_layer.items()
        },
        "per_step": per_step,
        "interpretation": "Profiled GPU sums, union and span are different quantities; gaps are not proven CPU bottlenecks. Families are name-based; router category excludes generic divide/multiply/cast operations. Up/gate and down are identified by verified alternating launch order (15 layers, 30 GEMMs/replay). No hardware bandwidth counters or learning-quality evidence.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.trace)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in {"top_graph_kernels", "per_step", "moe_layer_mean_us"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
