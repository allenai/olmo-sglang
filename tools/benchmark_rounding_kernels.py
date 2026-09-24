"""Graph-timed component comparisons for the hero BF16 rounding path."""

import argparse
import json
import statistics
from pathlib import Path

import torch
from torch.nn import functional as F

from olmo_sglang import core_compat, rounding_kernels


def microseconds(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(20):
            output = fn()
    samples = []
    for _ in range(3):
        for _ in range(5):
            graph.replay()
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        start.record()
        for _ in range(100):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / 2000)
    del output
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(765)
    results = []
    with torch.no_grad():
        for tokens in [1, 4, 256]:
            gate_up = torch.randn(
                tokens * 16, 2048, device="cuda", dtype=torch.bfloat16
            )
            down = torch.randn(tokens, 16, 1024, device="cuda", dtype=torch.bfloat16)
            weights = torch.rand(tokens, 16, device="cuda")
            value = torch.randn(tokens, 1024, device="cuda", dtype=torch.bfloat16)
            norm = core_compat.CoreRMSNorm(1024, 1e-6).cuda().bfloat16()
            norm.weight.uniform_(0.5, 1.5)
            operations = {
                "silu_mul": (
                    lambda: F.silu(gate_up[:, :1024]) * gate_up[:, 1024:],
                    lambda: rounding_kernels.silu_mul(gate_up),
                ),
                "weighted_sum": (
                    lambda: (down.float() * weights.unsqueeze(-1)).sum(1).bfloat16(),
                    lambda: rounding_kernels.weighted_sum(down, weights),
                ),
                "norm": (
                    lambda: norm(value),
                    lambda: rounding_kernels.rms_norm(value, norm.weight, 1e-6),
                ),
            }
            for name, (reference, fused) in operations.items():
                expected, actual = reference(), fused()
                row = {
                    "tokens": tokens,
                    "operation": name,
                    "torch_us": microseconds(reference),
                    "fused_us": microseconds(fused),
                    "equal_fraction": (actual == expected).float().mean().item(),
                    "max_abs_difference": (actual.float() - expected.float())
                    .abs()
                    .max()
                    .item(),
                }
                results.append(row)
                print("ROUNDING_KERNEL", json.dumps(row), flush=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
