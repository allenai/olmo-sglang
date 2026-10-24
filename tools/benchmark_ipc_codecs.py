"""Benchmark pinned SGLang IPC definitions without importing the GPU runtime.

Synthetic route values have real frozen-panel shapes; token IDs/logprobs are real.
No sockets, base64, GPU transfer, or scheduler work is included.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import os
import pickle
import platform
import random
import subprocess
import sys
import time
import types
from array import array
from pathlib import Path
from typing import Any, Optional, Union

import msgspec
import numpy as np
import torch

REV = "3145136dcd1238754e0ea2b2ffd546532119c71c"


def selected_source(sglang_repo, path, names):
    source = subprocess.check_output(
        ["git", "-C", str(sglang_repo), "show", f"{REV}:{path}"], text=True
    )
    tree = ast.parse(source)
    tree.body = [
        node
        for node in tree.body
        if getattr(node, "name", None) in names
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)
        )
    ]
    return source, tree


def load_codec(sglang_repo):
    module = types.ModuleType("pinned_sglang_codec")
    sys.modules[module.__name__] = (
        module  # Pickle must resolve the class by module name.
    )
    module.__dict__.update(
        Any=Any,
        Dict=dict,
        List=list,
        Optional=Optional,
        Type=type,
        Union=Union,
        array=array,
        torch=torch,
        np=np,
        pickle=pickle,
        msgspec=msgspec,
        _primitive_types=(int, float, bool, bytes),
    )
    weight_source, weights = selected_source(
        sglang_repo,
        "python/sglang/srt/utils/weight_versions.py",
        {"WeightVersionSpan", "WeightVersionSpans"},
    )
    exec(compile(weights, "pinned-weight-versions.py", "exec"), module.__dict__)
    names = {
        "BaseBatchReq",
        "BatchTokenIDOutput",
        "PickleWrapper",
        "TokenLogprobValues",
        "TokenLogprobIndices",
        "TopLogprobValues",
        "TopLogprobIndices",
        "TokenIdsLogprobValues",
        "TokenIdsLogprobIndices",
        "HiddenStateChunk",
        "OutputHiddenStates",
        "CachedTokensDetails",
        "FinishReasonDict",
        "enc_hook",
        "dec_hook",
        "_maybe_wrap_pickle",
        "_maybe_unwrap_pickle",
        "msgpack_encode",
        "msgpack_decode",
    }
    source, tree = selected_source(
        sglang_repo, "python/sglang/srt/managers/io_struct.py", names
    )
    exec(compile(tree, "pinned-io-struct.py", "exec"), module.__dict__)
    module._msgpack_encoder = msgspec.msgpack.Encoder(enc_hook=module.enc_hook)
    module._msgpack_decoder = msgspec.msgpack.Decoder(
        module.BatchTokenIDOutput | module.PickleWrapper | int | float | bool | bytes,
        dec_hook=module.dec_hook,
    )
    return module, {
        "io_struct_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "weight_versions_sha256": hashlib.sha256(weight_source.encode()).hexdigest(),
        "decoder_union": "BatchTokenIDOutput, PickleWrapper, int, float, bool, bytes; other unused message tags omitted",
    }


def equal(a, b):
    assert type(a) is type(b), (type(a), type(b))
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    elif isinstance(a, msgspec.Struct):
        for name in a.__struct_fields__:
            equal(getattr(a, name), getattr(b, name))
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    else:
        assert a == b


def make_message(codec, records, rows_per_token, with_routes, incremental=False):
    # Populate every required field, preserving the exact upstream types/layout.
    values = {
        f.name: None
        for f in msgspec.structs.fields(codec.BatchTokenIDOutput)
        if f.required
    }
    ids = [r["output_ids"][-1:] if incremental else r["output_ids"] for r in records]
    logs = [r["logprobs"][-1:] if incremental else r["logprobs"] for r in records]
    count = len(records)
    routes = None
    if with_routes:
        generator = torch.Generator().manual_seed(31)
        routes = [
            torch.randint(
                0,
                128,
                (len(r["request"]["input_ids"]) + len(r["output_ids"]) - 1, 16, 16),
                dtype=torch.int32,
                generator=generator,
            )
            for r in records
        ]
        assert rows_per_token == 16 * 16 * 4
        for record, route in zip(records, routes):
            assert (
                route.numel() * route.element_size() == record["routed_experts_bytes"]
            )
    values.update(
        rids=[f"request-{i:04d}" for i in range(count)],
        finished_reasons=[
            None if incremental else {"type": "length", "length": len(x)} for x in ids
        ],
        decoded_texts=["" for _ in records],
        decode_ids=[array("q", x) for x in ids],
        read_offsets=[0 for _ in records],
        output_ids=[array("q", x) for x in ids],
        skip_special_tokens=[True for _ in records],
        spaces_between_special_tokens=[True for _ in records],
        no_stop_trim=[False for _ in records],
        prompt_tokens=[len(r["request"]["input_ids"]) for r in records],
        reasoning_tokens=[0 for _ in records],
        completion_tokens=[len(r["output_ids"]) for r in records],
        cached_tokens=[0 for _ in records],
        output_token_logprobs_val=logs,
        output_token_logprobs_idx=ids,
        routed_experts=routes,
    )
    return codec.BatchTokenIDOutput(**values)


def block(func, argument, iterations):
    cpu_start = time.process_time_ns()
    start = time.perf_counter_ns()
    for _ in range(iterations):
        func(argument)
    return {
        "wall_ns": (time.perf_counter_ns() - start) / iterations,
        "cpu_ns": (time.process_time_ns() - cpu_start) / iterations,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sglang-repo", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=21)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()
    torch.set_num_threads(1)
    affinity = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {affinity[0]})
    codec, source = load_codec(args.sglang_repo)
    fixed_path = args.panel / "baseline-before/tokens-fixed-c64-r0.json"
    mixed_path = args.panel / "baseline-before/tokens-mixed-c64-r0.json"
    fixed = [
        {"request": r["request"], **r["result"]}
        for r in json.loads(fixed_path.read_text())
    ]
    mixed = [
        {"request": r["request"], **r["result"]}
        for r in json.loads(mixed_path.read_text())
    ]
    # Derive route row width from the exact same timed request, not call order.
    row_bytes = fixed[0]["routed_experts_bytes"] // (
        len(fixed[0]["request"]["input_ids"]) + len(fixed[0]["output_ids"]) - 1
    )
    cases = [
        ("incremental-64-no-routes", fixed, False, True),
        ("completion-1-fixed-no-routes", fixed[:1], False, False),
        ("completion-1-fixed-routes", fixed[:1], True, False),
        ("completion-8-fixed-routes", fixed[:8], True, False),
        ("completion-64-fixed-routes", fixed, True, False),
        (
            "completion-1-long-routes",
            [max(mixed, key=lambda r: len(r["output_ids"]))],
            True,
            False,
        ),
    ]
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "msgspec": msgspec.__version__,
        "pickle_protocol": pickle.HIGHEST_PROTOCOL,
        "cpu_model": next(
            x.split(":", 1)[1].strip()
            for x in Path("/proc/cpuinfo").read_text().splitlines()
            if x.startswith("model name")
        ),
        "affinity": sorted(os.sched_getaffinity(0)),
        "sglang_revision": REV,
        "source": source,
        "rounds": args.rounds,
        "seed": args.seed,
        "gc_enabled": gc.isenabled(),
        "timing": "randomized codec/operation order in each round; >=20ms calibrated blocks; 5 warmups; GC outside each block; CPU/wall include result destruction; data creation/validation excluded",
        "route_row_bytes": row_bytes,
        "token_file_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [fixed_path, mixed_path]
        },
        "cases": [],
    }
    rng = random.Random(args.seed)
    for name, records, with_routes, incremental in cases:
        message = make_message(codec, records, row_bytes, with_routes, incremental)
        operations, entries = {}, {}
        for label, enc, dec in [
            (
                "pickle",
                lambda x: pickle.dumps(x, protocol=pickle.HIGHEST_PROTOCOL),
                pickle.loads,
            ),
            ("msgspec", codec.msgpack_encode, codec.msgpack_decode),
        ]:
            raw = enc(message)
            equal(message, dec(raw))
            entries[label] = {"bytes": len(raw), "encode": [], "decode": []}
            for operation, func, value in [
                ("encode", enc, message),
                ("decode", dec, raw),
            ]:
                for _ in range(5):
                    func(value)
                estimate = block(func, value, 3)["wall_ns"]
                iterations = max(1, min(10000, int(20_000_000 / estimate)))
                operations[label, operation] = (func, value, iterations)
                entries[label][operation + "_iterations"] = iterations
        for _ in range(args.rounds):
            order = list(operations)
            rng.shuffle(order)
            for label, operation in order:
                gc.collect()
                entries[label][operation].append(block(*operations[label, operation]))
        for entry in entries.values():
            entry["summary"] = {
                operation: {
                    metric.replace("_ns", "_us"): dict(
                        zip(
                            ["p10", "median", "p90"],
                            np.percentile(
                                [sample[metric] / 1000 for sample in entry[operation]],
                                [10, 50, 90],
                            ).tolist(),
                        )
                    )
                    for metric in ["wall_ns", "cpu_ns"]
                }
                for operation in ["encode", "decode"]
            }
        row = {
            "name": name,
            "requests": len(records),
            "output_tokens": [len(r["output_ids"]) for r in records],
            "route_bytes": sum(
                t.numel() * t.element_size() for t in (message.routed_experts or [])
            ),
            "exact_roundtrip": True,
            "codecs": entries,
        }
        report["cases"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "case": name,
                    "summary": {
                        k: {"bytes": v["bytes"], **v["summary"]}
                        for k, v in entries.items()
                    },
                }
            ),
            flush=True,
        )
        del operations, entries, message


if __name__ == "__main__":
    main()
