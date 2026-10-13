"""Versioned FLA kernel choices with strict replay and diagnostic recording."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.metadata
import json
import logging
import os
import subprocess
from pathlib import Path

LOGGER = logging.getLogger(__name__)
PROFILE = "OLMO_SGLANG_KERNEL_PROFILE"
MODE = "OLMO_SGLANG_KERNEL_PROFILE_MODE"
DIGEST = "OLMO_SGLANG_KERNEL_PROFILE_SHA256"
REPORT = "OLMO_SGLANG_KERNEL_PROFILE_REPORT"
REASON = "OLMO_SGLANG_DIAGNOSTIC_OVERRIDE_REASON"
SCOPE = "fla-kda-forward-v1"
KERNELS = (
    "fla.ops.common.chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
    "fla.ops.gla.chunk.chunk_gla_fwd_kernel_o",
    "fla.ops.kda.chunk_intra_token_parallel.chunk_kda_fwd_kernel_intra_token_parallel",
    "fla.ops.kda.wy_fast.recompute_w_u_fwd_kda_kernel",
    "fla.ops.kda.chunk_intra.chunk_kda_fwd_kernel_inter_solve_fused",
    "fla.ops.kda.gate.kda_gate_chunk_cumsum_vector_kernel",
)
_ACTIVE = None


def digest(document):
    """Identify profile content independent of whitespace and key ordering."""
    return hashlib.sha256(
        json.dumps(
            document, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def write_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def identity(config, dtype, arithmetic):
    """Describe the hardware, arithmetic and implementation a profile qualifies."""
    import torch

    from olmo_sglang import core_compat

    gpu = torch.cuda.get_device_properties(torch.cuda.current_device())
    root = Path(__file__).parent
    files = (
        "core_compat.py",
        "rounding.py",
        "rounding_kernels.py",
        "kda/backend.py",
        "kda/packed_decode.py",
        "models/olmo3_moe.py",
    )
    fields = (*core_compat._ROUNDING_PROFILE, "dense_layers_indices", "layer_types")
    return {
        "gpu": {
            "name": gpu.name,
            "capability": [gpu.major, gpu.minor],
            "sm_count": gpu.multi_processor_count,
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "triton", "flash-linear-attention", "sglang")
        },
        "cuda": torch.version.cuda,
        "driver": sorted(
            set(
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=driver_version",
                        "--format=csv,noheader",
                    ],
                    text=True,
                    timeout=15,
                ).split()
            )
        ),
        "kernel_sources": {
            name: hashlib.sha256(
                Path(
                    importlib.import_module(name.rsplit(".", 1)[0]).__file__
                ).read_bytes()
            ).hexdigest()
            for name in KERNELS
        },
        "arithmetic": arithmetic,
        "dtype": str(dtype),
        "model": {name: getattr(config, name, None) for name in fields},
        "implementation": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in files
        },
    }


def tuner(name):
    module, attribute = name.rsplit(".", 1)
    value = getattr(importlib.import_module(module), attribute)
    for _ in range(6):
        if isinstance(getattr(value, "cache", None), dict) and hasattr(
            value, "configs"
        ):
            return value
        value = getattr(value, "fn", None)
    raise ValueError(f"No supported autotuner for {name}")


def configuration(config):
    return {
        "kwargs": config.kwargs,
        "num_warps": config.num_warps,
        "num_stages": config.num_stages,
        "num_ctas": config.num_ctas,
        "maxnreg": config.maxnreg,
    }


def validate(document):
    if set(document) != {
        "schema_version",
        "scope",
        "identity",
        "kernels",
        "provenance",
    }:
        raise ValueError("Kernel profile has missing or unknown fields")
    if document["schema_version"] != 1 or document["scope"] != SCOPE:
        raise ValueError("Unsupported kernel profile schema or scope")
    if set(document["kernels"]) != set(KERNELS):
        raise ValueError("Kernel profile must cover the complete managed kernel set")
    for name, entries in document["kernels"].items():
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"No recorded choices for {name}")
        seen = set()
        for entry in entries:
            if set(entry) != {"key", "config"} or not isinstance(entry["key"], list):
                raise ValueError(f"Invalid tuning entry for {name}")
            if any(
                type(v) not in (int, float, bool, str, type(None)) for v in entry["key"]
            ):
                raise ValueError(f"Unsupported tuning key for {name}")
            key = tuple(entry["key"])
            if key in seen:
                raise ValueError(f"Duplicate tuning key for {name}")
            seen.add(key)
            config = entry["config"]
            if not isinstance(config, dict) or set(config) != {
                "kwargs",
                "num_warps",
                "num_stages",
                "num_ctas",
                "maxnreg",
            }:
                raise ValueError(f"Invalid configuration for {name}")


class StrictCache(dict):
    """Reject new tuning keys and mutation of approved choices."""

    def __init__(self, values, name, used):
        super().__init__(values)
        self.name = name
        self.used = used

    def __contains__(self, key):
        if not dict.__contains__(self, key):
            raise RuntimeError(
                f"Kernel profile coverage miss: {self.name}, key={key!r}. "
                "Refusing independent autotuning; record and qualify an expanded profile."
            )
        return True

    def __getitem__(self, key):
        self.__contains__(key)
        self.used.add((self.name, key))
        return dict.__getitem__(self, key)

    def __setitem__(self, key, value):
        raise RuntimeError(f"Cannot mutate strict kernel choices for {self.name}")

    def clear(self):
        raise RuntimeError(f"Cannot clear strict kernel choices for {self.name}")

    def update(self, *args, **kwargs):
        raise RuntimeError(f"Cannot update strict kernel choices for {self.name}")

    def __delitem__(self, key):
        raise RuntimeError(f"Cannot delete strict kernel choices for {self.name}")

    def pop(self, *args):
        raise RuntimeError(f"Cannot pop strict kernel choices for {self.name}")

    def popitem(self):
        raise RuntimeError(f"Cannot pop strict kernel choices for {self.name}")

    def setdefault(self, key, default=None):
        return self[key]

    def __ior__(self, other):
        self.update(other)


class Session:
    """Own the process's profile selection and bounded coverage reporting."""

    def __init__(self, mode, path, actual, document=None, report=None):
        self.mode, self.path, self.actual = mode, path, actual
        self.document = document
        self.profile_id = digest(document) if document is not None else None
        self.report_path = report
        self.used = set()
        self.last_report = None
        self.last_record = None
        self.last_used_count = None
        self.tuners = {name: tuner(name) for name in KERNELS}
        self.caches = {}
        if mode == "strict":
            validate(document)
            if digest(document["identity"]) != digest(actual):
                differences = [
                    key
                    for key in actual
                    if digest(actual[key]) != digest(document["identity"].get(key))
                ]
                raise ValueError(f"Kernel profile runtime mismatch: {differences}")
            # Resolve every choice before changing any process-global autotuner.
            for name, engine in self.tuners.items():
                if len(engine.configs) < 2:
                    raise ValueError(f"Managed autotuner contract changed for {name}")
                choices = {digest(configuration(c)): c for c in engine.configs}
                values = {}
                for entry in document["kernels"][name]:
                    chosen = choices.get(digest(entry["config"]))
                    if chosen is None:
                        raise ValueError(
                            f"Kernel profile choice unavailable: {name}, {entry['config']}"
                        )
                    values[tuple(entry["key"])] = chosen
                self.caches[name] = StrictCache(values, name, self.used)
            for name, engine in self.tuners.items():
                engine.cache = self.caches[name]
        elif mode == "record":
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x"):
                pass
        else:
            raise ValueError(f"Unsupported kernel profile mode {mode}")
        LOGGER.info(
            "Olmo kernel profile: mode=%s sha256=%s scope=%s path=%s",
            mode,
            self.profile_id,
            SCOPE,
            path,
        )
        self.observe()

    def observe(self):
        if self.mode == "strict":
            if any(
                engine.cache is not self.caches[name]
                for name, engine in self.tuners.items()
            ):
                raise RuntimeError("Strict kernel profile cache was replaced")
            if self.last_used_count == len(self.used):
                return
            self.last_used_count = len(self.used)
        else:
            kernels = {
                name: [
                    {"key": list(key), "config": configuration(value)}
                    for key, value in engine.cache.items()
                ]
                for name, engine in self.tuners.items()
            }
            self.document = {
                "schema_version": 1,
                "scope": SCOPE,
                "identity": self.actual,
                "kernels": kernels,
                "provenance": {
                    "kind": "recorded",
                    "diagnostic_reason": os.environ[REASON],
                },
            }
            current = digest(self.document)
            if current != self.last_record:
                write_json(self.path, self.document)
                self.last_record = current
                self.profile_id = current
        report = {
            "mode": self.mode,
            "profile_sha256": self.profile_id,
            "scope": SCOPE,
            "used": [
                {"kernel": name, "key": list(key)}
                for name, key in sorted(self.used, key=repr)
            ],
        }
        current = digest(report)
        if self.report_path and current != self.last_report:
            write_json(self.report_path, report)
            self.last_report = current


def configure(config, dtype, arithmetic):
    """Apply an explicitly selected profile before warmup or graph capture."""
    global _ACTIVE
    path = os.environ.get(PROFILE)
    mode = os.environ.get(MODE, "strict" if path else "off")
    if mode == "off":
        if path and not os.environ.get(REASON, "").strip():
            raise ValueError(
                "Ignoring a configured kernel profile requires a diagnostic reason"
            )
        if _ACTIVE is not None:
            raise RuntimeError(
                "Cannot disable an installed process-global kernel profile"
            )
        return None
    if mode not in {"strict", "record"} or not path:
        raise ValueError(
            "Kernel profile requires mode=strict/record and a profile path"
        )
    path = Path(path.format(pid=os.getpid()))
    if mode == "record" and not os.environ.get(REASON, "").strip():
        raise ValueError("Recording kernel choices requires a diagnostic reason")
    document = json.loads(path.read_text()) if mode == "strict" else None
    if mode == "strict":
        expected = os.environ.get(DIGEST)
        if not expected or digest(document) != expected:
            raise ValueError(
                "Kernel profile SHA256 is missing or differs from the required content identity"
            )
    actual = identity(config, dtype, arithmetic)
    if _ACTIVE is not None:
        if (_ACTIVE.mode, _ACTIVE.path, digest(_ACTIVE.actual), _ACTIVE.profile_id) != (
            mode,
            path,
            digest(actual),
            digest(document) if document is not None else _ACTIVE.profile_id,
        ):
            raise RuntimeError(
                "Cannot mix kernel profiles or runtime identities within one process"
            )
        return _ACTIVE
    report = os.environ.get(REPORT)
    _ACTIVE = Session(
        mode,
        path,
        actual,
        document,
        Path(report.format(pid=os.getpid())) if report else None,
    )
    return _ACTIVE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser(
        "inspect", help="Validate a profile and print its content identity"
    )
    inspect.add_argument("profile", type=Path)
    convert = sub.add_parser(
        "import-replay", help="Convert diagnostic choices on the target GPU/runtime"
    )
    convert.add_argument("replay", type=Path)
    convert.add_argument("output", type=Path)
    convert.add_argument("--model", required=True)
    convert.add_argument(
        "--arithmetic", choices=["rounding", "full"], default="rounding"
    )
    args = parser.parse_args()
    if args.command == "inspect":
        document = json.loads(args.profile.read_text())
        validate(document)
        print(json.dumps({"sha256": digest(document), "scope": document["scope"]}))
        return
    import torch
    from transformers import AutoConfig

    if args.output.exists():
        raise FileExistsError(args.output)
    recorded = json.loads(args.replay.read_text())
    kernels = {}
    for name in KERNELS:
        choices = {str(c): configuration(c) for c in tuner(name).configs}
        kernels[name] = [
            {"key": list(ast.literal_eval(k)), "config": choices[v]}
            for k, v in recorded[name].items()
        ]
    document = {
        "schema_version": 1,
        "scope": SCOPE,
        "identity": identity(
            AutoConfig.from_pretrained(args.model, trust_remote_code=True),
            torch.bfloat16,
            args.arithmetic,
        ),
        "kernels": kernels,
        "provenance": {
            "kind": "imported-diagnostic-replay",
            "input_sha256": digest(recorded),
        },
    }
    validate(document)
    write_json(args.output, document)
    print(json.dumps({"sha256": digest(document), "scope": SCOPE}))


if __name__ == "__main__":
    main()
