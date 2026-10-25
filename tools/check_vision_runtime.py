"""Load a complete vision checkpoint and exercise a tiny text-only SGLang forward.

This deliberately disables image request processing (planning #65). It verifies
native model discovery, complete weight loading, KDA cache setup and generation,
not image-conditioned accuracy or full-context memory/performance.
"""

import argparse
import json
import math
import time
from pathlib import Path

import sglang

import olmo_sglang
from olmo_sglang.vision.checkpoint import audit_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit = audit_checkpoint(args.checkpoint)
    olmo_sglang.register()
    start = time.perf_counter()
    engine = sglang.Engine(
        model_path=args.checkpoint,
        trust_remote_code=True,
        enable_multimodal=False,
        context_length=256,
        max_total_tokens=512,
        max_running_requests=2,
        mem_fraction_static=0.65,
        disable_cuda_graph=True,
        disable_radix_cache=True,
        attention_backend="triton",
        dtype="bfloat16",
    )
    load_seconds = time.perf_counter() - start
    try:
        result = engine.generate(
            "Hello",
            sampling_params={
                "temperature": 0.0,
                "max_new_tokens": 4,
                "ignore_eos": True,
            },
            return_logprob=True,
        )
        logprobs = result["meta_info"]["output_token_logprobs"]
        if len(logprobs) != 4 or not all(math.isfinite(row[0]) for row in logprobs):
            raise AssertionError(f"Expected four finite token logprobs, got {logprobs}")
        report = {
            "passed": True,
            "checkpoint": args.checkpoint,
            "audit": audit,
            "load_seconds": load_seconds,
            "result": result,
            "scope": "Native complete-model loading and four-token text forward; no image request support",
        }
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
