"""Generate text from an OLMo checkpoint with an embedded SGLang engine."""

from __future__ import annotations

import argparse

from olmo_sglang import register


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Hugging Face checkpoint path or model ID")
    parser.add_argument("--prompt", default="The future of language modeling is")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--tp-size", type=int, default=1)
    args = parser.parse_args()

    register()

    import sglang as sgl

    engine = sgl.Engine(
        model_path=args.model,
        trust_remote_code=True,
        tp_size=args.tp_size,
        attention_backend="torch_native",
        disable_radix_cache=True,
        cuda_graph_backend_decode="disabled",
        cuda_graph_backend_prefill="disabled",
        max_total_tokens=128,
        mem_fraction_static=0.15,
    )
    try:
        result = engine.generate(
            prompt=args.prompt,
            sampling_params={
                "temperature": 0,
                "max_new_tokens": args.max_new_tokens,
            },
        )
        print(result["text"])
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
