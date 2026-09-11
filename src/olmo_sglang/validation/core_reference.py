"""Optional native Core forward comparison using exactly the HF forced prefixes."""

from __future__ import annotations

import gc
import hashlib
import json
import os
from importlib import import_module
from pathlib import Path
from typing import Any

import torch


def forced_prefix_cases(
    prompts: list[list[int]], references: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Make explicit, shared token sequences; reject incomplete reference records."""
    if len(prompts) != len(references) or not prompts:
        raise ValueError("Core comparison requires one reference per nonempty prompt")
    cases = []
    for prompt_index, (prompt, reference) in enumerate(zip(prompts, references)):
        if not prompt or len(reference["tokens"]) != len(reference["logprobs"]):
            raise ValueError("HF token and distribution counts must align")
        if not reference["tokens"]:
            raise ValueError("HF reference must include at least one prediction")
        for step, expected in enumerate(reference["logprobs"]):
            ids = list(prompt) + reference["tokens"][:step]
            token = reference["tokens"][step]
            if not 0 <= token < len(expected):
                raise ValueError("HF next token lies outside its distribution")
            cases.append(
                {
                    "prompt_index": prompt_index,
                    "step": step,
                    "input_ids": ids,
                    "input_ids_sha256": hashlib.sha256(
                        json.dumps(ids, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "hf_next_token": token,
                    "hf_logprobs": expected,
                }
            )
    return cases


def compare_logprobs(actual: torch.Tensor, expected: list[float]) -> dict[str, Any]:
    """Measure the entire vocabulary, without recentering or selective filtering."""
    reference = torch.tensor(expected, dtype=torch.float32)
    actual = actual.detach().float().cpu()
    if actual.shape != reference.shape or actual.ndim != 1 or actual.numel() == 0:
        raise ValueError("Core and HF vocabulary dimensions must match exactly")
    if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        raise ValueError("Core and HF logprobs must be finite")
    error = (actual - reference).abs()
    return {
        "max_abs_logprob_error": float(error.max()),
        "mean_abs_logprob_error": float(error.mean()),
        "sum_abs_logprob_error": float(error.double().sum()),
        "vocab_size": actual.numel(),
        "core_next_token": int(actual.argmax()),
    }


def evaluate_core_reference(
    *,
    hf_config: Any,
    hf_state: dict[str, torch.Tensor],
    prompts: list[list[int]],
    references: list[dict[str, Any]],
    logprob_atol: float,
    recurrent_kda: bool,
) -> dict[str, Any]:
    """Use native Core factory/import with torch attention and no distributed EP.

    This is an explicitly labeled semantic forward gate. It does not configure
    an optimizer, training, FlashAttention, EP, or alternative expert kernels.
    The caller's opt-in HF recurrent-prefill helper also affects Core's FLA
    dispatcher; that substitution is recorded rather than silently hidden.
    """
    cases = forced_prefix_cases(prompts, references)
    core_config = import_module("olmo_core.config")
    attention = import_module("olmo_core.nn.attention")
    olmo3 = import_module("olmo_core.nn.moe.v2.olmo3")
    routed_experts = import_module("olmo_core.nn.moe.v2.routed_experts")
    config = olmo3.build_olmo3_moe_config_from_hf_config(
        hf_config,
        dtype=core_config.DType.bfloat16,
        attention_backend=attention.AttentionBackendName.torch,
        router_aux_loss_weight=0.0,
        router_z_loss_weight=0.0,
    )
    model = config.build(init_device="meta")
    longest = max(len(case["input_ids"]) for case in cases)
    model.init_weights(
        max_seq_len=longest,
        max_local_microbatch_size=longest,
        device=torch.device("cuda"),
    )
    olmo3.load_olmo3_moe_hf_state(model, hf_config, hf_state)
    model.eval()
    records = []
    try:
        with torch.no_grad():
            for case in cases:
                ids = torch.tensor([case["input_ids"]], device="cuda")
                logits = model(ids)
                if logits.shape != (1, len(case["input_ids"]), hf_config.vocab_size):
                    raise ValueError(
                        "Native Core logits lost sequence/vocabulary alignment"
                    )
                logprobs = logits[0, -1].float().log_softmax(-1)
                record = {
                    key: value for key, value in case.items() if key != "hf_logprobs"
                }
                record.update(compare_logprobs(logprobs, case["hf_logprobs"]))
                record["core_logprob_of_hf_token"] = float(
                    logprobs[case["hf_next_token"]]
                )
                record["hf_logprob_of_hf_token"] = case["hf_logprobs"][
                    case["hf_next_token"]
                ]
                records.append(record)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    maximum = max(record["max_abs_logprob_error"] for record in records)
    total_error = sum(record["sum_abs_logprob_error"] for record in records)
    total_values = sum(record["vocab_size"] for record in records)
    requires_host_splits = routed_experts.requires_host_side_split_sizes()
    return {
        "gate": "native_core_semantic_forward_reference",
        "core_factory_source_sha256": hashlib.sha256(
            Path(olmo3.__file__).read_bytes()
        ).hexdigest(),
        "expert_backend_environment": os.getenv("OLMO_USE_TORCH_GROUPED_MM"),
        "attention_backend": "torch_sdpa",
        "kda_backend": "fla_recurrent_prefill"
        if recurrent_kda
        else "fla_chunk_prefill",
        "moe_path": "native_no_ep",
        "expert_backend": "torch_grouped_mm"
        if routed_experts.use_torch_grouped_mm()
        else "grouped_gemm",
        "routed_experts_require_host_split_sizes": requires_host_splits,
        "dtype": "bfloat16",
        "training": False,
        "weight_import": "native_factory_strict_hf_state",
        "router_aux_loss_weight": 0.0,
        "router_z_loss_weight": 0.0,
        "optimized_training_qualified": False,
        "records": records,
        "max_abs_logprob_error": maximum,
        "mean_abs_logprob_error": total_error / total_values,
        "token_parity": all(
            r["core_next_token"] == r["hf_next_token"] for r in records
        ),
        "logprob_atol": logprob_atol,
        "passed": maximum <= logprob_atol,
    }
