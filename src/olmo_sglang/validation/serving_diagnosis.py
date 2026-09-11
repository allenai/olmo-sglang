"""Describe mode differences on identical prefixes without changing gate tolerances."""

from itertools import combinations


def compare_modes(modes):
    comparisons = []
    for first, second in combinations(modes, 2):
        same_graphs = first["graphs"] == second["graphs"]
        same_chunk = first["chunk"] == second["chunk"]
        if same_graphs == same_chunk:
            continue
        if len(first["forced"]) != len(second["forced"]):
            raise ValueError("Mode prompt counts differ")
        records = []
        for prompt, (left, right) in enumerate(zip(first["forced"], second["forced"])):
            if len(left) != len(right):
                raise ValueError("Mode prediction counts differ")
            for step, (a, b) in enumerate(zip(left, right)):
                if a["input_ids"] != b["input_ids"]:
                    raise ValueError(
                        "Cannot compare modes on different forced prefixes"
                    )
                values_a = {
                    row["token"]: row["sglang"] for row in a["checked_top_logprobs"]
                }
                values_b = {
                    row["token"]: row["sglang"] for row in b["checked_top_logprobs"]
                }
                common = values_a.keys() & values_b.keys()
                records.append(
                    {
                        "prompt_index": prompt,
                        "step": step,
                        "compared_tokens": len(common),
                        "first_only_tokens": sorted(values_a.keys() - values_b.keys()),
                        "second_only_tokens": sorted(values_b.keys() - values_a.keys()),
                        "max_abs_common_token_logprob_difference": max(
                            (
                                abs(values_a[token] - values_b[token])
                                for token in common
                            ),
                            default=None,
                        ),
                    }
                )
        comparisons.append(
            {
                "variable": "chunk" if same_graphs else "graphs",
                "first": {key: first[key] for key in ("graphs", "chunk")},
                "second": {key: second[key] for key in ("graphs", "chunk")},
                "records": records,
                "interpretation": (
                    "Intersection of returned top-token sets only; "
                    "not full-vocabulary parity"
                ),
            }
        )
    return comparisons
