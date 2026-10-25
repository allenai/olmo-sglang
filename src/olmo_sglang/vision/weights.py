"""Completeness checks for the native decoder's packed checkpoint parameters."""


def validate_language_coverage(parameters, targets, stacked_mapping, num_experts):
    """Require every packed slice, not merely one write to each destination.

    Native loaders track destination parameter names. That alone cannot detect
    a missing K/V projection or one missing expert in a packed MoE parameter.
    ``targets`` contains the native resolver's (kind, name, shard, expert) tuples.
    """
    by_parameter = {}
    for kind, name, shard, expert in targets:
        by_parameter.setdefault(name, set()).add((kind, shard, expert))
    for name in parameters:
        actual = by_parameter.get(name, set())
        whole = {slot for slot in actual if slot[0] in ("direct", "fused")}
        if whole:
            if len(actual) != 1:
                raise ValueError(f"Overlapping language checkpoint writes: {name}")
            continue
        if name.endswith(("experts.w13_weight", "experts.w2_weight")):
            shards = ("w1", "w3") if name.endswith("w13_weight") else ("w2",)
            expected = {
                ("expert", shard, expert)
                for expert in range(num_experts)
                for shard in shards
            }
        else:
            expected = {
                ("stacked", shard, None)
                for destination, _, shard in stacked_mapping
                if destination + "." in name
            }
        if not expected or actual != expected:
            missing = expected - actual
            raise ValueError(
                f"Incomplete language checkpoint parameter {name}: "
                f"{len(missing) if expected else 1} missing slices"
            )
