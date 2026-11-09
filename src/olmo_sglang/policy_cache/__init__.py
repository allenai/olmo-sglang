"""Opt-in policy namespaces through SGLang's registered cache-backend interface."""

BACKEND = "olmo_policy_versioned_v1"


def register() -> None:
    # Registration runs during model discovery; defer engine/cache imports until
    # the factory is selected, after the runtime has resolved the model.
    from sglang.srt.mem_cache import registry

    factory = registry.get_radix_cache_factory(BACKEND)
    if factory is None:
        registry.register_radix_cache_backend(BACKEND, _create)
    elif factory is not _create:
        raise RuntimeError(
            f"Cache backend {BACKEND!r} is already registered by another package"
        )


def _create(context):
    from olmo_sglang.policy_cache.backend import create

    return create(context)
