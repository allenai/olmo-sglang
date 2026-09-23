"""Select tests that can run without the SGLang/CUDA runtime installed."""

# These modules import SGLang internals during collection, even for tests whose
# tensors live on the CPU. Exclude them before import only when explicitly asked.
_RUNTIME_TEST_MODULES = {
    "test_ep_diagnostics.py",
    "test_kda_backend.py",
    "test_kda_radix_cache.py",
    "test_kda_tp.py",
    "test_speculative_kda.py",
    "test_weight_targets.py",
}


def pytest_addoption(parser):
    parser.addoption(
        "--cpu-only",
        action="store_true",
        help="Run portable tests without importing the SGLang/CUDA runtime.",
    )


def pytest_ignore_collect(collection_path, config):
    if config.getoption("--cpu-only") and collection_path.name in _RUNTIME_TEST_MODULES:
        return True
    return None
