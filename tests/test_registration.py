import sys
from types import ModuleType, SimpleNamespace

import pytest

from olmo_sglang import MODEL_ARCHITECTURE, MODEL_PACKAGE, register


def test_public_model_metadata():
    assert MODEL_ARCHITECTURE == "Olmo3MoeForCausalLM"
    assert MODEL_PACKAGE == "olmo_sglang.models"


def test_register_configures_workers_and_current_process(monkeypatch):
    calls = []
    registry_module = ModuleType("sglang.srt.models.registry")
    registry_module.ModelRegistry = SimpleNamespace(
        register=lambda *args, **kwargs: calls.append((args, kwargs))
    )
    backend_module = ModuleType("olmo_sglang.kda.backend")
    backend_module.register_olmo_kda_backend = lambda: calls.append("kda")
    monkeypatch.setitem(sys.modules, "sglang.srt.models.registry", registry_module)
    monkeypatch.setitem(sys.modules, "olmo_sglang.kda.backend", backend_module)
    monkeypatch.delenv("SGLANG_EXTERNAL_MODEL_PACKAGE", raising=False)

    register()

    assert calls == ["kda", ((MODEL_PACKAGE,), {"overwrite": True, "strict": True})]
    assert __import__("os").environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] == MODEL_PACKAGE


def test_register_rejects_a_different_external_package(monkeypatch):
    monkeypatch.setenv("SGLANG_EXTERNAL_MODEL_PACKAGE", "another.models")

    with pytest.raises(RuntimeError, match="another.models"):
        register()
