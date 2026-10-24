"""Do not replace upstream tuning for unqualified checkpoint shapes."""

import os
from types import SimpleNamespace

import pytest

from olmo_sglang import moe_configs

QUALIFIED_RUNTIME = dict(device_name="NVIDIA B300 SXM6 AC", tp_size=1, ep_size=1)


@pytest.mark.parametrize(
    "override",
    [
        {"n_routed_experts": 8},
        {"moe_intermediate_size": 2048},
        {"latent_moe_dim": 1024},
        {"latent_moe_dim": None},
    ],
)
def test_other_shapes_keep_upstream_lookup(monkeypatch, override):
    monkeypatch.delenv(moe_configs.ENVIRONMENT_VARIABLE, raising=False)
    values = dict(
        n_routed_experts=512,
        moe_intermediate_size=1024,
        latent_moe_dim=512,
        hidden_size=1024,
    )
    values.update(override)
    assert (
        moe_configs.install_config_dir(SimpleNamespace(**values), **QUALIFIED_RUNTIME)
        is None
    )
    assert moe_configs.ENVIRONMENT_VARIABLE not in os.environ


def test_qualified_shape_respects_explicit_directory(monkeypatch):
    config = SimpleNamespace(
        n_routed_experts=512, moe_intermediate_size=1024, latent_moe_dim=512
    )
    monkeypatch.delenv(moe_configs.ENVIRONMENT_VARIABLE, raising=False)
    assert moe_configs.install_config_dir(config, **QUALIFIED_RUNTIME) == str(
        moe_configs.CONFIG_ROOT
    )
    monkeypatch.setenv(moe_configs.ENVIRONMENT_VARIABLE, "/caller/configs")
    assert moe_configs.install_config_dir(config, **QUALIFIED_RUNTIME) is None
    assert os.environ[moe_configs.ENVIRONMENT_VARIABLE] == "/caller/configs"


@pytest.mark.parametrize(
    "override",
    [
        {"device_name": "NVIDIA B200"},
        {"device_name": "NVIDIA H100 80GB HBM3"},
        {"device_name": ""},
        {"tp_size": 4},
        {"ep_size": 4},
        {"quantized": True},
    ],
)
def test_unqualified_runtime_keeps_upstream_lookup(monkeypatch, override):
    monkeypatch.delenv(moe_configs.ENVIRONMENT_VARIABLE, raising=False)
    config = SimpleNamespace(
        n_routed_experts=512, moe_intermediate_size=1024, latent_moe_dim=512
    )
    assert (
        moe_configs.install_config_dir(config, **(QUALIFIED_RUNTIME | override)) is None
    )
    assert moe_configs.ENVIRONMENT_VARIABLE not in os.environ
