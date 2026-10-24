"""Do not replace upstream tuning for unqualified checkpoint shapes."""

import os
from types import SimpleNamespace

import pytest

from olmo_sglang import moe_configs


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
    assert moe_configs.install_config_dir(SimpleNamespace(**values)) is None
    assert moe_configs.ENVIRONMENT_VARIABLE not in os.environ


def test_qualified_shape_respects_explicit_directory(monkeypatch):
    config = SimpleNamespace(
        n_routed_experts=512, moe_intermediate_size=1024, latent_moe_dim=512
    )
    monkeypatch.delenv(moe_configs.ENVIRONMENT_VARIABLE, raising=False)
    assert moe_configs.install_config_dir(config) == str(moe_configs.CONFIG_ROOT)
    monkeypatch.setenv(moe_configs.ENVIRONMENT_VARIABLE, "/caller/configs")
    assert moe_configs.install_config_dir(config) is None
    assert os.environ[moe_configs.ENVIRONMENT_VARIABLE] == "/caller/configs"
