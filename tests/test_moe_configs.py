"""Packaged fused-MoE kernel configurations and their SGLang discovery."""

import json
import os
from importlib import import_module
from types import SimpleNamespace

import pytest

from olmo_sglang import moe_configs
from olmo_sglang.kda import backend

EXPECTED_BATCH_KEYS = {
    1,
    2,
    4,
    8,
    16,
    24,
    32,
    48,
    64,
    96,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
}
CONFIG_FIELDS = {
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "GROUP_SIZE_M",
    "num_warps",
    "num_stages",
}


def test_shipped_files_cover_hero_shape_on_each_gpu():
    files = moe_configs.shipped_config_files()
    names = {path.name for path in files}
    assert names == {
        "E=512,N=1024,device_name=NVIDIA_B300_SXM6_AC.json",
        "E=512,N=1024,device_name=NVIDIA_H100_80GB_HBM3.json",
    }
    for path in files:
        assert path.parent.name == "triton_3_7_1"
        table = json.load(open(path))
        assert {int(key) for key in table} == EXPECTED_BATCH_KEYS
        for config in table.values():
            assert set(config) == CONFIG_FIELDS
            assert config["BLOCK_SIZE_M"] in {16, 32, 64, 128}


def _olmo_config():
    return SimpleNamespace(
        architectures=["Olmo3MoeForCausalLM"],
        layer_types=["full_attention"],
    )


def test_olmo_config_hook_sets_config_dir_only_when_unset(monkeypatch):
    monkeypatch.delenv(moe_configs.ENVIRONMENT_VARIABLE, raising=False)
    backend._prepare_olmo_config(_olmo_config())
    assert os.environ[moe_configs.ENVIRONMENT_VARIABLE] == str(moe_configs.CONFIG_ROOT)

    monkeypatch.setenv(moe_configs.ENVIRONMENT_VARIABLE, "/explicit/choice")
    backend._prepare_olmo_config(_olmo_config())
    assert os.environ[moe_configs.ENVIRONMENT_VARIABLE] == "/explicit/choice"


def test_other_architectures_do_not_touch_config_dir(monkeypatch):
    monkeypatch.delenv(moe_configs.ENVIRONMENT_VARIABLE, raising=False)
    assert (
        backend._prepare_olmo_config(
            SimpleNamespace(architectures=["Qwen3MoeForCausalLM"])
        )
        is False
    )
    assert moe_configs.ENVIRONMENT_VARIABLE not in os.environ


@pytest.mark.parametrize(
    "device_name, expected_64",
    [
        (
            "NVIDIA B300 SXM6 AC",
            {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128},
        ),
        (
            "NVIDIA H100 80GB HBM3",
            {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128},
        ),
    ],
)
def test_sglang_lookup_finds_packaged_file(monkeypatch, device_name, expected_64):
    config_module = import_module(
        "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config"
    )
    runtime = import_module("sglang.srt.runtime_context")
    server_args = import_module("sglang.srt.server_args")
    monkeypatch.setattr(
        runtime, "_CONTEXT", runtime.RuntimeContext(parallel=runtime.ParallelContext())
    )
    with monkeypatch.context() as setup:
        setup.setattr(server_args.ServerArgs, "__post_init__", lambda self: None)
        args = server_args.ServerArgs(model_path="unused")
    runtime.publish(args, role="test")
    monkeypatch.setattr(config_module, "get_device_name", lambda *_: device_name)
    monkeypatch.setenv(moe_configs.ENVIRONMENT_VARIABLE, str(moe_configs.CONFIG_ROOT))
    config_module.get_moe_configs.cache_clear()
    try:
        configs = config_module.get_moe_configs(512, 1024, None)
    finally:
        config_module.get_moe_configs.cache_clear()
    assert configs is not None
    assert {k: configs[64][k] for k in expected_64} == expected_64
    # Shapes without a packaged file still fall through to SGLang's default path.
    config_module.get_moe_configs.cache_clear()
    try:
        assert config_module.get_moe_configs(8, 32, None) is None
    finally:
        config_module.get_moe_configs.cache_clear()
