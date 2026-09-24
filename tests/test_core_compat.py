"""Storage, publication and numerical controls for Core execution."""

import importlib
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from olmo_sglang import core_compat


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_permutation_orders_nondefault_stream_and_growing_workspace():
    utils = pytest.importorskip("olmo_core.nn.moe.utils")
    stream = torch.cuda.Stream()
    for tokens in [16, 1, 81, 4, 257, 1, 83]:
        with torch.cuda.stream(stream):
            # Delay input production so a default-stream sort cannot safely race it.
            torch.cuda._sleep(2_000_000)
            value = (
                torch.arange(tokens * 128, device="cuda")
                .reshape(tokens, 128)
                .bfloat16()
            )
            routes = torch.randint(
                0, 512, (tokens, 16), device="cuda", dtype=torch.int32
            )
            actual, reverse = core_compat.permute_on_default_stream(
                utils.moe_permute_no_compile, value, routes
            )
            order = routes.flatten().argsort(stable=True) // routes.shape[1]
            expected = value.index_select(0, order)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert reverse.numel() == routes.numel()
    stream.synchronize()


def test_default_off_and_invalid_setting(monkeypatch):
    monkeypatch.delenv(core_compat.ENVIRONMENT_VARIABLE, raising=False)
    assert not core_compat.enabled()
    monkeypatch.setenv(core_compat.ENVIRONMENT_VARIABLE, "1")
    assert core_compat.enabled()
    monkeypatch.setenv(core_compat.ENVIRONMENT_VARIABLE, "typo")
    with pytest.raises(ValueError):
        core_compat.enabled()


def test_fused_rounding_has_explicit_tensor_reference(monkeypatch):
    monkeypatch.delenv("OLMO_SGLANG_ROUNDING_KERNELS", raising=False)
    assert core_compat.fused_rounding_enabled()
    monkeypatch.setenv("OLMO_SGLANG_ROUNDING_KERNELS", "torch")
    assert not core_compat.fused_rounding_enabled()
    assert not core_compat.fused_rounding_enabled("norms")
    monkeypatch.setenv("OLMO_SGLANG_ROUNDING_KERNELS", "moe")
    assert core_compat.fused_rounding_enabled()
    assert not core_compat.fused_rounding_enabled("norms")
    monkeypatch.setenv("OLMO_SGLANG_ROUNDING_KERNELS", "norms")
    assert not core_compat.fused_rounding_enabled()
    assert core_compat.fused_rounding_enabled("norms")
    monkeypatch.setenv("OLMO_SGLANG_ROUNDING_KERNELS", "typo")
    with pytest.raises(ValueError, match="ROUNDING_KERNELS"):
        core_compat.fused_rounding_enabled()


def test_runtime_rejects_unsupported_execution(monkeypatch):
    monkeypatch.setenv(core_compat.ENVIRONMENT_VARIABLE, "1")
    config = SimpleNamespace(
        attention_bias=False, use_rope=False, layer_types=["full_attention"]
    )
    parallel = SimpleNamespace(tp_size=1, moe_ep_size=1)
    args = SimpleNamespace(
        dtype="bfloat16",
        cuda_graph_backend_decode="disabled",
        cuda_graph_backend_prefill="disabled",
    )
    core_compat.validate_runtime(config, parallel, args, None)
    for obj, key, value in [
        (parallel, "tp_size", 2),
        (parallel, "moe_ep_size", 2),
        (args, "cuda_graph_backend_decode", "full"),
        (args, "dtype", "float16"),
    ]:
        old = getattr(obj, key)
        setattr(obj, key, value)
        with pytest.raises(ValueError):
            core_compat.validate_runtime(config, parallel, args, None)
        setattr(obj, key, old)


def test_rounding_mode_keeps_graphs_and_default_attention(monkeypatch):
    monkeypatch.setenv(core_compat.ENVIRONMENT_VARIABLE, "rounding")
    assert core_compat.rounding_enabled()
    assert core_compat.norms_enabled()
    assert not core_compat.enabled()
    config = SimpleNamespace(
        attention_bias=False, use_rope=False, layer_types=["linear_attention"]
    )
    parallel = SimpleNamespace(tp_size=1, moe_ep_size=1)
    args = SimpleNamespace(
        dtype="bfloat16",
        cuda_graph_backend_decode="full",
        cuda_graph_backend_prefill="disabled",
    )
    core_compat.validate_runtime(config, parallel, args, None)
    parallel.tp_size = 2
    with pytest.raises(ValueError, match="TP1"):
        core_compat.validate_runtime(config, parallel, args, None)


def test_dense_layout_and_publication_preserve_used_storage():
    torch.manual_seed(42)
    model = core_compat.CoreDenseMLP(16, 32).bfloat16()
    gate, up = [torch.randn(32, 16).bfloat16() for _ in range(2)]
    down = torch.randn(16, 32).bfloat16()
    model.load_up_gate(model.gate_up_proj.weight, gate, 0)
    model.load_up_gate(model.gate_up_proj.weight, up, 1)
    with torch.no_grad():
        model.down_proj.weight.copy_(down)
    pointers = [p.data_ptr() for p in model.parameters()]
    x = torch.randn(7, 16).bfloat16()

    def reference():
        u, g = (x @ torch.cat((up, gate)).t().contiguous()).chunk(2, -1)
        return torch.bmm(
            (u * F.silu(g)).unsqueeze(0), down.t().contiguous().unsqueeze(0)
        ).squeeze(0)

    torch.testing.assert_close(model(x), reference(), rtol=0, atol=0)
    old = model(x).clone()
    gate.add_(1)
    model.load_up_gate(model.gate_up_proj.weight, gate, 0)
    assert pointers == [p.data_ptr() for p in model.parameters()]
    assert not torch.equal(old, model(x))
    torch.testing.assert_close(model(x), reference(), rtol=0, atol=0)


def test_expert_hf_and_fused_publications_have_identical_storage():
    pytest.importorskip("olmo_core")
    a = core_compat.CoreExperts(3, 16, 32).bfloat16()
    b = core_compat.CoreExperts(3, 16, 32).bfloat16()
    pointers = [p.data_ptr() for p in a.parameters()]
    for _ in range(2):
        gate, up = [torch.randn(3, 32, 16).bfloat16() for _ in range(2)]
        down = torch.randn(3, 16, 32).bfloat16()
        for index in range(3):
            for value, shard, param in [
                (gate, "w1", a.w13_weight),
                (up, "w3", a.w13_weight),
                (down, "w2", a.w2_weight),
            ]:
                a.weight_loader(
                    param, value[index], "weight", shard_id=shard, expert_id=index
                )
        b.weight_loader_fused(
            b.w13_weight, torch.cat((gate, up), 1), "weight", shard_id="w13"
        )
        b.weight_loader_fused(b.w2_weight, down, "weight", shard_id="w2")
        assert pointers == [p.data_ptr() for p in a.parameters()]
        torch.testing.assert_close(a.w13_weight, b.w13_weight, rtol=0, atol=0)
        torch.testing.assert_close(a.w2_weight, b.w2_weight, rtol=0, atol=0)
        torch.testing.assert_close(
            a.w13_weight, torch.cat((up, gate), 1), rtol=0, atol=0
        )
        assert a.w2_weight.transpose(1, 2).is_contiguous()


@pytest.fixture
def hero_runtime():
    # An independent checkpoint fixture, not a copy of the selector's constants.
    config = SimpleNamespace(
        hidden_size=1024,
        attention_hidden_size=1024,
        num_hidden_layers=16,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=128,
        n_routed_experts=512,
        num_experts_per_tok=16,
        moe_intermediate_size=1024,
        latent_moe_dim=512,
        latent_moe_bias=False,
        latent_moe_up_proj_input_norm=False,
        shared_expert_intermediate_size=1024,
        dense_mlp_intermediate_size=8192,
        dense_layers_indices=[0],
        dense_layers_use_shared_expert=True,
        hidden_act="silu",
        gating_function="softmax",
        normalize_expert_weights=1.0,
        restore_weight_scale=True,
        original_num_experts_per_tok=None,
        embed_norm=True,
        embed_scale=32.0,
        scalable_softmax=True,
        rms_norm_eps=1e-6,
        linear_norm_eps=1e-5,
        use_peri_ln=True,
        use_head_qk_norm=True,
        qk_norm_per_head_gains=True,
        attention_bias=False,
        use_rope=False,
        attention_gate_type="elementwise",
        attention_gate_full_precision=True,
        linear_num_key_heads=8,
        linear_num_value_heads=8,
        linear_key_head_dim=128,
        linear_value_head_dim=256,
        linear_conv_kernel_dim=4,
        linear_allow_neg_eigval=True,
        layer_types=(["linear_attention"] * 7 + ["full_attention"]) * 2,
        vocab_size=100278,
    )
    parallel = SimpleNamespace(tp_size=1, moe_ep_size=1)
    args = SimpleNamespace(
        dtype="bfloat16",
        cuda_graph_backend_decode="full",
        cuda_graph_backend_prefill="disabled",
        moe_runner_backend="triton",
        speculative_algorithm=None,
        enable_torch_compile=False,
    )
    return config, parallel, args, None


@pytest.mark.parametrize(
    "setting,expected",
    [
        (None, "rounding"),
        ("auto", "rounding"),
        ("0", "off"),
        ("off", "off"),
        ("rounding", "rounding"),
    ],
)
def test_hero_default_and_explicit_overrides(
    monkeypatch, hero_runtime, setting, expected
):
    monkeypatch.delenv(core_compat.ENVIRONMENT_VARIABLE, raising=False)
    if setting is not None:
        monkeypatch.setenv(core_compat.ENVIRONMENT_VARIABLE, setting)
    with core_compat.model_mode(*hero_runtime) as selected:
        assert selected == expected
        assert core_compat.rounding_enabled() == (expected == "rounding")
        assert not core_compat.enabled()
    if setting in (None, "auto"):
        assert core_compat.mode() == "off"
        assert not core_compat.norms_enabled()


@pytest.mark.parametrize(
    "index,key,value",
    [
        (0, "hidden_size", 2048),
        (0, "n_routed_experts", 64),
        (0, "latent_moe_dim", None),
        (0, "num_hidden_layers", 31),
        (0, "qk_norm_per_head_gains", False),
        (0, "use_rope", True),
        (0, "layer_types", ["full_attention"] * 16),
        (0, "dense_layers_indices", [0, 1]),
        (1, "tp_size", 2),
        (1, "moe_ep_size", 2),
        (2, "dtype", "float16"),
        (2, "moe_runner_backend", "flashinfer_trtllm"),
        (2, "cuda_graph_backend_decode", "breakable"),
        (2, "cuda_graph_backend_prefill", "full"),
        (2, "enable_torch_compile", True),
        (2, "speculative_algorithm", "EAGLE"),
    ],
)
def test_auto_preserves_unqualified_execution(
    monkeypatch, hero_runtime, index, key, value
):
    monkeypatch.delenv(core_compat.ENVIRONMENT_VARIABLE, raising=False)
    setattr(hero_runtime[index], key, value)
    with core_compat.model_mode(*hero_runtime) as selected:
        assert selected == "off"


def test_auto_preserves_quantization_and_missing_profile(monkeypatch, hero_runtime):
    monkeypatch.delenv(core_compat.ENVIRONMENT_VARIABLE, raising=False)
    config, parallel, args, _ = hero_runtime
    with core_compat.model_mode(config, parallel, args, object()) as selected:
        assert selected == "off"
    del config.latent_moe_dim
    with core_compat.model_mode(*hero_runtime) as selected:
        assert selected == "off"


def test_auto_context_resets_after_construction_error(monkeypatch, hero_runtime):
    monkeypatch.delenv(core_compat.ENVIRONMENT_VARIABLE, raising=False)
    with pytest.raises(RuntimeError, match="load failed"):
        with core_compat.model_mode(*hero_runtime):
            assert core_compat.rounding_enabled()
            raise RuntimeError("load failed")
    assert core_compat.mode() == "off"


def test_explicit_full_still_requires_eager_graphs(monkeypatch, hero_runtime):
    monkeypatch.setenv(core_compat.ENVIRONMENT_VARIABLE, "full")
    with pytest.raises(ValueError, match="disabled.*CUDA graphs"):
        with core_compat.model_mode(*hero_runtime):
            pass
    hero_runtime[2].cuda_graph_backend_decode = "disabled"
    with core_compat.model_mode(*hero_runtime) as selected:
        assert selected == "full"
        assert core_compat.enabled()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="SGLang CUDA runtime required"
)
def test_model_constructs_selected_norms_without_leaking_auto(
    monkeypatch, hero_runtime
):
    model_module = importlib.import_module("olmo_sglang.models.olmo3_moe")
    config, parallel, args, _ = hero_runtime
    monkeypatch.delenv(core_compat.ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.setattr(model_module, "get_parallel", lambda: parallel)
    monkeypatch.setattr(model_module, "get_server_args", lambda: args)
    monkeypatch.setattr(
        model_module, "ParallelLMHead", lambda *a, **kw: torch.nn.Identity()
    )
    monkeypatch.setattr(model_module, "LogitsProcessor", lambda *a: torch.nn.Identity())
    monkeypatch.setattr(
        model_module,
        "Olmo3MoeModel",
        lambda *a, **kw: model_module._rms_norm_class()(1024, eps=1e-6),
    )
    monkeypatch.setattr(torch, "get_default_dtype", lambda: torch.bfloat16)
    args.dtype = "auto"
    model = model_module.Olmo3MoeForCausalLM(config)
    assert model.core_compat_mode == "rounding"
    assert isinstance(model.model, model_module.RoundingRMSNorm)
    assert core_compat.mode() == "off"
    config.hidden_size = 2048
    other = model_module.Olmo3MoeForCausalLM(config)
    assert other.core_compat_mode == "off"
    assert isinstance(other.model, model_module.RMSNorm)
    assert isinstance(model.model, model_module.RoundingRMSNorm)


@pytest.mark.parametrize(
    "dtype,expected",
    [(torch.bfloat16, "rounding"), (torch.float16, "off"), (torch.float32, "off")],
)
@pytest.mark.parametrize("requested", ["auto", "bfloat16"])
def test_auto_uses_loader_resolved_dtype(
    monkeypatch, hero_runtime, dtype, expected, requested
):
    monkeypatch.delenv(core_compat.ENVIRONMENT_VARIABLE, raising=False)
    hero_runtime[2].dtype = requested
    with core_compat.model_mode(*hero_runtime, dtype=dtype) as selected:
        assert selected == expected


def test_explicit_rounding_accepts_auto_resolved_to_bf16(monkeypatch, hero_runtime):
    monkeypatch.setenv(core_compat.ENVIRONMENT_VARIABLE, "rounding")
    hero_runtime[2].dtype = "auto"
    with core_compat.model_mode(*hero_runtime, dtype=torch.bfloat16) as selected:
        assert selected == "rounding"
    with pytest.raises(ValueError, match="unquantized BF16"):
        with core_compat.model_mode(*hero_runtime, dtype=torch.float16):
            pass
