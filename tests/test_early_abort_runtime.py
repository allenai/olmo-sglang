"""Early-abort hooks on the pinned SGLang TokenizerManager.

These tests run the real ``abort_request``, ``_init_req_state`` and
``_send_one_request`` on a manager without its sockets or event loop, so the
scheduler is a list of what would have been sent to it.
"""

import asyncio
import pkgutil
from types import SimpleNamespace

import pytest

from olmo_sglang import early_abort

tokenizer_manager = pytest.importorskip("sglang.srt.managers.tokenizer_manager")
hook_registry = pytest.importorskip("sglang.srt.plugins.hook_registry")
from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.managers.io_struct import AbortReq, GenerateReqInput  # noqa: E402

TokenizerManager = tokenizer_manager.TokenizerManager
HookType = hook_registry.HookType


def hooked_manager_class():
    """A TokenizerManager subclass wrapped the way HookRegistry wraps the real class."""
    wrap = hook_registry._wrap_fn
    return type(
        "HookedTokenizerManager",
        (TokenizerManager,),
        {
            "abort_request": wrap(
                TokenizerManager.abort_request,
                early_abort.abort_request_hook,
                HookType.AROUND,
            ),
            "_init_req_state": wrap(
                TokenizerManager._init_req_state,
                early_abort.init_req_state_hook,
                HookType.AFTER,
            ),
        },
    )


@pytest.fixture
def manager(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(
        early_abort,
        "_table",
        early_abort.EarlyAbortTable(60.0, 16, clock=lambda: clock.now),
    )
    monkeypatch.setattr(
        tokenizer_manager,
        "get_serving",
        lambda: SimpleNamespace(tokenizer_worker_num=1),
    )
    manager = object.__new__(hooked_manager_class())
    manager.rid_to_state = {}
    manager.encoder_dispatch_ready = {}
    manager.enable_trace = False
    manager.enable_metrics = False
    manager.enable_lora = False
    manager.incremental_streaming_output = False
    manager.disaggregation_mode = DisaggregationMode.NULL
    manager.config_value = lambda name: "test"
    manager.sent = []
    manager._dispatch_to_scheduler = manager.sent.append
    manager.clock = clock
    return manager


def register(manager, rid):
    obj = GenerateReqInput(
        rid=rid, input_ids=[1, 2, 3], sampling_params={"max_new_tokens": 4}
    )
    obj.normalize_batch_and_arguments()
    manager._init_req_state(obj)
    return manager.rid_to_state[rid]


def dispatch(manager, rid):
    """Run the dispatch guard of _send_one_request; True means the scheduler never sees rid."""
    return manager._abort_instead_of_dispatch(SimpleNamespace(rid=rid))


def test_abort_before_registration_finishes_request_as_aborted(manager):
    manager.abort_request(rid="late")
    assert manager.sent == []  # Unknown rid: SGLang itself drops the abort.

    state = register(manager, "late")
    other = register(manager, "other")
    asyncio.run(manager._send_one_request(SimpleNamespace(rid="late")))

    assert manager.sent == []
    assert "late" not in manager.rid_to_state
    assert state.finished
    (out,) = state.out_list
    assert out["meta_info"]["id"] == "late"
    assert out["meta_info"]["finish_reason"]["type"] == "abort"
    assert out["output_ids"] == []
    assert not other.abort_before_dispatch
    assert not dispatch(manager, "other")


def test_abort_after_registration_uses_sglang_abort(manager):
    state = register(manager, "active")
    manager.abort_request(rid="active")

    (sent,) = manager.sent
    assert isinstance(sent, AbortReq)
    assert (sent.rid, sent.abort_all, sent.prefix) == ("active", False, False)
    assert state.abort_before_dispatch  # SGLang's own flag for a tokenizer-held request
    assert early_abort.stats().recorded == 0


def test_expired_abort_does_not_stop_the_request(manager):
    manager.abort_request(rid="slow")
    manager.clock.now += 61
    state = register(manager, "slow")
    assert not state.abort_before_dispatch
    assert not dispatch(manager, "slow")


def test_abort_all_is_unchanged(manager):
    state = register(manager, "a")
    manager.abort_request(abort_all=True)
    assert manager.sent[0].abort_all
    assert state.abort_before_dispatch
    assert early_abort.stats().exact_aborts == 0


def test_hook_targets_resolve_on_pinned_sglang():
    for target in (early_abort.ABORT_TARGET, early_abort.REGISTER_TARGET):
        assert callable(pkgutil.resolve_name(target))
    assert hasattr(TokenizerManager, "_abort_instead_of_dispatch")


def test_install_registers_with_hook_registry(monkeypatch):
    monkeypatch.setattr(early_abort, "_table", None)
    monkeypatch.delenv(early_abort.TTL_ENV, raising=False)
    registry = hook_registry.HookRegistry
    monkeypatch.setattr(registry, "_hooks", type(registry._hooks)(list))
    assert early_abort.install()
    assert [hook[:2] for hook in registry._hooks[early_abort.ABORT_TARGET]] == [
        (HookType.AROUND, early_abort.abort_request_hook)
    ]
    assert [hook[:2] for hook in registry._hooks[early_abort.REGISTER_TARGET]] == [
        (HookType.AFTER, early_abort.init_req_state_hook)
    ]
