"""Early-abort hooks on the pinned SGLang TokenizerManager.

The tests drive the real ``generate_request``, ``abort_request``,
``_init_req_state``, ``_handle_abort_req`` and ``_stream_one_response`` on a
manager without sockets or a model. Tokenization and dispatch are stubbed: the
stub scheduler records each rid it receives and finishes it on the next loop turn.
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
from sglang.srt.utils.aio_rwlock import RWLock  # noqa: E402

TokenizerManager = tokenizer_manager.TokenizerManager
HookType = hook_registry.HookType
WRAPPED = {
    "abort_request": early_abort.abort_request_hook,
    "_init_req_state": early_abort.init_req_state_hook,
    "generate_request": early_abort.generate_request_hook,
}


def hooked_manager_class():
    """A TokenizerManager subclass wrapped the way HookRegistry wraps the real class."""
    attrs = {
        name: hook_registry._wrap_fn(
            getattr(TokenizerManager, name), hook, HookType.AROUND
        )
        for name, hook in WRAPPED.items()
    }
    return type("HookedTokenizerManager", (TokenizerManager,), attrs)


class Harness:
    def __init__(self, manager):
        self.manager = manager
        self.scheduled = []
        self.tokenizing = {}

    async def tokenize(self, obj):
        # Park here so a test can abort while the request is tokenizer-held.
        gate = self.tokenizing.get(obj.rid)
        if gate is not None:
            await gate.wait()
        return SimpleNamespace(rid=obj.rid, input_ids=list(obj.input_ids))

    async def send(self, tokenized):
        manager = self.manager
        if manager._abort_instead_of_dispatch(tokenized):
            return
        self.scheduled.append(tokenized.rid)
        # The scheduler answers later, as in serving, after the waiter is bound.
        asyncio.get_running_loop().call_soon(self.finish, tokenized.rid)

    def finish(self, rid):
        state = self.manager.rid_to_state.pop(rid)
        state.finished = True
        state.out_list.append(
            {
                "text": "ok",
                "output_ids": [7],
                "meta_info": {
                    "id": rid,
                    "finish_reason": {"type": "length", "length": 1},
                },
            }
        )
        state.event.set()


@pytest.fixture
def harness(monkeypatch):
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
    monkeypatch.setattr(
        tokenizer_manager, "get_disagg", lambda: SimpleNamespace(language_only=False)
    )
    manager = object.__new__(hooked_manager_class())
    harness = Harness(manager)
    harness.clock = clock
    noop = lambda *args, **kwargs: None  # noqa: E731

    async def no_lora(obj):
        return None

    manager.__dict__.update(
        rid_to_state={},
        encoder_dispatch_ready={},
        enable_trace=False,
        enable_metrics=False,
        enable_lora=False,
        enable_priority_scheduling=False,
        incremental_streaming_output=False,
        disaggregation_mode=DisaggregationMode.NULL,
        config_value=lambda name: "test",
        auto_create_handle_loop=noop,
        request_logger=SimpleNamespace(
            log_received_request=noop, log_finished_request=noop
        ),
        request_metrics_exporter_manager=SimpleNamespace(
            exporter_enabled=lambda: False
        ),
        _is_pause=False,
        tokenizer=None,
        _validate_and_resolve_lora=no_lora,
        _tokenize_one_request=harness.tokenize,
        _send_one_request=harness.send,
        aborts_to_scheduler=[],
    )
    manager._dispatch_to_scheduler = manager.aborts_to_scheduler.append
    return harness


def generate(manager, rid):
    obj = GenerateReqInput(
        rid=rid, input_ids=[1, 2, 3], sampling_params={"max_new_tokens": 4}
    )
    return manager.generate_request(obj).__anext__()


async def with_loop_state(manager):
    manager.is_pause_cond = asyncio.Condition()
    manager.model_update_lock = RWLock()


def test_abort_before_generate_returns_abort_and_never_schedules(harness):
    manager = harness.manager

    async def scenario():
        await with_loop_state(manager)
        manager.abort_request(rid="late")
        assert (
            manager.aborts_to_scheduler == []
        )  # Unknown rid: SGLang itself drops the abort.
        return await generate(manager, "late"), await generate(manager, "other")

    late, other = asyncio.run(scenario())

    assert harness.scheduled == ["other"]
    assert late["meta_info"]["id"] == "late"
    assert late["meta_info"]["finish_reason"] == {
        "type": "abort",
        "message": early_abort.ABORT_MESSAGE,
    }
    assert late["output_ids"] == [] and late["meta_info"]["completion_tokens"] == 0
    assert other["meta_info"]["finish_reason"]["type"] == "length"
    assert manager.rid_to_state == {}
    assert len(early_abort.table()) == 0


def test_abort_after_registration_uses_sglang_abort(harness):
    manager = harness.manager

    async def scenario():
        await with_loop_state(manager)
        harness.tokenizing["held"] = asyncio.Event()
        task = asyncio.create_task(generate(manager, "held"))
        while "held" not in manager.rid_to_state:
            await asyncio.sleep(0)
        manager.abort_request(rid="held")
        task.cancel()

    asyncio.run(scenario())
    (sent,) = manager.aborts_to_scheduler
    assert isinstance(sent, AbortReq)
    assert (sent.rid, sent.abort_all, sent.prefix) == ("held", False, False)
    assert early_abort.stats().recorded == 0
    assert early_abort.stats().active_aborts == 1


def test_expired_abort_lets_the_request_run(harness):
    manager = harness.manager

    async def scenario():
        await with_loop_state(manager)
        manager.abort_request(rid="slow")
        harness.clock.now += 61
        return await generate(manager, "slow")

    out = asyncio.run(scenario())
    assert harness.scheduled == ["slow"]
    assert out["meta_info"]["finish_reason"]["type"] == "length"


def test_batch_is_left_to_sglang(harness):
    manager = harness.manager
    manager.abort_request(rid="b1")
    obj = GenerateReqInput(
        rid=["b0", "b1"], input_ids=[[1], [2]], sampling_params={"max_new_tokens": 1}
    )
    obj.normalize_batch_and_arguments()
    manager._init_req_state(obj)
    assert set(manager.rid_to_state) == {"b0", "b1"}
    assert not manager.rid_to_state["b1"].abort_before_dispatch
    assert len(early_abort.table()) == 1


def test_abort_all_is_unchanged(harness):
    manager = harness.manager
    obj = GenerateReqInput(
        rid="a", input_ids=[1], sampling_params={"max_new_tokens": 1}
    )
    obj.normalize_batch_and_arguments()
    manager._init_req_state(obj)
    manager.abort_request(abort_all=True)
    assert manager.aborts_to_scheduler[0].abort_all
    assert manager.rid_to_state["a"].abort_before_dispatch
    assert early_abort.stats().exact_aborts == 0


def test_health_check_abort_does_not_stop_the_request(harness):
    manager = harness.manager
    rid = early_abort.HEALTH_CHECK_RID_PREFIX + "_probe"

    async def scenario():
        await with_loop_state(manager)
        manager.abort_request(rid=rid)
        return await generate(manager, rid)

    out = asyncio.run(scenario())
    assert harness.scheduled == [rid]
    assert out["meta_info"]["finish_reason"]["type"] == "length"
    assert manager.rid_to_state == {}


def test_hook_targets_resolve_on_pinned_sglang():
    from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX

    assert early_abort.HEALTH_CHECK_RID_PREFIX == HEALTH_CHECK_RID_PREFIX
    for target in (
        early_abort.ABORT_TARGET,
        early_abort.REGISTER_TARGET,
        early_abort.GENERATE_TARGET,
    ):
        assert callable(pkgutil.resolve_name(target))
    for name in early_abort._REQUIRED_METHODS:
        assert hasattr(TokenizerManager, name)


def test_install_registers_with_hook_registry(monkeypatch):
    monkeypatch.setattr(early_abort, "_table", None)
    monkeypatch.delenv(early_abort.TTL_ENV, raising=False)
    registry = hook_registry.HookRegistry
    monkeypatch.setattr(registry, "_hooks", type(registry._hooks)(list))
    assert early_abort.install()
    for target, hook in (
        (early_abort.ABORT_TARGET, early_abort.abort_request_hook),
        (early_abort.REGISTER_TARGET, early_abort.init_req_state_hook),
        (early_abort.GENERATE_TARGET, early_abort.generate_request_hook),
    ):
        assert [entry[:2] for entry in registry._hooks[target]] == [
            (HookType.AROUND, hook)
        ]
