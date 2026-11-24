"""Early-abort table and hook contract, without the SGLang runtime."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
import tomllib

from olmo_sglang import early_abort


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class FakeManager:
    """The TokenizerManager methods the hooks wrap or call, reduced to rid bookkeeping.

    The three hooked methods are wrapped the way HookRegistry applies an AROUND hook.
    """

    def __init__(self):
        self.rid_to_state = {}
        self.scheduler_aborts = []
        self.scheduled = []

    def _abort_request(self, rid="", abort_all=False, prefix=False):
        # Pinned single-tokenizer behaviour: an exact abort for an unknown rid is dropped.
        if not abort_all and not prefix and rid not in self.rid_to_state:
            return
        self.scheduler_aborts.append(rid)

    def abort_request(self, *args, **kwargs):
        return early_abort.abort_request_hook(
            FakeManager._abort_request, self, *args, **kwargs
        )

    def _register(self, obj, request=None):
        rids = [obj.rid] if obj.is_single else obj.rid
        for rid in rids:
            if rid in self.rid_to_state:
                raise ValueError(f"Duplicate request ID detected: {rid}")
            self.rid_to_state[rid] = SimpleNamespace(out_list=[], finished=False)

    def _init_req_state(self, obj, request=None):
        return early_abort.init_req_state_hook(
            FakeManager._register, self, obj, request
        )

    async def _generate(self, obj, request=None):
        self._init_req_state(obj, request)
        await asyncio.sleep(0)
        rids = [obj.rid] if obj.is_single else obj.rid
        self.scheduled.extend(rids)
        for rid in rids:
            del self.rid_to_state[rid]
        yield {"meta_info": {"id": obj.rid, "finish_reason": {"type": "length"}}}

    def generate_request(self, obj, request=None):
        return early_abort.generate_request_hook(
            FakeManager._generate, self, obj, request
        )

    def _handle_abort_req(self, req):
        state = self.rid_to_state.pop(req.rid)
        state.finished = True
        state.out_list.append(
            {"meta_info": {"id": req.rid, "finish_reason": {"type": "abort"}}}
        )

    async def _stream_one_response(self, obj, state, request=None):
        assert state.finished
        yield state.out_list[0]


def request(rid):
    return SimpleNamespace(rid=rid, is_single=not isinstance(rid, list))


def generate(manager, rid):
    async def first():
        return await manager.generate_request(request(rid)).__anext__()

    return asyncio.run(first())


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(
        early_abort, "_table", early_abort.EarlyAbortTable(60.0, 4, clock=clock)
    )
    monkeypatch.setattr(early_abort, "_is_generation_request", lambda obj: True)
    monkeypatch.setattr(early_abort, "_abort_req", lambda rid: SimpleNamespace(rid=rid))
    return clock


def test_entry_expires_after_its_ttl(clock):
    table = early_abort.table()
    table.record("a")
    clock.now += 59.9
    assert len(table) == 1
    clock.now += 0.2
    assert len(table) == 0
    assert not table.consume("a")
    assert table.snapshot().expired == 1


def test_consume_removes_the_entry(clock):
    table = early_abort.table()
    table.record("a")
    assert table.consume("a")
    assert not table.consume("a")
    assert table.snapshot().rejected_on_arrival == 1


def test_repeat_restarts_the_clock(clock):
    table = early_abort.table()
    table.record("a")
    clock.now += 50
    table.record("b")
    table.record("a")
    clock.now += 20
    # "b" and the renewed "a" are both live; the original "a" expiry no longer applies.
    assert table.consume("a")
    assert table.consume("b")


def test_full_table_evicts_oldest(clock):
    table = early_abort.table()
    for rid in "abcde":
        clock.now += 1
        table.record(rid)
    assert len(table) == 4
    assert not table.consume("a")
    assert table.consume("e")
    assert table.snapshot().evicted == 1


def test_expired_entries_free_space_before_eviction(clock):
    table = early_abort.table()
    for rid in "abcd":
        table.record(rid)
    clock.now += 61
    table.record("e")
    snapshot = table.snapshot()
    assert (snapshot.size, snapshot.evicted, snapshot.expired) == (1, 0, 4)


@pytest.mark.parametrize(
    ("ttl", "entries"), [(0, 1), (-1, 1), (float("nan"), 1), (float("inf"), 1), (1, 0)]
)
def test_rejects_invalid_bounds(ttl, entries):
    with pytest.raises(ValueError):
        early_abort.EarlyAbortTable(ttl, entries)


def test_abort_before_registration_aborts_the_request_on_arrival(clock):
    manager = FakeManager()
    manager.abort_request(rid="late")
    assert manager.scheduler_aborts == []

    late = generate(manager, "late")
    other = generate(manager, "other")

    assert late["meta_info"] == {"id": "late", "finish_reason": {"type": "abort"}}
    assert other["meta_info"]["finish_reason"] == {"type": "length"}
    assert manager.scheduled == ["other"]
    assert manager.rid_to_state == {}
    assert len(early_abort.table()) == 0


def test_abort_of_active_request_takes_sglang_path(clock):
    manager = FakeManager()
    manager._init_req_state(request("active"))
    manager.abort_request(rid="active")

    assert manager.scheduler_aborts == ["active"]
    snapshot = early_abort.stats()
    assert (
        snapshot.exact_aborts,
        snapshot.active_aborts,
        snapshot.recorded,
        snapshot.size,
    ) == (1, 1, 0, 0)


def test_expired_abort_lets_the_request_run(clock):
    manager = FakeManager()
    manager.abort_request(rid="slow")
    clock.now += 61
    assert generate(manager, "slow")["meta_info"]["finish_reason"] == {"type": "length"}
    assert manager.scheduled == ["slow"]


def test_batches_are_left_alone(clock):
    manager = FakeManager()
    manager.abort_request(rid="b1")
    generate(manager, ["b0", "b1"])
    assert manager.scheduled == ["b0", "b1"]
    assert len(early_abort.table()) == 1


def test_long_and_health_check_rids_are_not_retained(clock):
    manager = FakeManager()
    manager.abort_request(rid="x" * (early_abort.MAX_RID_LENGTH + 1))
    manager.abort_request(rid="y" * early_abort.MAX_RID_LENGTH)
    manager.abort_request(rid=early_abort.HEALTH_CHECK_RID_PREFIX + "_1")
    snapshot = early_abort.stats()
    assert (snapshot.exact_aborts, snapshot.recorded) == (3, 1)


def test_request_runs_if_sglang_does_not_finish_it(clock, monkeypatch):
    manager = FakeManager()
    monkeypatch.setattr(manager, "_handle_abort_req", lambda req: None)
    manager.abort_request(rid="kept")
    assert generate(manager, "kept")["meta_info"]["finish_reason"] == {"type": "length"}
    assert manager.scheduled == ["kept"]


@pytest.mark.parametrize(
    "kwargs", [{"abort_all": True}, {"rid": "x", "prefix": True}, {"rid": ""}]
)
def test_non_exact_aborts_are_not_recorded(clock, kwargs):
    FakeManager().abort_request(**kwargs)
    assert early_abort.stats().recorded == 0


def test_positional_abort_arguments_pass_through(clock):
    manager = FakeManager()
    manager._init_req_state(request("p"))
    manager.abort_request("p", False, False)
    assert manager.scheduler_aborts == ["p"]


def test_hooks_are_inert_without_a_table(monkeypatch):
    monkeypatch.setattr(early_abort, "_table", None)
    manager = FakeManager()
    manager.abort_request(rid="x")
    generate(manager, "x")
    assert manager.scheduled == ["x"]


def test_install_registers_the_hooks(monkeypatch):
    monkeypatch.setattr(early_abort, "_table", None)
    monkeypatch.setenv(early_abort.TTL_ENV, "120")
    monkeypatch.setenv(early_abort.MAX_ENTRIES_ENV, "10")
    registered = []
    assert early_abort.install(lambda *args: registered.append(args))
    assert registered == [
        (early_abort.ABORT_TARGET, early_abort.abort_request_hook, "AROUND"),
        (early_abort.REGISTER_TARGET, early_abort.init_req_state_hook, "AROUND"),
        (early_abort.GENERATE_TARGET, early_abort.generate_request_hook, "AROUND"),
    ]
    assert (early_abort.table().ttl_seconds, early_abort.table().max_entries) == (
        120.0,
        10,
    )


def test_zero_ttl_disables_the_plugin(monkeypatch):
    monkeypatch.setattr(early_abort, "_table", None)
    monkeypatch.setenv(early_abort.TTL_ENV, "0")
    registered = []
    assert not early_abort.install(lambda *args: registered.append(args))
    assert registered == []
    assert early_abort.table() is None


def test_entry_point_is_declared():
    metadata = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    plugins = metadata["project"]["entry-points"]["sglang.srt.plugins"]
    assert plugins == {"olmo_sglang_early_abort": "olmo_sglang.early_abort:install"}
