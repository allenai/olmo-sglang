"""Early-abort table and hook contract, without the SGLang runtime."""

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
    """The two TokenizerManager methods the hooks wrap, reduced to their rid bookkeeping."""

    def __init__(self):
        self.rid_to_state = {}
        self.scheduler_aborts = []

    def abort_request(self, rid="", abort_all=False, prefix=False):
        # Pinned single-tokenizer behaviour: an exact abort for an unknown rid is dropped.
        if not abort_all and not prefix and rid not in self.rid_to_state:
            return
        self.scheduler_aborts.append(rid)

    def init_req_state(self, obj):
        rids = [obj.rid] if obj.is_single else obj.rid
        for rid in rids:
            if rid in self.rid_to_state:
                raise ValueError(f"Duplicate request ID detected: {rid}")
            self.rid_to_state[rid] = SimpleNamespace(abort_before_dispatch=False)


def hooked_abort(manager, *args, **kwargs):
    return early_abort.abort_request_hook(
        FakeManager.abort_request, manager, *args, **kwargs
    )


def hooked_register(manager, obj):
    return early_abort.init_req_state_hook(manager.init_req_state(obj), manager, obj)


def request(rid):
    return SimpleNamespace(rid=rid, is_single=not isinstance(rid, list))


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(
        early_abort, "_table", early_abort.EarlyAbortTable(60.0, 4, clock=clock)
    )
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


@pytest.mark.parametrize(("ttl", "entries"), [(0, 1), (-1, 1), (1, 0)])
def test_rejects_invalid_bounds(ttl, entries):
    with pytest.raises(ValueError):
        early_abort.EarlyAbortTable(ttl, entries)


def test_abort_before_registration_aborts_the_request_on_arrival(clock):
    manager = FakeManager()
    hooked_abort(manager, rid="late")
    assert manager.scheduler_aborts == []

    hooked_register(manager, request("late"))
    hooked_register(manager, request("other"))

    assert manager.rid_to_state["late"].abort_before_dispatch
    assert not manager.rid_to_state["other"].abort_before_dispatch
    assert len(early_abort.table()) == 0


def test_abort_of_active_request_takes_sglang_path(clock):
    manager = FakeManager()
    hooked_register(manager, request("active"))
    hooked_abort(manager, rid="active")

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
    hooked_abort(manager, rid="slow")
    clock.now += 61
    hooked_register(manager, request("slow"))
    assert not manager.rid_to_state["slow"].abort_before_dispatch


def test_batch_registration_flags_only_aborted_rids(clock):
    manager = FakeManager()
    hooked_abort(manager, rid="b1")
    hooked_register(manager, request(["b0", "b1"]))
    assert not manager.rid_to_state["b0"].abort_before_dispatch
    assert manager.rid_to_state["b1"].abort_before_dispatch


@pytest.mark.parametrize(
    "kwargs", [{"abort_all": True}, {"rid": "x", "prefix": True}, {"rid": ""}]
)
def test_non_exact_aborts_are_not_recorded(clock, kwargs):
    hooked_abort(FakeManager(), **kwargs)
    assert early_abort.stats().recorded == 0


def test_positional_abort_arguments_pass_through(clock):
    manager = FakeManager()
    hooked_register(manager, request("p"))
    hooked_abort(manager, "p", False, False)
    assert manager.scheduler_aborts == ["p"]


def test_hooks_are_inert_without_a_table(monkeypatch):
    monkeypatch.setattr(early_abort, "_table", None)
    manager = FakeManager()
    hooked_abort(manager, rid="x")
    hooked_register(manager, request("x"))
    assert not manager.rid_to_state["x"].abort_before_dispatch


def test_install_registers_both_hooks(monkeypatch):
    monkeypatch.setattr(early_abort, "_table", None)
    monkeypatch.setenv(early_abort.TTL_ENV, "120")
    monkeypatch.setenv(early_abort.MAX_ENTRIES_ENV, "10")
    registered = []
    assert early_abort.install(lambda *args: registered.append(args))
    assert registered == [
        (early_abort.ABORT_TARGET, early_abort.abort_request_hook, "AROUND"),
        (early_abort.REGISTER_TARGET, early_abort.init_req_state_hook, "AFTER"),
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
