# SPDX-License-Identifier: Apache-2.0

"""Remember aborts that arrive before their request, and abort it on arrival.

A client that abandons a request does not know which engine its router chose,
so it sends one exact-rid ``/abort_request`` to every engine. An abort can then
reach the engine before the ``/generate`` it cancels: the two travel on
separate connections, and the router may still be holding the generation. At
the pinned SGLang revision the tokenizer manager acknowledges an abort for an
rid it has not registered and forgets it, so that generation later decodes to
the end for nobody.

This plugin keeps a bounded table of such early aborts, each for a fixed time
to live. When ``generate_request`` registers a single generation whose rid is in
the table, the generation is finished at once through SGLang's own
``_handle_abort_req`` and answered through SGLang's own response path, so the
client gets the usual abort response and the scheduler never sees the request.
Aborts for an rid the engine already holds take SGLang's normal path.

The abort hook and the registration hook wrap synchronous methods that run on
the tokenizer manager's event loop, so a registration and an abort for the
same rid never interleave: whichever runs second sees what the first did. The
table still takes a lock, so a call from another thread cannot corrupt it.

SGLang loads this module through its ``sglang.srt.plugins`` entry point in
every server process. ``OLMO_SGLANG_EARLY_ABORT_TTL_SECONDS=0`` turns it off.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

TTL_ENV = "OLMO_SGLANG_EARLY_ABORT_TTL_SECONDS"
MAX_ENTRIES_ENV = "OLMO_SGLANG_EARLY_ABORT_MAX_ENTRIES"
# An entry must outlive the longest gap between a client's abort and the
# engine registering that request, which router queueing dominates. Thirty
# minutes covers the longest MILES request deadlines in use.
DEFAULT_TTL_SECONDS = 1800.0
# Every engine stores every broadcast abort, though at most one ever sees the
# request. 65,536 short rids take a few MiB.
DEFAULT_MAX_ENTRIES = 65_536

_TOKENIZER_MANAGER = "sglang.srt.managers.tokenizer_manager.TokenizerManager"
ABORT_TARGET = f"{_TOKENIZER_MANAGER}.abort_request"
REGISTER_TARGET = f"{_TOKENIZER_MANAGER}._init_req_state"
GENERATE_TARGET = f"{_TOKENIZER_MANAGER}.generate_request"
# The pinned-revision methods the hooks call to finish and answer a request.
_REQUIRED_METHODS = ("_handle_abort_req", "_stream_one_response")
# Longer rids are not retained, so abort callers cannot pin unbounded bytes.
MAX_RID_LENGTH = 256
# sglang.srt.constants.HEALTH_CHECK_RID_PREFIX: SGLang's abort handler ignores
# these rids, so the plugin must never finish one itself.
HEALTH_CHECK_RID_PREFIX = "HEALTH_CHECK"
ABORT_MESSAGE = "Aborted on arrival: an abort for this rid arrived first"


@dataclass
class EarlyAbortStats:
    """Counters for one process; read them with ``stats()``."""

    exact_aborts: int = 0  # exact-rid aborts received
    active_aborts: int = 0  # of those, for an rid the engine already held
    recorded: int = 0  # of those, for an unknown rid, so added to the table
    rejected_on_arrival: int = 0  # registrations that matched a live entry
    expired: int = 0
    evicted: int = 0  # dropped while still live because the table was full
    size: int = 0


class EarlyAbortTable:
    """Rids aborted before registration, each kept until its expiry.

    Entries are ordered by insertion; with one time to live that is also expiry
    order, so expired entries are trimmed from the front, and a full table
    evicts its oldest entry.
    """

    def __init__(
        self,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        clock: Callable[[], float] = time.monotonic,
    ):
        if not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError(
                f"ttl_seconds must be positive and finite, got {ttl_seconds}"
            )
        if max_entries <= 0:
            raise ValueError(f"max_entries must be positive, got {max_entries}")
        self.ttl_seconds = float(ttl_seconds)
        self.max_entries = int(max_entries)
        self._clock = clock
        self._expiry: OrderedDict[str, float] = OrderedDict()
        self._lock = threading.Lock()
        self.stats = EarlyAbortStats()

    def record(self, rid: str) -> None:
        """Remember ``rid`` for one time to live; a repeat restarts its clock."""
        with self._lock:
            now = self._clock()
            self._trim(now)
            self._expiry.pop(rid, None)
            while len(self._expiry) >= self.max_entries:
                self._expiry.popitem(last=False)
                self.stats.evicted += 1
                if self.stats.evicted == 1 or self.stats.evicted % 1000 == 0:
                    logger.warning(
                        "Early-abort table is full (%d entries); evicted %d live entries so far. "
                        "A request that arrives after its entry was evicted will run. Raise %s.",
                        self.max_entries,
                        self.stats.evicted,
                        MAX_ENTRIES_ENV,
                    )
            self._expiry[rid] = now + self.ttl_seconds
            self.stats.recorded += 1

    def count_abort(self, *, active: bool) -> None:
        """Count one exact-rid abort, and whether the engine already held its rid."""
        with self._lock:
            self.stats.exact_aborts += 1
            if active:
                self.stats.active_aborts += 1

    def consume(self, rid: str) -> bool:
        """Remove ``rid`` and return whether it had a live entry."""
        with self._lock:
            self._trim(self._clock())
            if self._expiry.pop(rid, None) is None:
                return False
            self.stats.rejected_on_arrival += 1
            return True

    def __len__(self) -> int:
        with self._lock:
            self._trim(self._clock())
            return len(self._expiry)

    def snapshot(self) -> EarlyAbortStats:
        with self._lock:
            self._trim(self._clock())
            return EarlyAbortStats(**{**vars(self.stats), "size": len(self._expiry)})

    def _trim(self, now: float) -> None:
        while self._expiry:
            rid, expiry = next(iter(self._expiry.items()))
            if expiry > now:
                return
            del self._expiry[rid]
            self.stats.expired += 1


_table: EarlyAbortTable | None = None


def table() -> EarlyAbortTable | None:
    """This process's table, or None when the plugin is off or not loaded."""
    return _table


def stats() -> EarlyAbortStats | None:
    return None if _table is None else _table.snapshot()


class AbortedOnArrival(Exception):
    """Carries a request finished at registration out of ``generate_request``."""

    def __init__(self, rid: str, state):
        super().__init__(rid)
        self.rid = rid
        self.state = state


def abort_request_hook(
    original, manager, rid: str = "", abort_all: bool = False, prefix: bool = False
):
    """Around ``TokenizerManager.abort_request``: remember an exact abort for an unknown rid."""
    current = _table
    if (
        current is not None
        and isinstance(rid, str)
        and rid
        and not abort_all
        and not prefix
    ):
        active = rid in manager.rid_to_state
        current.count_abort(active=active)
        if not active and _retainable(rid):
            current.record(rid)
        snapshot = current.snapshot()
        if snapshot.exact_aborts % 1000 == 0:
            logger.info("Early-abort table: %s", snapshot)
    return original(manager, rid, abort_all, prefix)


def init_req_state_hook(original, manager, obj, request=None):
    """Around ``TokenizerManager._init_req_state``: finish a single generation aborted earlier.

    The request is registered, then finished by SGLang's ``_handle_abort_req``,
    which queues the abort response on its state and unregisters it. Raising
    ``AbortedOnArrival`` stops ``generate_request`` before the pause gate,
    tokenization and dispatch; ``generate_request_hook`` turns it into the
    response. Embeddings and batches, including ``n > 1``, are left alone.
    """
    result = original(manager, obj, request)
    current = _table
    if (
        current is None
        or not getattr(obj, "is_single", True)
        or not _is_generation_request(obj)
    ):
        return result
    rid = obj.rid
    if not isinstance(rid, str) or not current.consume(rid):
        return result
    state = manager.rid_to_state[rid]
    manager._handle_abort_req(_abort_req(rid))
    if not state.finished:
        # SGLang declined to finish it; let the request run rather than wait forever.
        logger.warning("Could not abort request %s on arrival; it will run", rid)
        return result
    logger.info(
        "Aborted request %s on arrival: its abort arrived first (%s)",
        rid,
        current.snapshot(),
    )
    raise AbortedOnArrival(rid, state)


def _retainable(rid: str) -> bool:
    return len(rid) <= MAX_RID_LENGTH and not rid.startswith(HEALTH_CHECK_RID_PREFIX)


def _is_generation_request(obj) -> bool:
    from sglang.srt.managers.io_struct import GenerateReqInput

    return isinstance(obj, GenerateReqInput)


def _abort_req(rid: str):
    from sglang.srt.managers.io_struct import AbortReq

    return AbortReq(rid=rid, abort_message=ABORT_MESSAGE)


def generate_request_hook(original, manager, obj, request=None):
    """Around ``TokenizerManager.generate_request``: answer a request aborted on arrival."""
    return _generate_request(original, manager, obj, request)


async def _generate_request(original, manager, obj, request):
    responses = original(manager, obj, request)
    try:
        first = await responses.__anext__()
    except AbortedOnArrival as aborted:
        # The same response path, and so the same response, as any aborted request.
        async for out in manager._stream_one_response(
            obj=obj, state=aborted.state, request=request
        ):
            yield out
        return
    except StopAsyncIteration:
        return
    try:
        yield first
        async for out in responses:
            yield out
    finally:
        await responses.aclose()


def _read_config() -> tuple[float, int]:
    ttl = float(os.environ.get(TTL_ENV, DEFAULT_TTL_SECONDS))
    max_entries = int(os.environ.get(MAX_ENTRIES_ENV, DEFAULT_MAX_ENTRIES))
    return ttl, max_entries


def install(register_hook: Callable | None = None) -> bool:
    """Create this process's table and register the two hooks; return whether it did.

    SGLang calls this through the ``sglang.srt.plugins`` entry point, then
    applies the registered hooks. ``register_hook`` replaces
    ``HookRegistry.register`` in tests.
    """
    global _table
    ttl, max_entries = _read_config()
    if ttl <= 0:
        logger.info("Early-abort table disabled by %s=%s", TTL_ENV, ttl)
        return False
    if register_hook is None:
        from sglang.srt.managers.tokenizer_manager import TokenizerManager
        from sglang.srt.plugins.hook_registry import HookRegistry, HookType

        missing = [
            name for name in _REQUIRED_METHODS if not hasattr(TokenizerManager, name)
        ]
        if missing:
            logger.warning(
                "Early-abort table not installed: this SGLang's TokenizerManager lacks %s",
                missing,
            )
            return False

        def register_hook(target, hook, kind):
            HookRegistry.register(target, hook, getattr(HookType, kind))

    if _table is None:
        _table = EarlyAbortTable(ttl, max_entries)
    register_hook(ABORT_TARGET, abort_request_hook, "AROUND")
    register_hook(REGISTER_TARGET, init_req_state_hook, "AROUND")
    register_hook(GENERATE_TARGET, generate_request_hook, "AROUND")
    logger.info("Early-abort table installed: ttl=%gs max_entries=%d", ttl, max_entries)
    return True
