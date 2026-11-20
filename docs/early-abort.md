# Early aborts

An RL producer that abandons a generation (a request deadline, or a failed
sibling in its group) sends `/abort_request` with the exact request id to
every engine, because the router does not tell it which engine took the
request. The abort can reach the chosen engine before the `/generate` it
cancels: they travel on separate connections, and the router may still be
queueing the generation. At the pinned SGLang revision the tokenizer manager
acknowledges an abort for an unregistered rid with HTTP 200 and forgets it.
The generation then registers and decodes to its token limit for nobody.

This plugin makes such an abort stick. It is installed through SGLang's
`sglang.srt.plugins` entry point and needs no SGLang change.

## Behaviour

- **Abort for a registered rid:** unchanged. SGLang's own path aborts it.
- **Exact abort for an unknown rid:** SGLang's behaviour is unchanged, and the
  rid is also recorded in a per-process table with a time to live.
- **Request registration** (`TokenizerManager._init_req_state`, the first step
  of `generate_request`, before the pause gate, the weight-update lock and
  tokenization): if the rid has a live entry, the entry is removed and the
  request gets SGLang's own `abort_before_dispatch` flag. The dispatch path
  then finishes it as aborted without sending it to the scheduler. The
  response is the one SGLang returns for any request aborted before dispatch:
  HTTP 200, `finish_reason = {"type": "abort", "message": "Aborted by AbortReq
  before dispatch to scheduler"}`, no output tokens.
- `abort_all`, prefix aborts and empty rids are not recorded.

Both hooks wrap synchronous methods that run on the tokenizer manager's event
loop, so an abort and a registration for one rid never interleave. Whichever
runs second sees the first. The table also takes a lock against calls from
other threads.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `OLMO_SGLANG_EARLY_ABORT_TTL_SECONDS` | `1800` | How long an entry lives. `0` disables the plugin. |
| `OLMO_SGLANG_EARLY_ABORT_MAX_ENTRIES` | `65536` | Table bound. A full table evicts its oldest entry. |

The time to live must exceed the longest delay between the client's abort and
the engine registering that request, which router queueing dominates. A
request that registers after its entry expired or was evicted runs as before.
Because each abort goes to every engine and at most one engine sees the
request, most entries simply expire.

If `SGLANG_PLUGINS` restricts which plugins load, include
`olmo_sglang_early_abort`. The entry point is registered when the package is
installed (`pip install -e .` included). Only a `PYTHONPATH` checkout does not
register it.

## Observability

The plugin logs at startup (`Early-abort table installed: ttl=... max_entries=...`),
once per request aborted on arrival, every 1,000 exact aborts with counters, and
when the table evicts live entries. `olmo_sglang.early_abort.stats()` returns the
process's counters: exact aborts received, aborts of registered rids, recorded
unknown rids, requests aborted on arrival, expired and evicted entries, and the
current size.

## Assumptions and limits

- Request ids are never reused. A retry must use a new rid, or the table would
  abort it. MILES generates a fresh `uuid4` per attempt.
- With `--tokenizer-worker-num > 1`, each tokenizer worker has its own table,
  and an abort only matches a request registered by the worker that received
  the abort.
- Only the HTTP and `Engine` paths through `TokenizerManager` are covered;
  the gRPC server is not.
