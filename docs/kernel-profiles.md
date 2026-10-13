# Reproducible FLA kernel choices

Kernel profiles fix autotuning choices independently of Core arithmetic mode.
They currently manage six FLA KDA forward autotuners (`fla-kda-forward-v1`).
They do not cover every serving operator, Core training/backward kernels, or
eliminate the numerical difference between prefill and decode.

## Strict replay

Set these variables on every equivalent serving worker:

```sh
export OLMO_SGLANG_CORE_COMPAT=rounding
export OLMO_SGLANG_REQUIRE_CORE_ROUNDING=1
export OLMO_SGLANG_KERNEL_PROFILE=/shared/profiles/h045-b300.json
export OLMO_SGLANG_KERNEL_PROFILE_MODE=strict
export OLMO_SGLANG_KERNEL_PROFILE_SHA256=<canonical-content-hash>
export OLMO_SGLANG_KERNEL_PROFILE_REPORT='/output/kernel-profile-{pid}.json'
```

Obtain the canonical content hash with:

```sh
python -m olmo_sglang.kernel_profile inspect /shared/profiles/h045-b300.json
```

This hash identifies parsed JSON content, not whitespace in the file; it is not
the output of `sha256sum` on the file. Workers check it before installing the
profile. A mismatch in GPU architecture/model, driver, CUDA/Python package
versions, arithmetic, model geometry, or the recorded implementation hashes
fails startup. Every choice must be available in the installed autotuner.

The profile is installed during model construction, before serving warmup and
CUDA graph capture. It replaces restored tuning entries for the managed kernels.
An uncovered tuning key raises an error instead of benchmarking new choices or
consulting an on-disk tuning cache. A different profile cannot replace an active
profile in the same process. Usage reports identify the profile and observed
keys; reports are written only when coverage changes, not for every token.
Use separate report paths for workers on different nodes as well as processes.

Keep compiler caches private and writable per worker. Their compiled binaries
may be restored from compatible shared snapshots; they are not the authority
for kernel selection. The immutable profile is that authority. Rebuild and
requalify it when the bound runtime or implementation changes.

## Recording and qualification

A preparation server can observe the managed autotuners with:

```sh
export OLMO_SGLANG_KERNEL_PROFILE_MODE=record
export OLMO_SGLANG_KERNEL_PROFILE=/shared/staging/candidate.json
export OLMO_SGLANG_DIAGNOSTIC_OVERRIDE_REASON='representative workload qualification'
```

Recording reserves a new output path and refuses to overwrite an existing file.
Exercise representative sequence lengths, batches, teacher contexts and serving
paths. The file is updated atomically as choices appear. An incomplete profile
is rejected by `inspect` and strict loading. Recording is a diagnostic mode,
not a qualified training default: compare repeated timings, matched-worker
outputs, and throughput before promoting a candidate to a fixed profile.
A tuning key may omit sequence length; presence of a key does not certify all
sequence lengths or batching patterns. Maintain qualification evidence with the
profile and expand that evidence deliberately.

For migration of our saved diagnostic maps, run on the intended GPU/runtime:

```sh
python -m olmo_sglang.kernel_profile import-replay saved-tuning.json candidate.json \
  --model /path/to/hf-checkpoint --arithmetic rounding
```

This converts actual candidate configurations into the structured schema and
binds the runtime identity. It does not certify the imported choices. Numerical
experiments must distinguish diagnostic conversion from production qualification.

Without a configured profile the library preserves ordinary serving behavior.
If a profile is configured, strict mode is the default; explicitly disabling it
or recording new choices requires `OLMO_SGLANG_DIAGNOSTIC_OVERRIDE_REASON`.
The strong Core-rounding requirement remains a separate contract. No universally
qualified profile is bundled or silently selected for unknown hardware/models.
