"""Strict profile identity, coverage, and process-lifecycle contracts."""

import copy
import json
from types import SimpleNamespace

import pytest

from olmo_sglang import kernel_profile as profile


@pytest.fixture
def runtime(monkeypatch):
    first = SimpleNamespace(
        kwargs={"BV": 32}, num_warps=2, num_stages=3, num_ctas=1, maxnreg=None
    )
    second = SimpleNamespace(
        kwargs={"BV": 64}, num_warps=4, num_stages=2, num_ctas=1, maxnreg=None
    )
    tuners = {
        name: SimpleNamespace(cache={(99,): second}, configs=[first, second])
        for name in profile.KERNELS
    }
    monkeypatch.setattr(profile, "tuner", lambda name: tuners[name])
    monkeypatch.setattr(profile, "_ACTIVE", None)
    monkeypatch.setattr(
        profile, "identity", lambda *args: {"gpu": "B300", "dtype": "BF16"}
    )
    for key in (
        profile.PROFILE,
        profile.MODE,
        profile.DIGEST,
        profile.REPORT,
        profile.REASON,
    ):
        monkeypatch.delenv(key, raising=False)
    document = {
        "schema_version": 1,
        "scope": profile.SCOPE,
        "identity": {"gpu": "B300", "dtype": "BF16"},
        "kernels": {
            name: [
                {"key": [8, "torch.bfloat16"], "config": profile.configuration(first)}
            ]
            for name in profile.KERNELS
        },
        "provenance": {"kind": "test"},
    }
    return document, tuners, first


def setup(monkeypatch, tmp_path, document):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(document))
    monkeypatch.setenv(profile.PROFILE, str(path))
    monkeypatch.setenv(profile.DIGEST, profile.digest(document))
    return path


def test_strict_overrides_restored_cache_and_blocks_unknown_workload(
    runtime, tmp_path, monkeypatch
):
    document, tuners, first = runtime
    setup(monkeypatch, tmp_path, document)
    session = profile.configure(None, None, None)
    for name, tuner in tuners.items():
        assert tuner.cache[(8, "torch.bfloat16")] is first
        with pytest.raises(RuntimeError, match="coverage miss"):
            (99,) in tuner.cache
        with pytest.raises(RuntimeError, match="coverage miss"):
            tuner.cache[(123,)]
    assert len(session.used) == 6
    assert profile.configure(None, None, None) is session


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c.__setitem__((8, "torch.bfloat16"), None),
        lambda c: c.clear(),
        lambda c: c.update({}),
        lambda c: c.pop((8, "torch.bfloat16")),
        lambda c: c.popitem(),
        lambda c: c.__delitem__((8, "torch.bfloat16")),
    ],
)
def test_strict_cache_cannot_be_mutated(runtime, tmp_path, monkeypatch, mutation):
    document, tuners, _ = runtime
    setup(monkeypatch, tmp_path, document)
    profile.configure(None, None, None)
    with pytest.raises(RuntimeError):
        mutation(next(iter(tuners.values())).cache)


def test_cache_replacement_is_detected(runtime, tmp_path, monkeypatch):
    document, tuners, _ = runtime
    setup(monkeypatch, tmp_path, document)
    session = profile.configure(None, None, None)
    next(iter(tuners.values())).cache = {}
    with pytest.raises(RuntimeError, match="replaced"):
        session.observe()


@pytest.mark.parametrize(
    "change", ["digest", "runtime", "choice", "missing", "duplicate", "schema"]
)
def test_invalid_profiles_fail_before_mutating_any_tuner(
    runtime, tmp_path, monkeypatch, change
):
    document, tuners, _ = runtime
    if change == "runtime":
        document["identity"]["gpu"] = "H100"
    if change == "choice":
        document["kernels"][profile.KERNELS[-1]][0]["config"]["num_warps"] = 32
    if change == "missing":
        document["kernels"].pop(profile.KERNELS[-1])
    if change == "duplicate":
        document["kernels"][profile.KERNELS[-1]] *= 2
    if change == "schema":
        document["schema_version"] = 2
    setup(monkeypatch, tmp_path, document)
    if change == "digest":
        monkeypatch.setenv(profile.DIGEST, "0" * 64)
    with pytest.raises(ValueError):
        profile.configure(None, None, None)
    assert all(
        type(t.cache) is dict and list(t.cache) == [(99,)] for t in tuners.values()
    )


def test_different_profile_cannot_replace_active_session(
    runtime, tmp_path, monkeypatch
):
    document, _, _ = runtime
    setup(monkeypatch, tmp_path, document)
    profile.configure(None, None, None)
    modified = copy.deepcopy(document)
    modified["provenance"]["different"] = True
    setup(monkeypatch, tmp_path, modified)
    with pytest.raises(RuntimeError, match="Cannot mix"):
        profile.configure(None, None, None)


def test_record_requires_reason_and_does_not_overwrite(runtime, tmp_path, monkeypatch):
    _, tuners, first = runtime
    path = tmp_path / "record.json"
    monkeypatch.setenv(profile.PROFILE, str(path))
    monkeypatch.setenv(profile.MODE, "record")
    with pytest.raises(ValueError, match="diagnostic reason"):
        profile.configure(None, None, None)
    monkeypatch.setenv(profile.REASON, "representative-workload preparation")
    session = profile.configure(None, None, None)
    for t in tuners.values():
        t.cache[(8,)] = first
    session.observe()
    document = json.loads(path.read_text())
    profile.validate(document)
    assert len(document["kernels"][profile.KERNELS[0]]) == 2
    assert (
        document["provenance"]["diagnostic_reason"]
        == "representative-workload preparation"
    )
    monkeypatch.setattr(profile, "_ACTIVE", None)
    with pytest.raises(FileExistsError):
        profile.configure(None, None, None)


def test_report_records_content_identity_and_used_choices(
    runtime, tmp_path, monkeypatch
):
    document, tuners, _ = runtime
    setup(monkeypatch, tmp_path, document)
    report = tmp_path / "usage.json"
    monkeypatch.setenv(profile.REPORT, str(report))
    session = profile.configure(None, None, None)
    next(iter(tuners.values())).cache[(8, "torch.bfloat16")]
    session.observe()
    value = json.loads(report.read_text())
    assert value["profile_sha256"] == profile.digest(document)
    assert len(value["used"]) == 1
    before = report.stat().st_mtime_ns
    session.observe()
    assert report.stat().st_mtime_ns == before


def test_configured_profile_cannot_be_silently_disabled(runtime, tmp_path, monkeypatch):
    document, _, _ = runtime
    setup(monkeypatch, tmp_path, document)
    monkeypatch.setenv(profile.MODE, "off")
    with pytest.raises(ValueError, match="diagnostic reason"):
        profile.configure(None, None, None)
    monkeypatch.setenv(profile.REASON, "uncontrolled baseline")
    assert profile.configure(None, None, None) is None
