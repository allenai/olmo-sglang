from types import SimpleNamespace

import pytest

from olmo_sglang.policy_cache.namespaces import Namespaces


def request(salt=None):
    return SimpleNamespace(
        cache_salt=salt,
        session=None,
        multimodal_inputs=None,
        kv=SimpleNamespace(holds_kv=False, holds_mamba=False),
        lock_receipt=SimpleNamespace(node_id=None),
    )


def test_waiting_rebinds_but_retained_state_stays_private():
    namespaces = Namespaces("0")
    waiting, active = request("a"), request("a")
    assert namespaces.bind(waiting) and namespaces.bind(active)
    old = active.cache_salt
    active.kv.holds_mamba = True
    namespaces.observe("1")
    assert not namespaces.bind(active)
    assert active.cache_salt == old
    assert namespaces.bind(waiting)
    assert waiting.cache_salt != old
    active.kv.holds_mamba = False
    assert namespaces.bind(active)
    assert active.cache_salt == waiting.cache_salt


def test_client_salts_survive_many_publications_without_accumulation():
    ns = Namespaces("default")
    left, right = request("a"), request("b")
    for version in range(50):
        ns.observe(str(version))
        ns.bind(left)
        ns.bind(right)
        assert left.cache_salt != right.cache_salt
    assert len(left.cache_salt) < 50


def test_same_policy_republication_keeps_its_namespace():
    ns, req = Namespaces("3"), request()
    ns.bind(req)
    before = req.cache_salt
    ns.observe("3")
    assert ns.bind(req)
    assert req.cache_salt == before


@pytest.mark.parametrize("version", ["default", None, "2", "abc", "-1", 4])
def test_version_cannot_regress_or_become_invalid(version):
    ns = Namespaces("3")
    with pytest.raises((ValueError, RuntimeError)):
        ns.observe(version)


def test_unbound_owned_state_is_rejected():
    req = request()
    req.kv.holds_kv = True
    with pytest.raises(RuntimeError, match="before policy-cache binding"):
        Namespaces("0").bind(req)
