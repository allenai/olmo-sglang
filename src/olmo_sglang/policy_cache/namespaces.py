"""Request ownership survives publication; sharing is limited to one policy."""

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Binding:
    version: int | None
    client_salt: str | None

    @property
    def salt(self):
        return json.dumps(
            ["olmo-policy-cache-v1", self.version, self.client_salt],
            separators=(",", ":"),
        )


def version_number(value):
    if value is None or value == "default":
        return None
    if not isinstance(value, str) or not value.isascii() or not value.isdigit():
        raise ValueError(
            "Policy caching requires nonnegative integer weight-version labels"
        )
    return int(value)


class Namespaces:
    def __init__(self, version):
        self.version = version_number(version)

    def observe(self, version):
        current = version_number(version)
        if self.version is not None and (current is None or current < self.version):
            raise RuntimeError("Policy cache weight versions cannot move backwards")
        self.version = current

    def bind(self, req):
        if req.session is not None or req.multimodal_inputs is not None:
            raise ValueError("Policy caching supports single-turn text requests")
        binding = getattr(req, "_olmo_policy_cache_binding", None)
        owns_state = (
            req.kv.holds_kv
            or req.kv.holds_mamba
            or req.lock_receipt.node_id is not None
        )
        if binding is None:
            if owns_state:
                raise RuntimeError("Request acquired state before policy-cache binding")
            binding = Binding(self.version, req.cache_salt)
        elif binding.version != self.version:
            if owns_state:
                return False
            binding = Binding(self.version, binding.client_salt)
        req._olmo_policy_cache_binding = binding
        req.cache_salt = binding.salt
        return True
