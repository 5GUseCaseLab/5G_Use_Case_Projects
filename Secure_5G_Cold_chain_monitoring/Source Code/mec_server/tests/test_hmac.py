import json
import pytest
import security
from security import VerificationError


def packet(**over):
    p = {"device_id": "uno_q_edge_01", "seq": 1, "timestamp": 1000, "temperature": 4.25}
    p.update(over)
    return p


def test_valid_signature_accepted():
    env = security.make_envelope(packet())
    assert security.verify_envelope(json.dumps(env))["temperature"] == 4.25


def test_modified_temperature_rejected():
    env = security.make_envelope(packet())
    env["payload"]["temperature"] = 9.9           # tamper, keep old sig
    with pytest.raises(VerificationError, match="bad signature"):
        security.verify_envelope(json.dumps(env))


def test_wrong_key_rejected():
    env = security.make_envelope(packet(), key=b"attacker-key")
    with pytest.raises(VerificationError, match="bad signature"):
        security.verify_envelope(json.dumps(env))


def test_missing_or_empty_signature_rejected():
    env = security.make_envelope(packet())
    for bad in (None, ""):
        e = dict(env)
        if bad is None:
            e.pop("sig")
        else:
            e["sig"] = bad
        with pytest.raises(VerificationError, match="missing signature"):
            security.verify_envelope(json.dumps(e))


def test_garbage_rejected():
    for raw in (b"not json", b"[]", b'{"payload": 5, "sig": "x"}', b"\xff\xfe"):
        with pytest.raises(VerificationError):
            security.verify_envelope(raw)


def test_oversized_rejected():
    with pytest.raises(VerificationError, match="too large"):
        security.verify_envelope(b"{" + b" " * 10000 + b"}")


def test_key_order_does_not_matter():
    p = packet()
    env = {"sig": security.compute_sig(p), "payload": dict(reversed(list(p.items())))}
    assert security.verify_envelope(json.dumps(env)) == p
