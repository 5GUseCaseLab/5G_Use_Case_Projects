import json
import time
import pytest
import config
import security
from security import CommandRejected, StateError, NonceStore

DEV = config.DEVICE_ID


def cmd(nonce=17, pwm=150, ts=None, dev=DEV, key=None, **extra):
    p = {"device_id": dev, "nonce": nonce, "timestamp": int(ts or time.time()),
         "command": "set_cooling", "pwm": pwm}
    p.update(extra)
    return json.dumps(security.make_envelope(p, key))


@pytest.fixture
def path(tmp_path):
    p = str(tmp_path / "last_nonce.json")
    security.provision_state(p, DEV, 0)
    return p


def accept(raw, store):
    p = security.verify_command(raw, store)
    store.persist(p["nonce"])
    return p


def test_valid_command_accepted(path):
    assert accept(cmd(17), NonceStore(path))["pwm"] == 150


def test_replay_rejected_after_restart(path):           # the key test from your plan
    accept(cmd(17), NonceStore(path))
    restarted = NonceStore(path)                         # new process, same file
    with pytest.raises(CommandRejected, match="replayed"):
        security.verify_command(cmd(17), restarted)
    assert accept(cmd(18), restarted)["nonce"] == 18


def test_lower_nonce_rejected(path):
    s = NonceStore(path)
    accept(cmd(20), s)
    with pytest.raises(CommandRejected):
        security.verify_command(cmd(19), s)


def test_tampered_pwm_rejected(path):
    env = json.loads(cmd(30, pwm=10))
    env["payload"]["pwm"] = 255
    with pytest.raises(CommandRejected, match="bad signature"):
        security.verify_command(json.dumps(env), NonceStore(path))


def test_wrong_key_rejected(path):
    with pytest.raises(CommandRejected, match="bad signature"):
        security.verify_command(cmd(30, key=b"evil"), NonceStore(path))


def test_wrong_device_rejected(path):
    with pytest.raises(CommandRejected, match="another device"):
        security.verify_command(cmd(30, dev="other"), NonceStore(path))


def test_stale_command_rejected(path):
    with pytest.raises(CommandRejected, match="stale"):
        security.verify_command(cmd(30, ts=time.time() - 600), NonceStore(path))


@pytest.mark.parametrize("pwm", [-1, 256, 1.5, "100", True, None])
def test_bad_pwm_rejected(path, pwm):
    with pytest.raises(CommandRejected):
        security.verify_command(cmd(30, pwm=pwm), NonceStore(path))


def test_extra_fields_rejected(path):
    with pytest.raises(CommandRejected, match="unexpected"):
        security.verify_command(cmd(30, admin=True), NonceStore(path))


def test_garbage_rejected(path):
    for raw in (b"xx", b"[]", b'{"payload":{}}'):
        with pytest.raises(CommandRejected):
            security.verify_command(raw, NonceStore(path))


def test_missing_state_fails_closed(tmp_path):
    s = NonceStore(str(tmp_path / "nope.json"))
    assert s.status == "missing"
    with pytest.raises(CommandRejected, match="fail closed"):
        security.verify_command(cmd(1), s)
    with pytest.raises(StateError):
        s.persist(1)


def test_corrupt_state_fails_closed_not_reset(path):
    open(path, "w").write("{broken")
    s = NonceStore(path)
    assert s.status == "corrupt"
    with pytest.raises(CommandRejected):
        security.verify_command(cmd(1), s)


def test_unwritable_state_fails_closed(path, monkeypatch):
    s = NonceStore(path)
    p = security.verify_command(cmd(5), s)
    monkeypatch.setattr(security, "atomic_write_json", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(StateError):
        s.persist(p["nonce"])
    assert s.last_nonce == 0                              # not advanced -> caller must not actuate
