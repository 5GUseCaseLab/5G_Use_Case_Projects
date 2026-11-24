"""Command authentication, anti-replay, and atomic persistence on the UNO Q."""
import hashlib
import hmac
import json
import os
import tempfile
import time

import config


class CommandRejected(Exception):
    """A command failed validation. str(e) is the reason."""


class StateError(Exception):
    """Persistent state missing/corrupt/unwritable -> fail closed."""


# ---------------------------------------------------------------- crypto
def canonical_bytes(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def compute_sig(payload: dict, key: bytes = None) -> str:
    return hmac.new(key or config.SHARED_KEY, canonical_bytes(payload), hashlib.sha256).hexdigest()


def make_envelope(payload: dict, key: bytes = None) -> dict:
    return {"payload": payload, "sig": compute_sig(payload, key)}


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


# ------------------------------------------------------- atomic file I/O
def atomic_write_json(path: str, obj) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    try:
        dfd = os.open(d, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except OSError:
        pass


# ------------------------------------------------------------ nonce store
class NonceStore:
    """Last accepted command nonce, stored on disk.

    status: 'ok' | 'missing' | 'corrupt'. Anything but 'ok' means commands are
    refused (fail closed) until a human runs provision_state.py.
    """

    def __init__(self, path: str = None, device_id: str = None):
        self.path = path or config.NONCE_FILE
        self.device_id = device_id or config.DEVICE_ID
        self.last_nonce = None
        self.status = "missing"
        self.load()

    def load(self):
        self.last_nonce, self.status = None, "missing"
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            n = data["last_nonce"]
            if data["device_id"] != self.device_id or not _is_int(n) or n < 0:
                raise ValueError("bad content")
            self.last_nonce, self.status = n, "ok"
        except (OSError, ValueError, KeyError, TypeError):
            self.status = "corrupt"

    @property
    def ready(self) -> bool:
        return self.status == "ok"

    def persist(self, nonce: int) -> None:
        """Durably store `nonce`. Call BEFORE actuating. Raises StateError on any problem."""
        if not self.ready:
            raise StateError(f"nonce state is {self.status}; run provision_state.py")
        if not _is_int(nonce) or nonce <= self.last_nonce:
            raise StateError("refusing to persist a non-increasing nonce")
        try:
            atomic_write_json(self.path, {"device_id": self.device_id, "last_nonce": nonce})
        except OSError as e:
            raise StateError(f"cannot persist nonce: {e}")
        self.last_nonce = nonce


def provision_state(path: str, device_id: str, nonce: int = 0) -> None:
    """Explicit, trusted creation/reset of the nonce file (used by provision_state.py)."""
    atomic_write_json(path, {"device_id": device_id, "last_nonce": nonce})


# ------------------------------------------------------- command verifier
COMMAND_FIELDS = {"device_id", "nonce", "timestamp", "command", "pwm"}


def verify_command(raw, store: NonceStore, now: float = None, key: bytes = None) -> dict:
    """Validate in this order: envelope -> HMAC -> fields -> device -> freshness -> nonce.

    Returns the payload. Does NOT persist the nonce; the caller must call
    store.persist(nonce) and only then actuate.
    """
    now = time.time() if now is None else now
    if isinstance(raw, (bytes, bytearray)) and len(raw) > config.MAX_MESSAGE_BYTES:
        raise CommandRejected("message too large")
    try:
        env = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        raise CommandRejected("malformed JSON")
    if not isinstance(env, dict) or not isinstance(env.get("payload"), dict):
        raise CommandRejected("missing payload")
    sig, payload = env.get("sig"), env["payload"]
    if not isinstance(sig, str) or not sig:
        raise CommandRejected("missing signature")

    try:                                        # HMAC before trusting ANY field
        expected = compute_sig(payload, key)
    except (TypeError, ValueError):
        raise CommandRejected("payload not serializable")
    if not hmac.compare_digest(expected.encode("ascii"), sig.encode("utf-8", "replace")):
        raise CommandRejected("bad signature")

    if set(payload) != COMMAND_FIELDS:
        raise CommandRejected("unexpected or missing fields")
    if payload["command"] != "set_cooling":
        raise CommandRejected("unknown command")
    pwm = payload["pwm"]
    if not _is_int(pwm) or not 0 <= pwm <= 255:
        raise CommandRejected("pwm out of range")
    if payload["device_id"] != config.DEVICE_ID:
        raise CommandRejected("command for another device")
    ts = payload["timestamp"]
    if isinstance(ts, bool) or not isinstance(ts, (int, float)):
        raise CommandRejected("invalid timestamp")
    if abs(now - ts) > config.COMMAND_MAX_AGE_S:
        raise CommandRejected(f"stale/future timestamp (skew {now - ts:+.0f}s)")
    nonce = payload["nonce"]
    if not _is_int(nonce) or nonce <= 0:
        raise CommandRejected("invalid nonce")
    if not store.ready:
        raise CommandRejected(f"nonce state {store.status} (fail closed)")
    if nonce <= store.last_nonce:
        raise CommandRejected(f"replayed/old nonce {nonce} (last {store.last_nonce})")
    return payload
