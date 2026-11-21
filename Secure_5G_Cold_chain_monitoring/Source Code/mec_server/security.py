"""HMAC-SHA256 signing/verification, telemetry replay guard, command counter."""
import hashlib
import hmac
import json
import math
import os
import tempfile
import time

import config


class VerificationError(Exception):
    """A message must be rejected. str(e) is the human-readable reason."""


# ---------------------------------------------------------------- crypto
def canonical_bytes(payload: dict) -> bytes:
    """Deterministic serialization: sorted keys, no whitespace, no NaN."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def compute_sig(payload: dict, key: bytes = None) -> str:
    key = key or config.SHARED_KEY
    return hmac.new(key, canonical_bytes(payload), hashlib.sha256).hexdigest()


def make_envelope(payload: dict, key: bytes = None) -> dict:
    return {"payload": payload, "sig": compute_sig(payload, key)}


def verify_envelope(raw, key: bytes = None) -> dict:
    """Parse + authenticate. Returns the payload dict or raises VerificationError."""
    if isinstance(raw, (bytes, bytearray)) and len(raw) > config.MAX_MESSAGE_BYTES:
        raise VerificationError("message too large")
    try:
        env = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        raise VerificationError("malformed JSON")
    if not isinstance(env, dict):
        raise VerificationError("envelope is not an object")
    payload, sig = env.get("payload"), env.get("sig")
    if not isinstance(payload, dict):
        raise VerificationError("missing payload")
    if not isinstance(sig, str) or not sig:
        raise VerificationError("missing signature")
    try:
        expected = compute_sig(payload, key)
    except (TypeError, ValueError):
        raise VerificationError("payload not serializable")
    if not hmac.compare_digest(expected.encode("ascii"), sig.encode("utf-8", "replace")):
        raise VerificationError("bad signature")
    return payload


# ------------------------------------------------------- atomic file I/O
def atomic_write_json(path: str, obj) -> None:
    """Write temp file -> fsync -> atomic rename, so a crash never leaves half a file."""
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
    try:  # make the rename durable (not possible on Windows; ignore)
        dfd = os.open(d, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except OSError:
        pass


def _read_json_or_none(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        raise RuntimeError(f"State file {path} is corrupted ({e}). "
                           "Fix or delete it deliberately; the server will not guess.")


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


# --------------------------------------------------------- telemetry guard
class TelemetryGuard:
    """Field validation + freshness + strictly increasing seq per device."""

    def __init__(self, path: str = None):
        self.path = path or config.TELEMETRY_STATE_FILE
        data = _read_json_or_none(self.path)
        self.last_seq = dict(data) if isinstance(data, dict) else {}

    def check_and_accept(self, payload: dict, now: float = None) -> dict:
        now = time.time() if now is None else now
        dev, seq = payload.get("device_id"), payload.get("seq")
        ts, temp = payload.get("timestamp"), payload.get("temperature")
        if dev not in config.ALLOWED_DEVICES:
            raise VerificationError("unknown device_id")
        if not _is_int(seq) or seq < 0:
            raise VerificationError("invalid seq")
        if not _is_num(ts):
            raise VerificationError("invalid timestamp")
        if not _is_num(temp) or not math.isfinite(temp) or \
                not config.TEMP_MIN_PLAUSIBLE <= temp <= config.TEMP_MAX_PLAUSIBLE:
            raise VerificationError("invalid temperature")
        if abs(now - ts) > config.MAX_CLOCK_SKEW_S:
            raise VerificationError(f"stale/future timestamp (skew {now - ts:+.0f}s)")
        if seq <= self.last_seq.get(dev, -1):
            raise VerificationError(f"replayed/old seq {seq} (last {self.last_seq[dev]})")
        self.last_seq[dev] = seq
        atomic_write_json(self.path, self.last_seq)
        return payload


def check_event_fresh(payload: dict, now: float = None) -> None:
    now = time.time() if now is None else now
    if payload.get("device_id") not in config.ALLOWED_DEVICES:
        raise VerificationError("unknown device_id")
    ts = payload.get("timestamp")
    if not _is_num(ts) or abs(now - ts) > config.MAX_CLOCK_SKEW_S:
        raise VerificationError("stale/invalid event timestamp")


# --------------------------------------------------------- command counter
class CommandCounter:
    """Persistent, strictly increasing command nonce. Persisted BEFORE use.

    On first run (no file) it is seeded from the clock so that a wiped state
    file still produces nonces larger than any previously issued one.
    """

    def __init__(self, path: str = None):
        self.path = path or config.COMMAND_STATE_FILE
        data = _read_json_or_none(self.path)
        if data is None:
            self.last = int(time.time())
            atomic_write_json(self.path, {"last_nonce": self.last})
        else:
            n = data.get("last_nonce") if isinstance(data, dict) else None
            if not _is_int(n) or n < 0:
                raise RuntimeError(f"{self.path} has invalid content; fix deliberately.")
            self.last = n

    def next(self) -> int:
        n = self.last + 1
        atomic_write_json(self.path, {"last_nonce": n})   # persist first
        self.last = n
        return n


def make_command(device_id: str, nonce: int, pwm: int, now: float = None, key: bytes = None) -> dict:
    now = time.time() if now is None else now
    return make_envelope({"device_id": device_id, "nonce": nonce,
                          "timestamp": int(now), "command": "set_cooling",
                          "pwm": pwm}, key)
