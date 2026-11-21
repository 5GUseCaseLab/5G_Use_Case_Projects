"""Shared JSON/UDP protocol validation for the UNO Q sender and server."""

import json
import math


VERSION = 1
MAX_DATAGRAM = 4096
EVENTS = {"SAFE", "WARNING", "CRITICAL", "NORMAL", "SOIL_DRY", "HIGH_ALERT", "WATER_CRITICAL", "FLOOD_RISK"}


def encode(message):
    payload = json.dumps(message, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(payload) > MAX_DATAGRAM:
        raise ValueError("datagram too large")
    return payload


def decode(data):
    if not data or len(data) > MAX_DATAGRAM:
        raise ValueError("invalid datagram size")
    message = json.loads(data)
    if not isinstance(message, dict) or message.get("version") != VERSION:
        raise ValueError("unsupported message")
    return message


def integer(value, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("invalid integer")
    return value


def number(value, minimum, maximum):
    if type(value) not in (int, float) or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError("invalid number")
    return float(value)


def identity(message):
    session = message.get("session")
    if not isinstance(session, str) or not 1 <= len(session) <= 64:
        raise ValueError("invalid session")
    seq = integer(message.get("seq"), 1, 2**32 - 1)
    return session, seq


def validate_timing(message):
    session, seq = identity(message)
    return {
        "session": session,
        "seq": seq,
        "latency_ms": number(message.get("latency_ms"), 0, 60000),
        "network_ms": number(message.get("network_ms"), 0, 60000),
        "rtt_ms": number(message.get("rtt_ms"), 0, 120000),
    }


def validate_sample(message):
    session, seq = identity(message)
    sample = {"session": session, "seq": seq}
    sample["sample_millis"] = integer(message.get("sample_millis"), 0, 2**32 - 1)
    sample["sample_age_ms"] = number(message.get("sample_age_ms"), 0, 60000)
    sample["clock_uncertainty_ms"] = number(message.get("clock_uncertainty_ms"), 0, 60000)
    for name in ("water_raw", "soil_raw", "rain_raw"):
        sample[name] = integer(message.get(name), 0, 1023)
    for name in ("water", "soil", "rain"):
        sample[name] = number(message.get(name), 0, 100)
    sample["priority"] = integer(message.get("priority"), 0, 3)
    if message.get("event") not in EVENTS:
        raise ValueError("invalid event")
    sample["event"] = message["event"]
    timings = message.get("timings", [])
    if not isinstance(timings, list) or len(timings) > 10:
        raise ValueError("invalid timings")
    sample["timings"] = [validate_timing(t) for t in timings]
    if any(t["session"] != session for t in sample["timings"]):
        raise ValueError("timing session mismatch")
    return sample
