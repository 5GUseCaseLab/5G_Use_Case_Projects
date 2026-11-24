"""Build signed telemetry packets with a persistent sequence number."""
import json
import os
import time

import config
import security


class TelemetryBuilder:
    def __init__(self, path: str = None, device_id: str = None):
        self.path = path or config.TELEMETRY_SEQ_FILE
        self.device_id = device_id or config.DEVICE_ID
        self.seq = self._load()

    def _load(self) -> int:
        try:
            with open(self.path) as f:
                n = json.load(f)["seq"]
            if isinstance(n, int) and not isinstance(n, bool) and n >= 0:
                return n
        except (OSError, ValueError, KeyError, TypeError):
            pass
        # Missing/corrupt: seed from the clock so seq still exceeds anything sent earlier.
        return int(time.time())

    def build(self, temperature: float, now: float = None) -> dict:
        now = time.time() if now is None else now
        seq = self.seq + 1
        security.atomic_write_json(self.path, {"seq": seq})     # persist before publish
        self.seq = seq
        payload = {"device_id": self.device_id, "seq": seq, "timestamp": round(now, 3),
                   "temperature": round(float(temperature), 2)}
        return security.make_envelope(payload)


def build_event(event: str, **detail) -> dict:
    payload = {"device_id": config.DEVICE_ID, "timestamp": int(time.time()), "event": event}
    payload.update(detail)
    return security.make_envelope(payload)
