"""Console + file logging, and CSV log of VERIFIED telemetry only."""
import csv
import logging
import os
import time

import config


def setup_logging():
    os.makedirs(config.LOG_DIR, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in (logging.StreamHandler(), logging.FileHandler(config.SERVER_LOG)):
        h.setFormatter(fmt)
        root.addHandler(h)


class TelemetryCsv:
    HEADER = ["recv_time", "device_id", "seq", "device_time", "temperature", "pwm_cmd", "alarm"]

    def __init__(self, path=None):
        self.path = path or config.TELEMETRY_CSV
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(self.HEADER)

    def write(self, payload, pwm, alarm):
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([int(time.time()), payload["device_id"], payload["seq"],
                                    payload["timestamp"], payload["temperature"], pwm, alarm or ""])


class LatencyCsv:
    """Every latency sample: kind 'rtt' (command round-trip) or 'oneway' (telemetry, needs synced clocks)."""
    HEADER = ["time", "kind", "ms", "ref"]

    def __init__(self, path=None):
        self.path = path or os.path.join(config.LOG_DIR, "latency.csv")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(self.HEADER)

    def write(self, kind, ms, ref):
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([round(time.time(), 3), kind, round(ms, 1), ref])
