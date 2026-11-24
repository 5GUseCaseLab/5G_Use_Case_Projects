"""UDP telemetry receiver and dependency-free local web dashboard."""

import argparse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import socket
import threading
import time

from protocol import VERSION, decode, encode, identity, validate_sample, validate_timing


WINDOW_NS = 300_000_000_000
STALE_NS = 3_000_000_000
MAX_SEQUENCE_JUMP = 10000
SAMPLE_INTERVAL_NS = 200_000_000


@dataclass
class Record:
    seq: int
    status: str
    at_ns: int
    at_wall_ms: int
    sample: dict | None = None
    latency_ms: float | None = None


class TelemetryState:
    def __init__(self, deadline_ms=100.0):
        self.deadline_ms = deadline_ms
        self.lock = threading.Lock()
        self.session = None
        self.old_sessions = []
        self.records = {}
        self.max_seq = 0
        self.last_seen_ns = None
        self.unattributed_invalid = 0
        self.duplicates = 0
        self.reordered = 0
        self.pending_timings = {}

    def _reset_session(self, session):
        if self.session is not None:
            self.old_sessions.append(self.session)
            self.old_sessions = self.old_sessions[-100:]
        self.session = session
        self.records = {}
        self.max_seq = 0
        self.last_seen_ns = None
        self.pending_timings = {}
        self.duplicates = 0
        self.reordered = 0

    def _prune(self, now_ns):
        cutoff = now_ns - WINDOW_NS
        self.records = {seq: rec for seq, rec in self.records.items() if rec.at_ns >= cutoff}
        self.pending_timings = {
            seq: item for seq, item in self.pending_timings.items() if item[1] >= cutoff
        }

    def _apply_timing(self, timing, now_ns):
        if timing["session"] != self.session:
            return
        seq = timing["seq"]
        rec = self.records.get(seq)
        if rec is not None and rec.status == "valid":
            if rec.latency_ms is None:
                rec.latency_ms = timing["latency_ms"]
        elif rec is None or rec.status in ("missing", "invalid"):
            self.pending_timings[seq] = (timing, now_ns)

    def _add_missing(self, start_seq, end_seq, now_ns, wall_ms):
        for seq in range(start_seq, end_seq):
            age_ns = (end_seq - seq) * SAMPLE_INTERVAL_NS
            self.records[seq] = Record(seq, "missing", now_ns - age_ns,
                                       wall_ms - age_ns // 1_000_000)

    def _record_invalid(self, message, now_ns, wall_ms):
        try:
            session, seq = identity(message)
        except ValueError:
            self.unattributed_invalid += 1
            return
        if session != self.session or seq <= 0 or seq > self.max_seq + MAX_SEQUENCE_JUMP:
            self.unattributed_invalid += 1
            return
        if seq in self.records and self.records[seq].status == "valid":
            self.duplicates += 1
            return
        if seq > self.max_seq:
            self._add_missing(self.max_seq + 1, seq, now_ns, wall_ms)
            self.max_seq = seq
        self.records[seq] = Record(seq, "invalid", now_ns, wall_ms)

    def handle_datagram(self, data, now_ns=None, wall_ms=None):
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        wall_ms = int(time.time() * 1000) if wall_ms is None else wall_ms
        try:
            message = decode(data)
        except (ValueError, UnicodeDecodeError):
            with self.lock:
                self.unattributed_invalid += 1
            return None
        kind = message.get("kind")
        if kind == "timing":
            try:
                timing = validate_timing(message)
            except ValueError:
                with self.lock:
                    self.unattributed_invalid += 1
                return None
            with self.lock:
                self._prune(now_ns)
                self._apply_timing(timing, now_ns)
            return None
        if kind != "sample":
            with self.lock:
                self.unattributed_invalid += 1
            return None
        try:
            sample = validate_sample(message)
        except ValueError:
            with self.lock:
                self._record_invalid(message, now_ns, wall_ms)
            return None
        session, seq = sample["session"], sample["seq"]
        with self.lock:
            if session in self.old_sessions:
                return None
            if session != self.session:
                self._reset_session(session)
                self.max_seq = seq - 1  # The receiver cannot infer packets sent before it joined.
            self._prune(now_ns)
            if seq > self.max_seq + MAX_SEQUENCE_JUMP:
                self.unattributed_invalid += 1
                return None
            existing = self.records.get(seq)
            if existing is not None and existing.status == "valid":
                self.duplicates += 1
            elif seq <= self.max_seq and existing is None:
                return None  # Outside the rolling window.
            else:
                if seq > self.max_seq:
                    self._add_missing(self.max_seq + 1, seq, now_ns, wall_ms)
                    self.max_seq = seq
                else:
                    self.reordered += 1
                rec = Record(seq, "valid", now_ns, wall_ms, sample)
                self.records[seq] = rec
                pending = self.pending_timings.pop(seq, None)
                if pending is not None:
                    rec.latency_ms = pending[0]["latency_ms"]
                self.last_seen_ns = now_ns
            for timing in sample["timings"]:
                self._apply_timing(timing, now_ns)
            self._prune(now_ns)
        processing_ns = max(0, time.monotonic_ns() - now_ns)
        return encode({
            "version": VERSION, "kind": "echo", "session": session,
            "seq": seq, "server_processing_ns": processing_ns,
        })

    def snapshot(self, now_ns=None):
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        with self.lock:
            self._prune(now_ns)
            ordered = sorted(self.records.values(), key=lambda rec: rec.seq)
            valid = [rec for rec in ordered if rec.status == "valid"]
            latencies = [rec.latency_ms for rec in valid if rec.latency_ms is not None]
            expected = len(ordered)
            missing = sum(rec.status == "missing" for rec in ordered)
            invalid = sum(rec.status == "invalid" for rec in ordered)
            on_time = sum(value <= self.deadline_ms for value in latencies)
            jitter = (
                sum(abs(b - a) for a, b in zip(latencies, latencies[1:])) / (len(latencies) - 1)
                if len(latencies) > 1 else None
            )
            p99 = sorted(latencies)[math.ceil(0.99 * len(latencies)) - 1] if latencies else None
            latest = max(valid, key=lambda rec: rec.seq) if valid else None
            latest_timed = max((rec for rec in valid if rec.latency_ms is not None),
                               key=lambda rec: rec.seq, default=None)
            connected = self.last_seen_ns is not None and now_ns - self.last_seen_ns <= STALE_NS
            return {
                "session": self.session,
                "connected": connected,
                "last_seen_age_ms": round((now_ns - self.last_seen_ns) / 1_000_000)
                if self.last_seen_ns is not None else None,
                "deadline_ms": self.deadline_ms,
                "window_seconds": 300,
                "latest": ({**latest.sample, "latency_ms": latest.latency_ms,
                            "received_at_ms": latest.at_wall_ms} if latest else None),
                "metrics": {
                    "expected": expected, "valid": len(valid), "missing": missing,
                    "invalid": invalid, "unattributed_invalid": self.unattributed_invalid,
                    "duplicates": self.duplicates, "reordered": self.reordered,
                    "latency_samples": len(latencies),
                    "latency_ms": latest_timed.latency_ms if latest_timed else None,
                    "reliability_pct": 100 * len(valid) / expected if expected else None,
                    "packet_loss_pct": 100 * missing / expected if expected else None,
                    "jitter_ms": jitter, "p99_latency_ms": p99,
                    "deadline_success_pct": 100 * on_time / expected if expected else None,
                    "unknown_latency": len(valid) - len(latencies),
                },
                "history": [
                    {
                        "seq": rec.seq, "status": rec.status,
                        "at_ms": rec.at_wall_ms,
                        "latency_ms": rec.latency_ms,
                        "water": rec.sample["water"] if rec.sample else None,
                        "soil": rec.sample["soil"] if rec.sample else None,
                        "rain": rec.sample["rain"] if rec.sample else None,
                    }
                    for rec in ordered[-300:]
                ],
            }


def serve_udp(state, sock):
    with sock:
        while True:
            data, address = sock.recvfrom(65535)
            received_ns = time.monotonic_ns()
            echo = state.handle_datagram(data, received_ns)
            if echo is not None:
                sock.sendto(echo, address)


def make_handler(state):
    dashboard = (Path(__file__).parent / "dashboard.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/api/state":
                body = json.dumps(state.snapshot(), separators=(",", ":")).encode("utf-8")
                content_type = "application/json; charset=utf-8"
            elif self.path in ("/", "/index.html"):
                body = dashboard
                content_type = "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description="URLLC sensor UDP receiver and dashboard")
    parser.add_argument("--udp-host", default="0.0.0.0")
    parser.add_argument("--udp-port", type=int, default=9000)
    parser.add_argument("--web-host", default="127.0.0.1")
    parser.add_argument("--web-port", type=int, default=8000)
    parser.add_argument("--deadline-ms", type=float, default=100.0)
    args = parser.parse_args()
    if not math.isfinite(args.deadline_ms) or args.deadline_ms <= 0:
        parser.error("--deadline-ms must be a positive number")
    state = TelemetryState(args.deadline_ms)
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.bind((args.udp_host, args.udp_port))
    print(f"UDP receiver listening on {args.udp_host}:{args.udp_port}", flush=True)
    threading.Thread(target=serve_udp, args=(state, udp_socket), daemon=True).start()
    with ThreadingHTTPServer((args.web_host, args.web_port), make_handler(state)) as web:
        print(f"Dashboard: http://{args.web_host}:{args.web_port}", flush=True)
        web.serve_forever()


if __name__ == "__main__":
    main()
