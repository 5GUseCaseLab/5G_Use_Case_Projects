"""UNO Q Linux-side Bridge to UDP sender. Run in Arduino App Lab on the board."""

import argparse
from collections import deque
import os
import queue
import socket
import time
import uuid

from protocol import VERSION, decode, encode, integer


DEFAULT_SERVER_HOST = "192.168.31.121"  # This laptop's Wi-Fi address on 2026-11-11.
DEFAULT_UDP_PORT = 9000
SAMPLE_FIELDS = (
    "seq", "sample_millis", "water_raw", "soil_raw", "rain_raw",
    "water", "soil", "rain", "priority", "event",
)


class ClockMapper:
    """Map the MCU's wrapping millis() clock onto Linux monotonic time."""

    def __init__(self):
        self.mcu_millis = None
        self.linux_ns = None
        self.uncertainty_ms = None

    def calibrate(self, mcu_millis, before_ns, after_ns):
        self.mcu_millis = integer(mcu_millis, 0, 2**32 - 1)
        self.linux_ns = (before_ns + after_ns) // 2
        self.uncertainty_ms = (after_ns - before_ns) / 2_000_000

    def age_ms(self, sample_millis, now_ns):
        if self.mcu_millis is None:
            raise ValueError("MCU clock not calibrated")
        delta_ms = ((sample_millis - self.mcu_millis + 2**31) % 2**32) - 2**31
        age = (now_ns - (self.linux_ns + delta_ms * 1_000_000)) / 1_000_000
        return max(0.0, age)


class Sender:
    def __init__(self, host, port, bridge, sock=None):
        self.bridge = bridge
        self.samples = queue.Queue(maxsize=128)
        self.clock = ClockMapper()
        self.last_calibration_ns = 0
        self.session = uuid.uuid4().hex
        self.last_seq = 0
        self.pending = {}
        self.recent_timings = deque(maxlen=10)
        self.sock = sock or self._open_socket(host, port)
        self.sock.setblocking(False)
        self.bridge.provide("sensor_sample", self.on_sample)

    @staticmethod
    def _open_socket(host, port):
        info = socket.getaddrinfo(host, port, type=socket.SOCK_DGRAM)[0]
        sock = socket.socket(info[0], socket.SOCK_DGRAM)
        sock.connect(info[4])
        print("UDP SOCKET LOCAL:", sock.getsockname())
        print("UDP SOCKET REMOTE:", sock.getpeername())
        return sock
    def on_sample(self, *values):
        print("CALLBACK TRIGGERED")
        print("Received values:", len(values))
        print("Expected values:", len(SAMPLE_FIELDS))
        print("Data:", values)

        if len(values) != len(SAMPLE_FIELDS):
            print("ERROR: Sample fields mismatch")
            return

        try:
            self.samples.put_nowait(dict(zip(SAMPLE_FIELDS, values)))
            print("Sample added to queue")
        except queue.Full:
            print("Queue is full")

    def calibrate_clock(self):
        before_ns = time.monotonic_ns()
        mcu_millis = self.bridge.call("clock_millis", timeout=2)
        after_ns = time.monotonic_ns()
        self.clock.calibrate(mcu_millis, before_ns, after_ns)
        self.last_calibration_ns = after_ns

    def send_sample(self, sample):
        seq = integer(sample["seq"], 1, 2**32 - 1)
        sample_millis = integer(sample["sample_millis"], 0, 2**32 - 1)
        if seq < self.last_seq:
            self.session = uuid.uuid4().hex
            self.clock = ClockMapper()
            self.pending.clear()
            self.recent_timings.clear()
            self.last_seq = 0
            self.calibrate_clock()
        if seq <= self.last_seq:
            return
        self.last_seq = seq
        send_ns = time.monotonic_ns()
        age_ms = self.clock.age_ms(sample_millis, send_ns)
        print("SAMPLE AGE:", seq, round(age_ms, 3), "ms")
        message = {
            "version": VERSION, "kind": "sample", "session": self.session,
            **sample, "sample_age_ms": round(age_ms, 3),
            "clock_uncertainty_ms": round(self.clock.uncertainty_ms, 3),
            "timings": list(self.recent_timings),
        }
        if age_ms > 60000:
            print("SAMPLE DROPPED: age above 60000 ms")
            return
        self.sock.send(encode(message))
        print("UDP SENT:", seq)
        self.pending[seq] = (send_ns, age_ms)

    def receive_echoes(self):
        while True:
            try:
                data = self.sock.recv(4096)
            except BlockingIOError:
                break
            except OSError:
                break
            receive_ns = time.monotonic_ns()
            try:
                message = decode(data)
                if message.get("kind") != "echo" or message.get("session") != self.session:
                    continue
                seq = integer(message.get("seq"), 1, 2**32 - 1)
                processing_ns = integer(message.get("server_processing_ns"), 0, 5_000_000_000)
                pending = self.pending.pop(seq, None)
                if pending is None:
                    continue
                sent_ns, sample_age_ms = pending
                rtt_ms = max(0.0, (receive_ns - sent_ns - processing_ns) / 1_000_000)
                network_ms = rtt_ms / 2
                timing = {
                    "session": self.session, "seq": seq,
                    "latency_ms": round(sample_age_ms + network_ms, 3),
                    "network_ms": round(network_ms, 3),
                    "rtt_ms": round(rtt_ms, 3),
                }
                self.recent_timings.append(timing)
                self.sock.send(encode({"version": VERSION, "kind": "timing", **timing}))
            except (ValueError, KeyError, OSError):
                continue

    def step(self):
        now_ns = time.monotonic_ns()
        if now_ns - self.last_calibration_ns > 10_000_000_000:
            try:
                self.calibrate_clock()
            except (ConnectionError, TimeoutError, RuntimeError, ValueError) as exc:
                print("CLOCK CALIBRATION FAILED:", repr(exc))
                time.sleep(0.2)
                return
            else:
                print("CLOCK CALIBRATION SUCCESS")
        self.receive_echoes()
        try:
            sample = self.samples.get(timeout=0.02)
        except queue.Empty:
            sample = None
        if sample is not None:
            try:
                self.send_sample(sample)
            except (ValueError, OSError, ConnectionError, TimeoutError, RuntimeError) as exc:
                print("Sample send failed:", exc)
        cutoff_ns = time.monotonic_ns() - 5_000_000_000
        self.pending = {seq: item for seq, item in self.pending.items() if item[0] >= cutoff_ns}


def main():
    parser = argparse.ArgumentParser(description="UNO Q sensor UDP sender")
    parser.add_argument("--host", default=os.getenv("URLLC_SERVER_HOST", DEFAULT_SERVER_HOST))
    parser.add_argument("--port", type=int, default=int(os.getenv("URLLC_UDP_PORT", DEFAULT_UDP_PORT)))
    args = parser.parse_args()
    try:
        from arduino.app_utils import Bridge
        bridge = Bridge
    except ModuleNotFoundError:
        from router_bridge import RouterBridge
        bridge = RouterBridge()

    sender = Sender(args.host, args.port, bridge)
    print(f"Sending sensor packets to {args.host}:{args.port} (session {sender.session})")
    try:
        while True:
            sender.step()
    except KeyboardInterrupt:
        pass
    finally:
        sender.sock.close()
        if hasattr(bridge, "close"):
            bridge.close()


if __name__ == "__main__":
    main()
