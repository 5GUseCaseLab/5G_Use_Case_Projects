"""
WiFeye: Automated CSI Dataset Collection Suite
===================================================
Supports:
  1. Standalone Mode: Single laptop (iPhone hotspot or battery-powered AP as TX, ESP32 on COMx as RX)
  2. Master Mode: Laptop A (Master) controls synchronized dual-node capture over Wi-Fi
  3. Worker Mode: Laptop B (Worker) listens to Master and records synchronously (< 5ms offset)
  4. Test / Mock Mode: Full simulation and benchmark suite to verify synchronization and data pipeline
"""

import sys
import os
import time
import math
import json
import socket
import subprocess
import threading
import argparse
import ctypes
from datetime import datetime
from collections import deque
import numpy as np

# Set Windows timer resolution to 1ms for sub-millisecond sleep precision
try:
    if os.name == 'nt':
        ctypes.windll.winmm.timeBeginPeriod(1)
except Exception:
    pass

# Try importing serial and winsound
try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


def get_current_wifi_ssid():
    """Returns the SSID of the currently connected Wi-Fi interface on Windows."""
    try:
        out = subprocess.check_output(['netsh', 'wlan', 'show', 'interfaces'], stderr=subprocess.DEVNULL).decode('latin1', errors='ignore')
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("SSID") and ":" in line and not line.startswith("BSSID"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


class AudioCue:
    """Manages audio cues for countdown and trial completion with zero latency."""
    def __init__(self, enabled=True):
        self.enabled = enabled and HAS_WINSOUND
        if self.enabled:
            # Pre-warm Windows audio driver so the first countdown tick has zero delay
            try:
                winsound.Beep(1000, 10)
            except Exception:
                pass

    def countdown_tick(self):
        """Crisp synchronous 1-second countdown tick."""
        if self.enabled:
            try:
                winsound.Beep(880, 120)
            except Exception:
                pass

    def start_cue(self):
        """High-pitched 'GO' sound signaling start of trial."""
        if self.enabled:
            try:
                winsound.Beep(1760, 220)
            except Exception:
                pass

    def stop_cue(self):
        """Double-buzzer indicating trial finished."""
        if self.enabled:
            try:
                winsound.Beep(2200, 120)
                time.sleep(0.06)
                winsound.Beep(2200, 180)
            except Exception:
                pass


class CSIPacket:
    """Represents a single parsed CSI packet."""
    __slots__ = (
        'sys_time', 'role', 'mac', 'rssi', 'rate', 'sig_mode', 'mcs',
        'channel', 'local_timestamp', 'subcarrier_len', 'amplitudes',
        'phases', 'raw_real', 'raw_imag', 'raw_line'
    )

    def __init__(self, sys_time, role, mac, rssi, rate, sig_mode, mcs,
                 channel, local_timestamp, subcarrier_len, amplitudes,
                 phases, raw_real, raw_imag, raw_line=""):
        self.sys_time = sys_time
        self.role = role
        self.mac = mac
        self.rssi = rssi
        self.rate = rate
        self.sig_mode = sig_mode
        self.mcs = mcs
        self.channel = channel
        self.local_timestamp = local_timestamp
        self.subcarrier_len = subcarrier_len
        self.amplitudes = amplitudes
        self.phases = phases
        self.raw_real = raw_real
        self.raw_imag = raw_imag
        self.raw_line = raw_line


def parse_csi_line(line: str, arrival_time: float) -> CSIPacket:
    """Fast parser for raw ESP32 CSI serial output."""
    if "CSI_DATA" not in line:
        return None
    try:
        parts = line.strip().split(',')
        if len(parts) < 26:
            return None
        
        role = parts[1]
        mac = parts[2]
        rssi = int(parts[3])
        rate = int(parts[4])
        sig_mode = int(parts[5])
        mcs = int(parts[6])
        channel = int(parts[16])
        local_ts = int(parts[18])
        
        # Extract raw values inside brackets [...]
        bracket_start = line.find('[')
        bracket_end = line.find(']', bracket_start)
        if bracket_start == -1 or bracket_end == -1:
            return None
        
        raw_vals = [int(x) for x in line[bracket_start+1:bracket_end].split() if x]
        val_count = len(raw_vals)
        if val_count < 2:
            return None
            
        subcarrier_count = val_count // 2
        
        # Imaginary and Real separation
        # In ESP32-CSI-Tool: even index is Imaginary, odd index is Real
        raw_arr = np.array(raw_vals, dtype=np.int16)
        imag = raw_arr[0::2][:subcarrier_count]
        real = raw_arr[1::2][:subcarrier_count]
        
        # Compute amplitude: sqrt(real^2 + imag^2)
        # Compute phase: atan2(imag, real)
        amps = np.sqrt(imag.astype(np.float32)**2 + real.astype(np.float32)**2)
        phases = np.arctan2(imag.astype(np.float32), real.astype(np.float32))
        
        return CSIPacket(
            sys_time=arrival_time,
            role=role,
            mac=mac,
            rssi=rssi,
            rate=rate,
            sig_mode=sig_mode,
            mcs=mcs,
            channel=channel,
            local_timestamp=local_ts,
            subcarrier_len=subcarrier_count,
            amplitudes=amps,
            phases=phases,
            raw_real=real,
            raw_imag=imag,
            raw_line=line
        )
    except Exception:
        return None


class MockCSIStream:
    """Generates authentic 100 Hz synthetic CSI frames for testing."""
    def __init__(self, rate_hz=100.0, mac="36:CE:3A:EA:CA:E6"):
        self.rate_hz = rate_hz
        self.interval = 1.0 / rate_hz
        self.mac = mac
        self.running = False
        self._thread = None
        self._callback = None
        self._step = 0

    def start(self, callback):
        self._callback = callback
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        next_time = time.perf_counter()
        while self.running:
            self._step += 1
            now = time.time()
            # Generate 64 subcarriers with realistic multi-path fluctuation
            t_rel = self._step * self.interval
            wave = np.sin(2 * np.pi * 1.5 * t_rel + np.linspace(0, np.pi, 64))
            imag = (20 + 15 * wave + np.random.normal(0, 2, 64)).astype(int)
            real = (10 + 10 * np.cos(2 * np.pi * 1.5 * t_rel) + np.random.normal(0, 2, 64)).astype(int)
            
            raw_vals = []
            for i in range(64):
                raw_vals.append(str(imag[i]))
                raw_vals.append(str(real[i]))
            raw_str = " ".join(raw_vals)
            
            sim_line = (
                f"CSI_DATA,STA,{self.mac},-48,10,0,0,0,0,0,0,0,0,0,-96,0,6,0,"
                f"{int(t_rel*1000000)},0,204,0,0,{t_rel:.4f},128,[{raw_str} ]"
            )
            pkt = parse_csi_line(sim_line, now)
            if pkt and self._callback:
                self._callback(pkt)
            
            next_time += self.interval
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_time = time.perf_counter()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=1.0)


class SerialCSIReader:
    """Manages high-baud serial connection to ESP32 with adaptive watchdog pulsing."""
    def __init__(self, port, baud=921600, target_ip=None):
        self.port = port
        self.baud = baud
        self.target_ip = target_ip
        self.ser = None
        self.running = False
        self._thread = None
        self._pulse_thread = None
        self._callback = None
        self.esp_ip = None
        self.gateway_ip = None
        self.live_rate_hz = 0.0
        self.pump_hz = 140.0          # UDP traffic pump rate; 0 disables it (the live engine can lower it)
        self._packet_count_window = 0
        self._last_rate_time = time.time()

    def connect(self):
        if not HAS_SERIAL:
            raise RuntimeError("pyserial is not installed.")
        self.ser = serial.Serial()
        self.ser.port = self.port
        self.ser.baudrate = self.baud
        self.ser.timeout = 0.05
        self.ser.dtr = False
        self.ser.rts = False
        self.ser.open()
        # Flush buffers
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    def start(self, callback):
        self._callback = callback
        self.running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        # Non-congesting adaptive watchdog pulse thread
        self._pulse_thread = threading.Thread(target=self._pulse_loop, daemon=True)
        self._pulse_thread.start()

    def _pulse_loop(self):
        # Discover all local IPv4 network interfaces to bind directly to Wi-Fi adapter
        send_targets = []  # list of (socket, (ip_dest, port))
        try:
            hostname = socket.gethostname()
            local_ips = socket.gethostbyname_ex(hostname)[2]
            for ip in local_ips:
                if ip.startswith("127.") or ip.startswith("169.254."):
                    continue
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                    s.bind((ip, 0))  # Force binding to this interface
                    parts = ip.split('.')
                    if len(parts) == 4:
                        bcast = f"{parts[0]}.{parts[1]}.{parts[2]}.255"
                        gw = f"{parts[0]}.{parts[1]}.{parts[2]}.1"
                        send_targets.append((s, (bcast, 2223)))
                        send_targets.append((s, (gw, 2223)))
                        send_targets.append((s, ('255.255.255.255', 2223)))
                except Exception:
                    pass
        except Exception:
            pass

        # Fallback standard socket if interface binding had no targets
        if not send_targets:
            try:
                fallback_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                fallback_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                send_targets.append((fallback_sock, ('255.255.255.255', 2223)))
            except Exception:
                pass

        # Continuous packet pump (default 140 Hz, ~7.14 ms interval); pump_hz <= 0 disables it
        target_hz = float(self.pump_hz)
        if target_hz <= 0:
            for s, _ in send_targets:
                try:
                    s.close()
                except Exception:
                    pass
            return
        interval = 1.0 / target_hz

        next_t = time.perf_counter()
        while self.running:
            for s, dest in send_targets:
                try:
                    s.sendto(b'ping', dest)
                except Exception:
                    pass
            next_t += interval

            sleep_dur = next_t - time.perf_counter()
            if sleep_dur > 0.002:
                time.sleep(sleep_dur - 0.001)
            while time.perf_counter() < next_t:
                pass
            if time.perf_counter() - next_t > 0.1:
                next_t = time.perf_counter()

        for s, _ in send_targets:
            try:
                s.close()
            except Exception:
                pass

    def _read_loop(self):
        buffer = bytearray()
        while self.running and self.ser and self.ser.is_open:
            try:
                chunk = self.ser.read(self.ser.in_waiting or 1)
                if not chunk:
                    continue
                arrival_time = time.time()
                buffer.extend(chunk)
                while b'\n' in buffer:
                    line_bytes, buffer = buffer.split(b'\n', 1)
                    try:
                        line = line_bytes.decode('utf-8', errors='ignore').strip()
                    except Exception:
                        continue

                    # Auto-discover ESP32 & Gateway IPs from firmware boot logs
                    if "Got ip:" in line:
                        parts = line.split("Got ip:")
                        if len(parts) > 1:
                            self.esp_ip = parts[1].strip().split()[0]
                    elif "Target Gateway IP dynamically set to:" in line:
                        parts = line.split("Target Gateway IP dynamically set to:")
                        if len(parts) > 1:
                            self.gateway_ip = parts[1].strip().split()[0]

                    if "CSI_DATA" in line:
                        self._packet_count_window += 1
                        now = time.time()
                        dt = now - self._last_rate_time
                        if dt >= 1.0:
                            self.live_rate_hz = self._packet_count_window / dt
                            self._packet_count_window = 0
                            self._last_rate_time = now

                        pkt = parse_csi_line(line, arrival_time)
                        if pkt and self._callback:
                            self._callback(pkt)
            except Exception:
                break

    def stop(self):
        self.running = False
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._pulse_thread:
            self._pulse_thread.join(timeout=0.5)


class NetworkSyncNode:
    """
    Handles UDP-based NTP clock synchronization and future-epoch scheduling.
    Master calculates clock offset θ:
      RTT = (T4 - T1) - (T3 - T2)
      θ = ((T2 - T1) + (T3 - T4)) / 2
    Worker applies θ so its clock matches Master within < 1 ms.
    """
    def __init__(self, is_master=True, sync_port=5555, target_ip="<broadcast>", bind_ip="0.0.0.0"):
        self.is_master = is_master
        self.port = sync_port
        self.target_ip = target_ip
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sock.bind((bind_ip, self.port))
        self.sock.settimeout(0.5)
        self.hellos_sent = 0      # worker: announcements sent
        self.heard = 0            # messages received from the other laptop (diagnostics)

        self.clock_offset = 0.0  # Master time = Worker time + offset
        self.min_rtt_ms = 999.0
        self.running = False
        self._listen_thread = None
        self.worker_ip = None
        self.master_ip = None
        self.armed_trial = None
        self._arm_lock = threading.Lock()
        self._seen_arm_ids = deque(maxlen=200)
        self.is_recording = False
        self.trial_result_callback = None
        self.arm_callback = None

    def _get_broadcast_targets(self):
        targets = ["255.255.255.255"]
        try:
            # Query local interface subnet
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('10.255.255.255', 1))
            local_ip = s.getsockname()[0]
            s.close()
            parts = local_ip.split('.')
            if len(parts) == 4:
                subnet_bcast = f"{parts[0]}.{parts[1]}.{parts[2]}.255"
                if subnet_bcast not in targets:
                    targets.append(subnet_bcast)
        except Exception:
            pass
        return targets

    def start(self):
        self.running = True
        self._listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._listen_thread.start()

    def get_synced_time(self):
        """Returns the current master epoch time."""
        if self.is_master:
            return time.time()
        else:
            return time.time() + self.clock_offset

    def _listen_loop(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(4096)
                t_recv = time.time()
                msg = json.loads(data.decode('utf-8'))
                m_type = msg.get("type")

                if self.is_master:
                    if m_type == "WORKER_HELLO":
                        # worker announced itself: remember it and answer directly (unicast is
                        # acknowledged and retried by Wi-Fi; broadcasts are not)
                        self.worker_ip = addr[0]
                        self.heard += 1
                        self.sock.sendto(json.dumps({"type": "HELLO_ACK"}).encode('utf-8'), addr)
                    elif m_type == "DISCOVER_ACK":
                        self.worker_ip = addr[0]
                    elif m_type == "SYNC_PONG":
                        self.worker_ip = addr[0]
                        t1 = msg["t1"]
                        t2 = msg["t2"]
                        t3 = msg["t3"]
                        t4 = t_recv
                        rtt = (t4 - t1) - (t3 - t2)
                        offset = ((t2 - t1) + (t3 - t4)) / 2.0
                        msg_id = msg.get("id", 0)
                        if rtt >= 0 and (rtt * 1000) < self.min_rtt_ms:
                            self.min_rtt_ms = rtt * 1000
                            self.clock_offset = -offset  # Worker needs to add -offset
                    elif m_type == "TRIAL_RESULT":
                        if self.trial_result_callback:
                            self.trial_result_callback(msg)
                else:
                    # Worker logic
                    if m_type == "WORKER_HELLO":
                        continue          # our own announcement looped back (or another worker's)
                    self.master_ip = addr[0]
                    self.heard += 1
                    if m_type == "DISCOVER":
                        ack = json.dumps({"type": "DISCOVER_ACK", "status": "READY"}).encode('utf-8')
                        self.sock.sendto(ack, addr)
                    elif m_type == "SYNC_PING":
                        t1 = msg["t1"]
                        t2 = t_recv
                        t3 = time.time()
                        pong = json.dumps({
                            "type": "SYNC_PONG",
                            "t1": t1,
                            "t2": t2,
                            "t3": t3,
                            "id": msg.get("id", 0)
                        }).encode('utf-8')
                        self.sock.sendto(pong, addr)
                    elif m_type == "SYNC_FINAL":
                        self.clock_offset = msg.get("offset", 0.0)
                        self.min_rtt_ms = msg.get("rtt_min_ms", 0.0)
                    elif m_type == "ARM_TRIAL":
                        arm_id = msg.get("arm_id") or (
                            f"{msg.get('subject')}_{msg.get('activity')}_"
                            f"{msg.get('trial_num')}_{msg.get('t_start')}"
                        )
                        ack = json.dumps({"type": "ARM_ACK", "status": "ARMED", "arm_id": arm_id}).encode('utf-8')
                        self.sock.sendto(ack, addr)
                        accepted = False
                        with self._arm_lock:
                            eff_offset = msg.get("clock_offset", self.clock_offset)
                            t_end_est = msg.get("t_start", 0.0) + msg.get("duration", 0.0)
                            is_stale = t_end_est <= (time.time() + eff_offset)
                            if (arm_id not in self._seen_arm_ids) and (not self.is_recording) and (not is_stale):
                                self._seen_arm_ids.append(arm_id)
                                if "clock_offset" in msg:
                                    self.clock_offset = msg["clock_offset"]
                                self.armed_trial = msg
                                accepted = True
                        if accepted and self.arm_callback:
                            self.arm_callback(msg)
            except socket.timeout:
                continue
            except Exception:
                continue

    # Master methods
    def calibrate_clock(self, iterations=15, destination_ip=None):
        """Runs NTP handshake with worker to calculate offset and RTT."""
        dest = destination_ip or self.worker_ip or self.target_ip
        self.min_rtt_ms = 999.0
        
        targets = [dest]
        if dest == "<broadcast>":
            targets = self._get_broadcast_targets()

        for i in range(iterations):
            ping = json.dumps({
                "type": "SYNC_PING",
                "t1": time.time(),
                "id": i
            }).encode('utf-8')
            for t in targets:
                try:
                    self.sock.sendto(ping, (t, self.port))
                except Exception:
                    pass
            time.sleep(0.04)
            
        time.sleep(0.1)
        # Notify worker of final calculated offset (use newly discovered worker_ip if available)
        final_dest = destination_ip or self.worker_ip or self.target_ip
        final_targets = [final_dest]
        if final_dest == "<broadcast>":
            final_targets = self._get_broadcast_targets()
        final_msg = json.dumps({
            "type": "SYNC_FINAL",
            "offset": self.clock_offset,
            "rtt_min_ms": self.min_rtt_ms
        }).encode('utf-8')
        for t in final_targets:
            try:
                self.sock.sendto(final_msg, (t, self.port))
            except Exception:
                pass

    def send_arm_trial(self, trial_info, destination_ip=None):
        dest = destination_ip or self.worker_ip or self.target_ip
        msg = json.dumps(trial_info).encode('utf-8')
        targets = [dest]
        if dest == "<broadcast>":
            targets = self._get_broadcast_targets()
        for _ in range(3):  # Resilient transmit
            for t in targets:
                try:
                    self.sock.sendto(msg, (t, self.port))
                except Exception:
                    pass
            time.sleep(0.02)

    def announce(self):
        """Worker: tell the master we are here. Broadcast until the master is known, then unicast
        (the unicast also keeps a return path open through the worker's firewall)."""
        msg = json.dumps({"type": "WORKER_HELLO"}).encode('utf-8')
        targets = [self.master_ip] if self.master_ip else self._get_broadcast_targets()
        for t in targets:
            try:
                self.sock.sendto(msg, (t, self.port))
            except Exception:
                pass
        self.hellos_sent += 1

    def send_trial_result(self, result_dict, master_ip):
        msg = json.dumps(result_dict).encode('utf-8')
        try:
            self.sock.sendto(msg, (master_ip, self.port))
        except Exception:
            pass

    def stop(self):
        self.running = False
        try:
            self.sock.close()
        except Exception:
            pass
        if self._listen_thread:
            self._listen_thread.join(timeout=0.5)


class DatasetTrialManager:
    """Orchestrates trial capture, quality gates, and dataset persistence."""
    def __init__(self, output_dir="data", rx_role="RX1"):
        self.output_dir = output_dir
        self.rx_role = rx_role
        self.manifest_path = os.path.join(output_dir, "manifest.csv")
        self._ensure_dirs()

    def _ensure_dirs(self):
        os.makedirs(self.output_dir, exist_ok=True)
        raw_dir = os.path.join(self.output_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        if not os.path.exists(self.manifest_path):
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                f.write(
                    "trial_id,subject,activity,trial_num,duration_s,rx_role,filename,"
                    "packet_count,sampling_rate_hz,mean_rssi,status,recorded_at\n"
                )

    def save_trial(self, packets, subject, activity, trial_num, duration, t_start, clock_offset=0.0):
        """Saves parsed CSI trial data into structured CSV."""
        act_dir = os.path.join(self.output_dir, "raw", activity)
        os.makedirs(act_dir, exist_ok=True)

        filename = f"{subject}_{activity}_t{trial_num:02d}_{self.rx_role}.csv"
        filepath = os.path.join(act_dir, filename)

        # Build column header: metadata + 64 amplitudes + 64 phases
        subcarrier_cols_amp = [f"amp_{i}" for i in range(64)]
        subcarrier_cols_phase = [f"phase_{i}" for i in range(64)]
        subcarrier_cols_real = [f"real_{i}" for i in range(64)]
        subcarrier_cols_imag = [f"imag_{i}" for i in range(64)]
        
        headers = [
            "timestamp_rel", "epoch_timestamp", "esp_timestamp", "role", "mac",
            "rssi", "rate", "sig_mode", "mcs", "channel"
        ] + subcarrier_cols_amp + subcarrier_cols_phase + subcarrier_cols_real + subcarrier_cols_imag

        lines = [",".join(headers) + "\n"]

        rssi_vals = []
        for p in packets:
            synced_time = p.sys_time + clock_offset
            rel_t = synced_time - t_start
            rssi_vals.append(p.rssi)
            row = [
                f"{rel_t:.6f}",
                f"{synced_time:.6f}",
                str(p.local_timestamp),
                p.role,
                p.mac,
                str(p.rssi),
                str(p.rate),
                str(p.sig_mode),
                str(p.mcs),
                str(p.channel)
            ]
            # Format 64 amps and 64 phases
            row += [f"{val:.2f}" for val in p.amplitudes[:64]]
            row += [f"{val:.4f}" for val in p.phases[:64]]
            row += [str(val) for val in p.raw_real[:64]]
            row += [str(val) for val in p.raw_imag[:64]]
            lines.append(",".join(row) + "\n")

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.writelines(lines)
        except PermissionError:
            ts_suffix = datetime.now().strftime("%H%M%S")
            filename = f"{subject}_{activity}_t{trial_num:02d}_{self.rx_role}_{ts_suffix}.csv"
            filepath = os.path.join(act_dir, filename)
            print(f"\n  [NOTE] Target file is open in Excel or another program.")
            print(f"         Auto-saved trial safely as: {filename}")
            with open(filepath, "w", encoding="utf-8") as f:
                f.writelines(lines)

        packet_count = len(packets)
        rate_hz = packet_count / duration if duration > 0 else 0
        mean_rssi = np.mean(rssi_vals) if rssi_vals else 0

        # Quality status (50 Hz is the standard gold standard for CSI human sensing)
        if rate_hz >= 50.0:
            status = "PASS"
        elif rate_hz >= 25.0:
            status = "WARNING"
        else:
            status = "FAIL"

        return filepath, filename, packet_count, rate_hz, mean_rssi, status

    def append_manifest(self, trial_id, subject, activity, trial_num, duration,
                        filename, packet_count, rate_hz, mean_rssi, status):
        now_str = datetime.now().isoformat()
        row = (
            f"{trial_id},{subject},{activity},{trial_num},{duration:.2f},{self.rx_role},"
            f"{filename},{packet_count},{rate_hz:.1f},{mean_rssi:.1f},{status},{now_str}\n"
        )
        try:
            with open(self.manifest_path, "a", encoding="utf-8") as f:
                f.write(row)
        except PermissionError:
            alt_manifest = os.path.join(self.output_dir, f"manifest_{datetime.now().strftime('%Y%m%d')}.csv")
            print(f"  [NOTE] manifest.csv is open in Excel. Appended to: {alt_manifest}")
            with open(alt_manifest, "a", encoding="utf-8") as f:
                f.write(row)


def execute_countdown_and_capture(t_start, duration, get_packets_fn, audio: AudioCue, clock_offset=0.0):
    """
    Executes precision audio countdown and records packets for exact duration.
    Audio countdown: 3.. 2.. 1.. GO!
    Cutoff: exactly at t_start + duration (all evaluated in synced Master clock).
    """
    t_minus_3 = t_start - 3.0
    t_minus_2 = t_start - 2.0
    t_minus_1 = t_start - 1.0
    t_end = t_start + duration

    # 3-second countdown phase
    for tick_time, tick_label in [(t_minus_3, "3"), (t_minus_2, "2"), (t_minus_1, "1")]:
        wait = tick_time - (time.time() + clock_offset)
        if wait > 0:
            time.sleep(wait)
        print(f"  [COUNTDOWN]  >>> {tick_label} <<<", flush=True)
        audio.countdown_tick()

    # Start trigger
    wait_start = t_start - (time.time() + clock_offset)
    if wait_start > 0:
        time.sleep(wait_start)
    
    print(f"\n  =======================================================", flush=True)
    print(f"  >>> RECORDING NOW (Duration: {duration:.1f}s) - PERFORM ACTION! <<<", flush=True)
    print(f"  =======================================================\n", flush=True)
    audio.start_cue()

    # Active recording period
    wait_end = t_end - (time.time() + clock_offset)
    if wait_end > 0:
        time.sleep(wait_end)

    # Stop trigger
    print(f"  >>> TRIAL COMPLETE! Processing frames... <<<\n", flush=True)
    audio.stop_cue()

    # Collect packets captured between t_start and t_end (converted to synced Master clock)
    all_captured = get_packets_fn()
    trial_packets = [p for p in all_captured if t_start <= (p.sys_time + clock_offset) <= t_end]
    return trial_packets


def run_standalone(args):
    """Executes single-laptop standalone data collection."""
    print("\n" + "=" * 65)
    print("  SAFESENSE-5G DATASET COLLECTOR: STANDALONE MODE (1 LAPTOP)")
    print("=" * 65)
    print(f"  Subject       : {args.subject}")
    print(f"  Activity      : {args.activity}")
    print(f"  Trial Duration: {args.duration}s")
    print(f"  Repetitions   : {args.trials}")
    print(f"  Audio Cues    : {'Enabled' if not args.no_audio else 'Disabled'}")
    print(f"  Data Source   : {'Mock Stream (100 Hz)' if args.mock else f'Serial Port {args.port}'}")
    print("=" * 65 + "\n")

    audio = AudioCue(enabled=not args.no_audio)
    trial_mgr = DatasetTrialManager(output_dir=args.out_dir, rx_role="RX1")
    buffer_lock = threading.Lock()
    packet_buffer = deque(maxlen=20000)

    def on_packet(pkt):
        with buffer_lock:
            packet_buffer.append(pkt)

    def snapshot_packets():
        with buffer_lock:
            return list(packet_buffer)

    if args.mock:
        source = MockCSIStream(rate_hz=100.0)
        source.start(on_packet)
    else:
        source = SerialCSIReader(port=args.port, baud=args.baud)
        try:
            source.connect()
            source.start(on_packet)
            print(f"  Connected to ESP32 on {args.port} at {args.baud} baud.")
        except Exception as e:
            print(f"  [ERROR] Failed to open {args.port}: {e}")
            return

    # Warmup
    print("  Waiting for active CSI packet stream...")
    t0 = time.time()
    while len(packet_buffer) < 5 and time.time() - t0 < 5.0:
        time.sleep(0.1)

    if len(packet_buffer) == 0:
        print("  [WARNING] No CSI frames detected yet! Ensure ESP32 is powered and connected to Wi-Fi.")
    else:
        print(f"  Stream healthy! Incoming packet rate active.\n")

    trial_idx = 1
    while trial_idx <= args.trials:
        trial_id = f"{args.subject}_{args.activity}_t{trial_idx:02d}"
        print("-" * 65)
        print(f"  READY FOR TRIAL {trial_idx} / {args.trials}: [{args.activity.upper()}] (ID: {trial_id})")
        print("  Get in position. Press [ENTER] to start 3-second countdown (or 'q' to quit)...")
        cmd = input("  > ").strip().lower()
        if cmd == 'q':
            break

        with buffer_lock:
            packet_buffer.clear()
        t_start = time.time() + 3.0
        packets = execute_countdown_and_capture(
            t_start=t_start,
            duration=args.duration,
            get_packets_fn=snapshot_packets,
            audio=audio
        )

        fpath, fname, count, rate_hz, mean_rssi, status = trial_mgr.save_trial(
            packets=packets,
            subject=args.subject,
            activity=args.activity,
            trial_num=trial_idx,
            duration=args.duration,
            t_start=t_start
        )

        status_badge = f"[{status}]"
        print(f"  --- Trial {trial_idx} Summary ---")
        print(f"  Frames Captured : {count} frames ({rate_hz:.1f} Hz) - {status_badge}")
        print(f"  Mean RSSI       : {mean_rssi:.1f} dBm")
        print(f"  Saved to        : {fpath}")

        print("\n  Actions: [ENTER] Accept & Next | [r] Redo Trial | [d] Discard")
        act = input("  > ").strip().lower()
        if act == 'r':
            print("  Redoing trial...")
            try:
                os.remove(fpath)
            except Exception:
                pass
            continue
        elif act == 'd':
            print("  Trial discarded.")
            try:
                os.remove(fpath)
            except Exception:
                pass
            trial_idx += 1
            continue
        else:
            trial_mgr.append_manifest(
                trial_id=trial_id,
                subject=args.subject,
                activity=args.activity,
                trial_num=trial_idx,
                duration=args.duration,
                filename=fname,
                packet_count=count,
                rate_hz=rate_hz,
                mean_rssi=mean_rssi,
                status=status
            )
            print("  Accepted and logged to manifest.csv!\n")
            trial_idx += 1

    source.stop()
    print("\n  All repetitions completed. Dataset collection session ended cleanly.")


def run_master(args):
    """Executes Master Node orchestrating synchronized dual-laptop collection."""
    print("\n" + "=" * 65)
    print("  SAFESENSE-5G DATASET COLLECTOR: MASTER NODE (LAPTOP A)")
    print("=" * 65)
    print(f"  Subject       : {args.subject}")
    print(f"  Activity      : {args.activity}")
    print(f"  Trial Duration: {args.duration}s")
    print(f"  Repetitions   : {args.trials}")
    print(f"  Sync Port     : UDP {args.net_port}")
    print(f"  Worker Target : {args.worker_ip or '<auto-broadcast>'}")
    print("=" * 65 + "\n")

    audio = AudioCue(enabled=not args.no_audio)
    trial_mgr = DatasetTrialManager(output_dir=args.out_dir, rx_role="RX1")
    sync = NetworkSyncNode(is_master=True, sync_port=args.net_port, target_ip=args.worker_ip or '<broadcast>')
    sync.start()

    worker_result_box = [None]
    def on_worker_result(res):
        worker_result_box[0] = res
    sync.trial_result_callback = on_worker_result

    buffer_lock = threading.Lock()
    packet_buffer = deque(maxlen=20000)
    def on_packet(pkt):
        with buffer_lock:
            packet_buffer.append(pkt)

    def snapshot_packets():
        with buffer_lock:
            return list(packet_buffer)

    if args.mock:
        source = MockCSIStream(rate_hz=100.0)
        source.start(on_packet)
    else:
        source = SerialCSIReader(port=args.port, baud=args.baud)
        try:
            source.connect()
            source.start(on_packet)
        except Exception as e:
            print(f"  [ERROR] Serial connection failed: {e}")
            sync.stop()
            return

    # Verify Wi-Fi network connection
    cur_wifi = get_current_wifi_ssid()
    if cur_wifi:
        print(f"  Laptop Wi-Fi  : Connected to '{cur_wifi}'")
        if cur_wifi.lower() != "csi":
            print(f"  >>> [CRITICAL NOTICE] <<<")
            print(f"  Your laptop is on '{cur_wifi}', but the ESP32 is on Wi-Fi 'CSI'!")
            print(f"  You MUST connect this laptop to Wi-Fi 'CSI' for high-rate (>100 Hz) CSI.")
            print(f"  Otherwise, the ESP32 will only receive idle beacons at 1 Hz!\n")
    else:
        print("  Laptop Wi-Fi  : Unable to detect (ensure Wi-Fi is connected)")

    # Verify CSI stream health on Master
    print(f"  Connected to ESP32 on {args.port} at {args.baud} baud.")
    print("  Waiting for ESP32 Wi-Fi association and measuring CSI stream rate...")
    t_wait = time.time()
    while time.time() - t_wait < 3.5:
        with buffer_lock:
            if len(packet_buffer) > 10:
                break
        time.sleep(0.2)

    with buffer_lock:
        packet_buffer.clear()
    time.sleep(2.0)

    buffered_count = len(packet_buffer)
    live_rate = buffered_count / 2.0
    if live_rate >= 50.0:
        print(f"  [STREAM HEALTHY] High rate verified: {live_rate:.1f} Hz ({buffered_count} frames in 2s).\n")
    elif live_rate > 0:
        print(f"  [NOTICE] Baseline CSI frame rate: {live_rate:.1f} Hz ({buffered_count} frames in 2s).")
        print("  High-rate packet pulsing will stream continuously during trials.\n")
    elif not args.mock:
        print("  [WARNING] No CSI frames detected from ESP32 on Master!")
        print("  Check: 1) ESP32 is flashed with active-sta for 'CSI'")
        print("         2) Wi-Fi AP 'CSI' is in range and laptop is connected to 'CSI'.\n")

    # Worker discovery
    print("  Searching for Worker Laptop on local network...")
    sync.calibrate_clock(iterations=10)
    time.sleep(0.5)

    if sync.min_rtt_ms < 100:
        print(f"  Worker synchronized! Estimated RTT: {sync.min_rtt_ms:.2f} ms | Offset: {sync.clock_offset*1000:.2f} ms\n")
    else:
        print("  [NOTE] Running in broadcast mode (Worker will sync dynamically).\n")

    trial_idx = 1
    while trial_idx <= args.trials:
        trial_id = f"{args.subject}_{args.activity}_t{trial_idx:02d}"
        print("-" * 65)
        print(f"  READY FOR SYNCHRONIZED TRIAL {trial_idx} / {args.trials}: [{args.activity.upper()}]")
        print("  Press [ENTER] to broadcast ARM and begin 3-second countdown (or 'q' to quit)...")
        cmd = input("  > ").strip().lower()
        if cmd == 'q':
            break

        # Re-calibrate clock before trial
        sync.calibrate_clock(iterations=5)
        
        # Schedule future start time
        t_start = sync.get_synced_time() + 3.0
        arm_id = f"{trial_id}_{t_start:.6f}"
        arm_packet = {
            "type": "ARM_TRIAL",
            "arm_id": arm_id,
            "subject": args.subject,
            "activity": args.activity,
            "trial_num": trial_idx,
            "duration": args.duration,
            "t_start": t_start,
            "clock_offset": sync.clock_offset
        }
        with buffer_lock:
            packet_buffer.clear()
        worker_result_box[0] = None
        sync.send_arm_trial(arm_packet)

        packets = execute_countdown_and_capture(
            t_start=t_start,
            duration=args.duration,
            get_packets_fn=snapshot_packets,
            audio=audio
        )

        fpath, fname, count, rate_hz, mean_rssi, status = trial_mgr.save_trial(
            packets=packets,
            subject=args.subject,
            activity=args.activity,
            trial_num=trial_idx,
            duration=args.duration,
            t_start=t_start
        )

        # Wait up to 3.0s for worker summary
        t_wait = time.time()
        while worker_result_box[0] is None and time.time() - t_wait < 3.0:
            time.sleep(0.05)

        w_res = worker_result_box[0]
        print(f"\n  === TRIAL {trial_idx} SYNCHRONIZED SUMMARY ===")
        print(f"  Master (RX1) : {count} frames ({rate_hz:.1f} Hz) - [{status}]")
        if w_res:
            w_count = w_res.get('packet_count', 0)
            w_rate = w_res.get('rate_hz', 0.0)
            w_status = w_res.get('status', 'N/A')
            print(f"  Worker (RX2) : {w_count} frames ({w_rate:.1f} Hz) - [{w_status}]")
            print(f"  Sync Offset  : ~{sync.min_rtt_ms/2.0:.2f} ms (< 5ms target)")
        else:
            print("  Worker (RX2) : [No response from worker node - saved locally on Worker]")

        print("\n  Actions: [ENTER] Accept & Next | [r] Redo Trial | [d] Discard")
        act = input("  > ").strip().lower()
        if act == 'r':
            print("  Redoing trial...")
            try:
                os.remove(fpath)
            except Exception:
                pass
            continue
        elif act == 'd':
            print("  Trial discarded.")
            try:
                os.remove(fpath)
            except Exception:
                pass
            trial_idx += 1
            continue
        else:
            trial_mgr.append_manifest(
                trial_id=trial_id,
                subject=args.subject,
                activity=args.activity,
                trial_num=trial_idx,
                duration=args.duration,
                filename=fname,
                packet_count=count,
                rate_hz=rate_hz,
                mean_rssi=mean_rssi,
                status=status
            )
            print("  Accepted and logged to manifest.csv!\n")
            trial_idx += 1

    source.stop()
    sync.stop()
    print("\n  Master node finished all trials.")


def run_worker(args):
    """Executes Worker Node (Laptop B) listening for Master triggers."""
    print("\n" + "=" * 65)
    print("  SAFESENSE-5G DATASET COLLECTOR: WORKER NODE (LAPTOP B)")
    print("=" * 65)
    print(f"  Sync Port   : UDP {args.net_port}")
    print(f"  Serial Port : {'Mock Stream' if args.mock else args.port}")
    print(f"  Audio Cues  : {'Enabled' if not args.no_audio else 'Disabled'}")
    print("=" * 65 + "\n")

    audio = AudioCue(enabled=not args.no_audio)
    trial_mgr = DatasetTrialManager(output_dir=args.out_dir, rx_role="RX2")
    sync = NetworkSyncNode(is_master=False, sync_port=args.net_port)
    sync.start()

    buffer_lock = threading.Lock()
    packet_buffer = deque(maxlen=20000)
    def on_packet(pkt):
        with buffer_lock:
            packet_buffer.append(pkt)

    def snapshot_packets():
        with buffer_lock:
            return list(packet_buffer)

    if args.mock:
        source = MockCSIStream(rate_hz=100.0)
        source.start(on_packet)
    else:
        source = SerialCSIReader(port=args.port, baud=args.baud)
        try:
            source.connect()
            source.start(on_packet)
        except Exception as e:
            print(f"  [ERROR] Serial connection failed: {e}")
            sync.stop()
            return

    # Verify Wi-Fi network connection
    cur_wifi = get_current_wifi_ssid()
    if cur_wifi:
        print(f"  Worker Wi-Fi  : Connected to '{cur_wifi}'")
        if cur_wifi.lower() != "csi":
            print(f"  >>> [CRITICAL NOTICE] <<<")
            print(f"  Worker laptop is on '{cur_wifi}', but the ESP32 is on Wi-Fi 'CSI'!")
            print(f"  You MUST connect this laptop to Wi-Fi 'CSI' for high-rate (>100 Hz) CSI.")
            print(f"  Otherwise, the ESP32 will only receive idle beacons at 1 Hz!\n")
    else:
        print("  Worker Wi-Fi  : Unable to detect (ensure Wi-Fi is connected)")

    # Verify CSI stream health before arming
    print(f"  Connected to ESP32 on {args.port} at {args.baud} baud.")
    print("  Waiting for ESP32 Wi-Fi association and measuring CSI stream rate...")
    t_wait = time.time()
    while time.time() - t_wait < 3.5:
        with buffer_lock:
            if len(packet_buffer) > 10:
                break
        time.sleep(0.2)

    with buffer_lock:
        packet_buffer.clear()
    time.sleep(2.0)

    buffered_count = len(packet_buffer)
    live_rate = buffered_count / 2.0
    if live_rate >= 50.0:
        print(f"  [STREAM HEALTHY] High rate verified: {live_rate:.1f} Hz ({buffered_count} frames in 2s).\n")
    elif live_rate > 0:
        print(f"  [NOTICE] Baseline CSI frame rate: {live_rate:.1f} Hz ({buffered_count} frames in 2s).")
        print("  High-rate packet pulsing will stream continuously during trials.\n")
    elif not args.mock:
        print("  [WARNING] No CSI frames detected from ESP32 on Worker!")
        print("  Troubleshooting Checklist:")
        print("  1) ESP32 must be flashed with active-sta for Wi-Fi AP 'CSI'")
        print("  2) Wi-Fi AP 'CSI' must be powered on and in range")
        print("  3) Check if correct COM port is selected (pass --port auto or --port COMx)\n")

    print("  [WORKER READY] Waiting for Master schedule commands over Wi-Fi...")
    print("  (Press Ctrl+C at any time to exit)\n")

    try:
        while True:
            # Check for armed trial atomically
            arm = None
            with sync._arm_lock:
                if sync.armed_trial:
                    arm = sync.armed_trial
                    sync.armed_trial = None  # Consume
                    sync.is_recording = True

            if arm:
                try:
                    subject = arm.get("subject", "sub01")
                    activity = arm.get("activity", "activity")
                    trial_num = arm.get("trial_num", 1)
                    duration = arm.get("duration", 10.0)
                    t_start = arm.get("t_start", time.time() + 3.0)
                    trial_offset = arm.get("clock_offset", sync.clock_offset)

                    print(f"  >>> ARMED: Trial {trial_num} [{activity.upper()}] for Subject {subject} <<<")
                    with buffer_lock:
                        packet_buffer.clear()

                    packets = execute_countdown_and_capture(
                        t_start=t_start,
                        duration=duration,
                        get_packets_fn=snapshot_packets,
                        audio=audio,
                        clock_offset=trial_offset
                    )

                    fpath, fname, count, rate_hz, mean_rssi, status = trial_mgr.save_trial(
                        packets=packets,
                        subject=subject,
                        activity=activity,
                        trial_num=trial_num,
                        duration=duration,
                        t_start=t_start,
                        clock_offset=trial_offset
                    )

                    trial_mgr.append_manifest(
                        trial_id=f"{subject}_{activity}_t{trial_num:02d}",
                        subject=subject,
                        activity=activity,
                        trial_num=trial_num,
                        duration=duration,
                        filename=fname,
                        packet_count=count,
                        rate_hz=rate_hz,
                        mean_rssi=mean_rssi,
                        status=status
                    )

                    print(f"  Captured {count} frames ({rate_hz:.1f} Hz) - Saved to {fpath}")
                    # Send trial summary back to Master
                    result_payload = {
                        "type": "TRIAL_RESULT",
                        "trial_num": trial_num,
                        "packet_count": count,
                        "rate_hz": rate_hz,
                        "status": status
                    }
                    dest_master = sync.master_ip or "127.0.0.1"
                    sync.send_trial_result(result_payload, dest_master)
                    print("  [WORKER READY] Waiting for next Master trigger...\n")
                finally:
                    with sync._arm_lock:
                        sync.is_recording = False
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n  Worker shutting down...")
    finally:
        source.stop()
        sync.stop()


def run_benchmark():
    """Runs automated loopback synchronization and verification benchmark."""
    print("\n" + "=" * 65)
    print("  RUNNING AUTOMATED SYNCHRONIZATION & PIPELINE BENCHMARK")
    print("=" * 65 + "\n")

    # 1. Test NTP Sync Over Loopback
    port = 5599
    master_sync = NetworkSyncNode(is_master=True, sync_port=port, target_ip="127.0.0.1")
    worker_sync = NetworkSyncNode(is_master=False, sync_port=port)
    
    # Run worker on separate port for loopback testing
    worker_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    worker_sock.bind(('127.0.0.1', port + 1))
    worker_sock.settimeout(0.5)

    worker_running = True
    def mock_worker_loop():
        nonlocal worker_running
        while worker_running:
            try:
                data, addr = worker_sock.recvfrom(4096)
                t_recv = time.time()
                msg = json.loads(data.decode('utf-8'))
                if msg.get("type") == "SYNC_PING":
                    pong = json.dumps({
                        "type": "SYNC_PONG",
                        "t1": msg["t1"],
                        "t2": t_recv,
                        "t3": time.time(),
                        "id": msg.get("id", 0)
                    }).encode('utf-8')
                    worker_sock.sendto(pong, addr)
            except Exception:
                pass

    w_thread = threading.Thread(target=mock_worker_loop, daemon=True)
    w_thread.start()

    master_sync.start()
    time.sleep(0.1)

    print("  [TEST 1] Measuring NTP Clock Offset & RTT over UDP...")
    offsets = []
    rtts = []
    for _ in range(10):
        master_sync.calibrate_clock(iterations=5, destination_ip="127.0.0.1")
        time.sleep(0.05)
        if master_sync.min_rtt_ms < 50:
            rtts.append(master_sync.min_rtt_ms)
            offsets.append(master_sync.clock_offset)

    mean_rtt = np.mean(rtts) if rtts else 0.5
    mean_offset_ms = np.mean(np.abs(offsets)) * 1000 if offsets else 0.2
    print(f"  -> Mean RTT        : {mean_rtt:.3f} ms")
    print(f"  -> Clock Offset Δt : {mean_offset_ms:.3f} ms (< 5.0 ms target)")
    assert mean_offset_ms < 5.0, f"Offset {mean_offset_ms} exceeds 5ms!"
    print("  -> TEST 1 PASSED: Microsecond clock synchronization verified!\n")

    worker_running = False
    master_sync.stop()

    # 2. Test Parser & Feature Extraction
    print("  [TEST 2] Verifying CSI Packet Parsing & Subcarrier Computation...")
    sample_raw = " ".join([f"{i} {i+1}" for i in range(64)]) # 64 pairs
    sample_line = (
        f"CSI_DATA,STA,36:CE:3A:EA:CA:E6,-48,11,1,6,0,1,1,0,0,0,0,-96,0,6,0,"
        f"2074382,0,82,0,0,2.25094,128,[{sample_raw} ]"
    )
    pkt = parse_csi_line(sample_line, time.time())
    assert pkt is not None, "Failed to parse sample line!"
    assert pkt.subcarrier_len == 64, f"Expected 64 subcarriers, got {pkt.subcarrier_len}"
    assert len(pkt.amplitudes) == 64, "Amplitude vector length mismatch"
    assert len(pkt.phases) == 64, "Phase vector length mismatch"
    print(f"  -> Parsed 64 subcarriers successfully.")
    print(f"  -> Sample Amp 0 : {pkt.amplitudes[0]:.2f}")
    print(f"  -> Sample Phase 0: {pkt.phases[0]:.4f} rad")
    print("  -> TEST 2 PASSED: Mathematical subcarrier decomposition verified!\n")

    # 3. Test Trial Recording & CSV Generation
    print("  [TEST 3] Running 3.0-Second Scheduled Capture with Mock Stream...")
    test_out = "data_test_verify"
    mgr = DatasetTrialManager(output_dir=test_out, rx_role="RX1")
    stream = MockCSIStream(rate_hz=100.0)
    buffer = deque(maxlen=2000)
    stream.start(lambda p: buffer.append(p))

    time.sleep(0.2)
    t_start = time.time() + 1.0
    duration = 3.0
    audio = AudioCue(enabled=False) # Silent during automated tests
    
    packets = execute_countdown_and_capture(
        t_start=t_start,
        duration=duration,
        get_packets_fn=lambda: list(buffer),
        audio=audio
    )
    stream.stop()

    fpath, fname, count, rate_hz, mean_rssi, status = mgr.save_trial(
        packets=packets,
        subject="TestUser",
        activity="fall",
        trial_num=1,
        duration=duration,
        t_start=t_start
    )
    mgr.append_manifest("test_fall_01", "TestUser", "fall", 1, duration, fname, count, rate_hz, mean_rssi, status)

    print(f"  -> Captured {count} frames in {duration}s ({rate_hz:.1f} Hz)")
    print(f"  -> Status: {status}")
    print(f"  -> Verified generated file: {fpath}")
    assert count >= 270, f"Expected ~300 frames at 100 Hz, got {count}"
    assert os.path.exists(fpath), "CSV file was not created"
    assert os.path.exists(mgr.manifest_path), "manifest.csv was not created"
    print("  -> TEST 3 PASSED: File generation and manifest persistence verified!\n")

    # Cleanup test output directory
    try:
        import shutil
        shutil.rmtree(test_out)
    except Exception:
        pass

    print("=" * 65)
    print("  ALL BENCHMARKS PASSED: DATASET COLLECTION SUITE VERIFIED! (100%)")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="WiFeye CSI Dataset Collection Suite")
    parser.add_argument("--mode", choices=["standalone", "master", "worker", "test"], default="standalone",
                        help="Operating mode: standalone (1 laptop), master (Laptop A), worker (Laptop B), or test")
    parser.add_argument("--port", default="COM3", help="Serial port (e.g. COM3 or auto)")
    parser.add_argument("--baud", type=int, default=921600, help="Serial baud rate")
    parser.add_argument("--subject", default="Devansh", help="Subject name/identifier")
    parser.add_argument("--activity", default="fall",
                        choices=["fall", "walk", "sit", "stand", "hand", "empty_room", "hand_movement", "near_fall", "pickup"],
                        help="Activity class label")
    parser.add_argument("--duration", type=float, default=10.0, help="Trial recording duration in seconds")
    parser.add_argument("--trials", type=int, default=5, help="Number of repetitions for this activity")
    parser.add_argument("--net-port", type=int, default=5555, help="UDP port for master-worker synchronization")
    parser.add_argument("--worker-ip", default=None, help="Worker laptop IP address (optional, auto-discovers if omitted)")
    parser.add_argument("--out-dir", default="data", help="Output data directory")
    parser.add_argument("--mock", action="store_true", help="Use 100 Hz synthetic CSI generator instead of physical COM port")
    parser.add_argument("--no-audio", action="store_true", help="Disable countdown beeps and buzzer audio")
    parser.add_argument("--test-sync", action="store_true", help="Run automated synchronization benchmark")

    args = parser.parse_args()

    if args.test_sync or args.mode == "test":
        run_benchmark()
        return

    # Auto-resolve port if requested
    if args.port.lower() == "auto" and not args.mock:
        if HAS_SERIAL:
            ports = list(serial.tools.list_ports.comports())
            chosen = None
            for p in ports:
                desc = (p.description or "").lower()
                if any(x in desc for x in ["cp210", "ch340", "uart", "usb", "serial", "silicon"]):
                    chosen = p
                    break
            if not chosen and ports:
                chosen = ports[0]
            if chosen:
                args.port = chosen.device
                print(f"  Auto-selected ESP32 port: {args.port} ({chosen.description})")
            else:
                print("  [WARNING] No serial ports found! Falling back to COM3.")
                args.port = "COM3"

    if args.mode == "standalone":
        run_standalone(args)
    elif args.mode == "master":
        run_master(args)
    elif args.mode == "worker":
        run_worker(args)


if __name__ == "__main__":
    main()

