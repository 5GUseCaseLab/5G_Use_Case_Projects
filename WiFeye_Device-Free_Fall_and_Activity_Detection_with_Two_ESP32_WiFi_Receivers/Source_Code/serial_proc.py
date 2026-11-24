"""
WiFeye: ESP32 serial reader in its own process (master side)
=============================================================
The master laptop also runs the classifier (~0.3 s of CPU per second with two receivers) and the dashboard.
A serial-reading *thread* in the same process can be starved by that work (Python's GIL), and Windows' default
serial receive buffer is only ~4 KB (about 45 ms at 921600 baud), so a stall silently drops CSI lines. The
140 Hz traffic pump that keeps the router transmitting was a thread in the same process too.

This module runs both in a separate process. Kept frames (router -> ESP32 HT frames, 52 subcarriers) are sent
to the parent through a multiprocessing queue, which buffers them if the parent is momentarily busy.
Messages on the queue:  ("pkt", t, amps52, rssi) · ("ready", port) · ("macs", [(mac, n), ...]) · ("error", msg)
"""

import time
import numpy as np

KEEP_SC = [i for i in range(64) if i != 0 and not (27 <= i <= 37)]


def reader_main(port, own_mac, q, stop_evt, serial_hz, kept_total, pump_hz=140.0):
    from dataset_collector import SerialCSIReader
    r = SerialCSIReader(port, 921600)
    r.pump_hz = pump_hz
    try:
        r.connect()
    except Exception as ex:
        q.put(("error", f"{port}: {ex}"))
        return
    try:
        r.ser.set_buffer_size(rx_size=1 << 20)      # Windows: ask the driver for a 1 MB receive buffer
    except Exception:
        pass
    own = own_mac.upper()
    others = {}

    def on_pkt(pkt):
        if pkt is None or pkt.sig_mode != 1:
            return
        m = pkt.mac.upper()
        if m != own:
            others[m] = others.get(m, 0) + 1          # only used to explain "nothing kept"
            return
        a = np.asarray(pkt.amplitudes, dtype=np.float32)
        if len(a) < 64:
            return
        q.put(("pkt", pkt.sys_time, a[KEEP_SC], float(pkt.rssi)))
        with kept_total.get_lock():
            kept_total.value += 1

    r.start(on_pkt)                                   # also starts the 140 Hz traffic pump, in this process
    q.put(("ready", port))
    t0, warned = time.time(), False
    try:
        while not stop_evt.is_set():
            time.sleep(0.5)
            serial_hz.value = float(r.live_rate_hz)
            if not warned and kept_total.value == 0 and others and time.time() - t0 > 8:
                warned = True
                q.put(("macs", sorted(others.items(), key=lambda kv: -kv[1])[:4]))
    finally:
        r.stop()
