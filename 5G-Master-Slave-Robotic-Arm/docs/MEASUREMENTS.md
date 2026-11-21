# Measuring latency, jitter, delivery and tracking (v2 build)

[← Back to README](../README.md)

The `v2_measurement` build runs the same control loop as v1 and measures the link while you use the arm. Upload **both** v2 files on the master and the v2 sketch on the ESP32.

## How it measures without synchronised clocks

Every packet carries a sequence number and the UNO Q's own timestamp. The ESP32 applies the angles and immediately sends the packet back with the angles it actually applied. The UNO Q then computes:

- **Round-trip time (RTT)** = now − timestamp, both read from the UNO Q's clock. One-way Wi-Fi latency ≈ RTT / 2.
- **Tracking error** = |angle the master wanted − angle the ESP32 applied|, per joint.
- **Delivery:** packets with no echo after 1 s count as lost. The ESP32 counts gaps in the sequence numbers separately, so you get the UNO Q → ESP32 direction on its own too.

The ESP32 also computes **RFC 3550 inter-arrival jitter**. The two clocks have an unknown offset, but this estimator only uses the *change* in transit time between packets, so the offset cancels out.

## Where each output appears

| Where | What |
|---|---|
| **App Lab Python console** (UNO Q) | a `REPORT` block every 5 s |
| `python/arm_log.csv` (UNO Q app folder) | one row per echoed packet: time, seq, RTT, master angles, sent angles, slave angles. Use it for the angle-vs-time plot |
| **Arduino IDE Serial Monitor** (ESP32, 115200 baud) | a status line every second, and `TOTALS` every 10 s |

Example ESP32 lines:

```
pkts/s=50  kbit/s=8.64  lost=0  reord=0  maxGap=31.2ms  jitter=2.84ms  proc=44us(max 71)  angles=92,45,130  rssi=-48  link=up
---------------------------------------------------------------
TOTALS  received 2980  lost 12  delivery 99.60%  reordered 0  dup 0
        jitter 3.10 ms  biggest gap 142.0 ms  link outages 0 (longest 0 ms)
---------------------------------------------------------------
```

## Reading the UNO Q report

| Report line | Meaning |
|---|---|
| `Packets` | sent / echoed / lost in the last window, and for the whole run |
| `Delivery` | round-trip delivery, and UNO Q → ESP32 delivery from the ESP32's own counters |
| `Latency` | RTT mean, median, p95, p99, max; *whole run* line adds one-way ≈ RTT/2 |
| `Jitter` | RTT standard deviation, mean change between consecutive RTTs, spikes above `SPIKE_MS` |
| `Pipeline` | mean time in each stage: ADC, Bridge (half of a `Bridge.call` round trip), waiting for the next 50 Hz send tick, Python, Wi-Fi (RTT/2), ESP32. Their sum is **pot → servo command** latency. Servo travel time is not included |
| `Data rate` | payload kbit/s and with 28 B of UDP/IP headers per packet, each direction |
| `Tracking` | per joint `mean/RMS/max` of \|master − slave\| in degrees, plus the average over all joints |
| `Filter` | noise and lag for raw (α = 1.0), α = 0.3, α = 0.1 and the live output, all from the same samples |

The `Filter` block labels each window **STILL**, **MOVING** or **slow/mixed**:

- In a **STILL** window, read the `noise ±deg` column: this is jitter at rest.
- In a **MOVING** window, read the `lag ms` column: how far each filter trails the raw signal (found by cross-correlation).

## Suggested session (about 2 minutes)

1. Start the ESP32 (v2), then run the UNO Q app (v2) with `CALIBRATE = False`.
2. **30 s still:** hold the master arm steady. Note the `Filter` noise values.
3. **60 s moving:** sweep each joint through its range at a natural speed. Note the `Filter` lag values.
4. **30 s still** again.
5. Take the **whole-run** values from the last report, and the ESP32's last `TOTALS` block.
6. Copy `arm_log.csv` off the board for the plot, e.g. `adb pull /home/arduino/ArduinoApps/<app>/python/arm_log.csv`.

## Filling the poster tables

| Poster item | Take it from |
|---|---|
| Mean latency headline | `Pipeline` → *pot → servo command* (add servo response if you quote potentiometer → servo motion) |
| Latency budget bar | the stage times in `Pipeline` |
| Packet delivery | `Delivery` (whole run) or the ESP32 `TOTALS` |
| RMS tracking error | `Tracking` → *whole run* → `all` / RMS values |
| Wi-Fi spikes | `Latency` max / p99 and `Jitter` spike count |
| Table 2 · jitter at rest | `Filter` → noise column, from a STILL window |
| Table 2 · latency | *pot → servo command* + that filter's `lag ms` from a MOVING window |
| Table 2 · RMS error | only printed for the α the arm is running. Re-run with `ALPHA = 1.0` and `ALPHA = 0.1` to fill the other rows |
| Shoulder tracking plot | `arm_log.csv`: `master_shoulder` and `slave_shoulder` against `t_s` |

Example plot script (run on a laptop):

```python
import pandas as pd, matplotlib.pyplot as plt
d = pd.read_csv("arm_log.csv")
plt.plot(d.t_s, d.master_shoulder, label="master θ₂")
plt.plot(d.t_s, d.slave_shoulder, "--", label="slave ψ₂")
plt.xlabel("time (s)"); plt.ylabel("shoulder angle (°)"); plt.legend(); plt.grid(alpha=.3)
plt.savefig("shoulder_tracking.png", dpi=200, bbox_inches="tight")
```
