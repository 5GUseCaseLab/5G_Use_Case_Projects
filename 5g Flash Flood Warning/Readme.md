# 5G-Based Flash Flood Early Warning System (Arduino UNO Q + URLLC Edge Telemetry)

A real-time IoT flash-flood monitoring and alert system built on the **Arduino UNO Q** and a **local/edge server**, paired with a live web dashboard and **Ultra-Reliable Low-Latency Communication (URLLC)** verification metrics.

---

## 1. Project Description

Flash floods develop rapidly due to heavy rainfall, saturated soil, and rising water levels. This project implements an edge-assisted sensing and telemetry pipeline that:

1. **Acquires and conditions sensor data** every `200 ms` across three environmental sensors using 5-sample averaging, normalization (`0–100%`), 3-sample confirmation, and a `5%` hysteresis band to prevent noise and alert flicker.
2. **Classifies flood risk** on the Arduino UNO Q into `SAFE`, `WARNING`, and `CRITICAL` states tailored to flash-flood runoff conditions.
3. **Streams telemetry over UDP (port `9000`)** from the board's Linux environment to a local/edge server (`server.py`) hosting a live browser dashboard (`http://127.0.0.1:8000`).
4. **Verifies URLLC link performance** over a 5-minute rolling window using clock-offset/sample-age calibration and lightweight UDP echo packets, tracking reliability, packet loss, end-to-end latency, jitter, P99 latency, and deadline success against a strict `100 ms` deadline.

---

## 2. System Architecture & Overall Workflow

```text
[Sensors: Rain A0, Water A1, Soil A2]
                 │
                 ▼  (5-sample avg, 200 ms loop, 3-sample confirm, 5% hysteresis)
[Arduino UNO Q Microcontroller (sketch.ino)]
                 │
                 ▼  (Arduino_RouterBridge: Bridge.notify("sensor_sample", ...))
[Arduino UNO Q Linux / App Lab Container (board_sender.py + protocol.py)]
                 │
                 ▼  (UDP port 9000 JSON packets + periodic clock sync)
[Local / Edge Server (server.py + protocol.py)] ◄──► UDP Echo (RTT + server proc time)
                 │
                 ▼  (HTTP /api/state on port 8000)
[Live Web Dashboard (dashboard.html)]
```

### Flood Risk Classification Logic (`sketch.ino`)

Raw sensor readings are averaged (`5` samples per `200 ms` tick) and normalized to `0–100%` using configured calibration bounds:

* **Thresholds (with `3`-sample confirmation and `5.0%` hysteresis):**
  * **Water Warning:** `55.0%`
  * **Water Critical:** `80.0%`
  * **Rain Limit:** `35.0%`
  * **Saturated Soil Limit:** `70.0%` (high moisture indicates reduced infiltration and higher runoff risk)
* **Event States & Priority:**
  * **`CRITICAL` (Priority `0`):** Water reaches `>= 80.0%`, **or** elevated water (`>= 55.0%`), active rain (`>= 35.0%`), and saturated soil (`>= 70.0%`) occur together.
  * **`WARNING` (Priority `1`):** Any single flood precursor is active (water `>= 55.0%`, rain `>= 35.0%`, or saturated soil `>= 70.0%`).
  * **`SAFE` (Priority `3`):** None of the warning or critical conditions are active.

### End-to-End URLLC Latency & Reliability Verification

Because the microcontroller, Linux container, and edge server do not share a hardware-synchronized clock, latency is estimated continuously:

1. **Sample Age & Clock Mapping:** The board sender periodically calibrates microcontroller `millis()` against Linux monotonic time so it can compute **sample age** (time elapsed between sensor sampling and UDP transmission).
2. **UDP Echo Round-Trip:** When `server.py` receives a UDP packet on port `9000`, it immediately replies with a UDP echo containing its own server-side processing time. The sender subtracts server processing time from round-trip time, divides by two, and adds sample age to estimate **end-to-end latency**.
3. **Rolling Window Metrics (`300 s` / 5 minutes):** `dashboard.html` and `/api/state` report:
   * **Reliability (%) & Packet Loss (%)** via sequence number tracking
   * **End-to-End Latency (`ms`)** & **Jitter (`ms`)**
   * **P99 Latency (`ms`)** (worst 1% tail latency)
   * **Deadline Success (%)** against a configurable deadline (`100 ms`)

---

## 3. Hardware & Software Dependencies

### Hardware
* **Arduino UNO Q** (microcontroller + Linux processor)
* **Water-level sensor** connected to pin `A1`
* **Soil moisture sensor** connected to pin `A2`
* **Rain sensor** connected to pin `A0`
* **Breadboard & jumper wires**
* **Host computer / Edge Server** (e.g., MacBook) on the same local/gateway network as the UNO Q, plus wireless/5G gateway connectivity

### Software & Libraries
* **Arduino / App Lab:**
  * `Arduino_RouterBridge.h` (included in the Arduino UNO Q App Lab environment)
  * Python 3 on the UNO Q Linux side (`board_sender.py`, `protocol.py`)
* **Edge Server (`server.py`):**
  * Python `3.9+` (uses Python standard library modules such as `socket`, `threading`, `http.server`, `json`, `statistics`, etc.)
  * Shared `protocol.py` defining message schema, validation, and allowed events (`SAFE`, `WARNING`, `CRITICAL`, plus legacy event names if needed for compatibility).

---

## 4. Execution Instructions

### Step 1: Verify `protocol.py` Event Allowlist
Ensure `protocol.py` on the server (and board) includes `"SAFE"`, `"WARNING"`, and `"CRITICAL"` in `EVENTS`:

```python
EVENTS = {"SAFE", "WARNING", "CRITICAL", "NORMAL", "SOIL_DRY", "HIGH_ALERT", "WATER_CRITICAL", "FLOOD_RISK"}
```

### Step 2: Upload `sketch.ino` to the Arduino UNO Q
1. Open the project sketch (`sketch.ino`) in Arduino IDE / Arduino App Lab.
2. Verify sensor pin assignments (`A0` rain, `A1` water level, `A2` soil moisture) and calibration constants (`WATER_LOW`/`WATER_HIGH`, `SOIL_DRY`/`SOIL_WET`, `RAIN_DRY`/`RAIN_WET`).
3. Compile and upload `sketch.ino` to the **Arduino UNO Q**, then confirm `SYSTEM_STARTED` appears on Serial (`115200` baud).

### Step 3: Start the Edge Server & Dashboard
On your edge server (e.g., MacBook), run `server.py` from your project directory:

```bash
cd /path/to/urllc
python3 server.py --udp-host 0.0.0.0 --udp-port 9000 --web-port 8000 --deadline-ms 100
```

### Step 4: Start / Restart the Board Sender
Ensure the Arduino UNO Q Python sender (`board_sender.py`) is running in the App Lab environment and targeting the edge server's IP address on UDP port `9000` (for example, `192.0.0.2:9000` or your local gateway/LAN IP). Restart the app if it was running before the sketch upload.

### Step 5: View Live Telemetry & Verify API State
1. Open **`http://127.0.0.1:8000`** in a web browser to view live sensor percentages, ADC values, current risk state (`SAFE`, `WARNING`, `CRITICAL`), and URLLC link metrics.
2. Optionally inspect the raw JSON telemetry state from the terminal:

```bash
curl -s http://127.0.0.1:8000/api/state
```

> **Troubleshooting Tip:** If the dashboard stays on *"Waiting for data"* while `unattributed_invalid` increases in `/api/state`, check that `protocol.py` in the directory where `server.py` runs contains `"SAFE"`, `"WARNING"`, and `"CRITICAL"` in `EVENTS`, then restart `server.py`.
