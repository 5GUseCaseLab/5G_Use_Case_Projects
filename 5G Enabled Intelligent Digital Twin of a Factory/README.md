# The link for the demonstration video is attached here!

[Watch the Demonstration Video](https://iitgoffice-my.sharepoint.com/:v:/g/personal/anubhav_anand_iitg_ac_in/IQAyoiQQMM5MTIl13mO-zsuYAafoC6pwRvxKHbcAgmsV560?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXIiOiJNeUZpbGVzTGlua0NvcHkifX0&e=XEi7WN)

# 5G-Enabled Intelligent Digital Twin of a Factory

> **B.Tech Project — Department of Electronics and Electrical Engineering, IIT Guwahati**
> Anubhav Anand · Smit Kadu | Supervisor: Dr. Salil Kashyap

An end-to-end **digital twin** of a miniature factory cell. Sensors on the machines are read at the edge, safety rules run locally, telemetry goes to **AWS IoT Core**, a cloud classifier makes decisions, and a live **web dashboard** shows a mirror of the machines. Permitted commands flow back down to the actuators.

**Monitor → Analyze → Predict → Act**

---

## Table of Contents

1. [In Simple Words](#1-in-simple-words)
2. [System Architecture](#2-system-architecture)
3. [Overall Workflow](#3-overall-workflow)
4. [Repository Structure](#4-repository-structure)
5. [Hardware Requirements](#5-hardware-requirements)
6. [Wiring / Pin Map](#6-wiring--pin-map)
7. [Software Dependencies](#7-software-dependencies)
8. [Configuration — Values You Must Change](#8-configuration--values-you-must-change)
9. [How to Run (Step by Step)](#9-how-to-run-step-by-step)
10. [Message and Command Reference](#10-message-and-command-reference)
11. [Troubleshooting](#11-troubleshooting)
12. [Known Limitations and Scope](#12-known-limitations-and-scope)
13. [Acknowledgements and References](#13-acknowledgements-and-references)

---

## 1. In Simple Words

Imagine a toy factory with a little robot (**SARM**) and a motor.

- The **Arduino Nano** is the robot's *eyes and reflexes*. It sees how far things are, feels how hot it is, and if something comes too close it **stops the motor by itself**, without waiting for anyone.
- The **ESP32-S3** is the *messenger*. It carries the Nano's notes over Wi-Fi.
- The **Arduino UNO Q** is the *team leader*. It collects notes from everyone, drives its own motor, and talks to the cloud.
- **AWS** is the *big brain far away*. It remembers everything, decides "it's hot, turn the fan on", and sends you an email when something happens.
- The **Dashboard** is the *TV screen*. You watch the factory live and press buttons to turn motors on or off.

The key rule: **safety always stays local.** Even if the internet breaks, the robot still stops for obstacles.

---

## 2. System Architecture

```
 PHYSICAL LAYER          EDGE LAYER                         NETWORK              CLOUD LAYER (AWS)                       DASHBOARD
 ───────────────         ──────────                         ───────              ─────────────────                       ─────────
 HC-SR04 ×4  ─┐
 DHT11       ─┼──► Arduino Nano ──UART 9600──► ESP32-S3 ──UDP:5005──► UNO Q  ──MQTT/TLS:8883──► AWS IoT Core ─┬─► S3 (history)
 Fan relay   ◄┤    (safety interlock)          (Wi-Fi bridge)          (Linux MPU)                 topic:        ├─► Lambda HotColdInference (ML fan)
 Motor relay ◄┘                                                         │                          esp32s3/pub   └─► SNS IoTAlerts (email)
                                                                        │
 DC motor + encoder ─┐                                                  │
 ACS712 current     ─┼──► UNO Q STM32 MCU ◄──Bridge RPC──► UNO Q Linux MPU
 DHT11              ─┘                                                                         Lambda GetSensorData ◄── GET  /data    ◄── Dashboard.html
                                                                                               Lambda SendCommand   ◄── POST /command ◄── (browser)
                                                                                                     │
                                                                          ◄──── topic: esp32s3/sub ──┘
```

**Return path for commands:** Dashboard → API Gateway → `SendCommand` Lambda → AWS IoT (`esp32s3/sub`) → UNO Q → (STM32 motor **or** UDP → ESP32 → Nano).

| Layer | Component | Role |
|---|---|---|
| Physical | Sensors, relays, DC motors, SARM manipulator | Sense the environment and act on it |
| Edge | Arduino Nano | Reads SARM sensors, runs the **obstacle interlock**, drives fan/motor relays |
| Edge | ESP32-S3 | Two-way UART ↔ UDP bridge between the Nano and the UNO Q |
| Edge | Arduino UNO Q (STM32 MCU) | Reads encoder, current and DHT11; drives its own DC motor |
| Edge | Arduino UNO Q (Linux MPU) | Gateway: aggregates telemetry, MQTT/TLS link to AWS, routes commands |
| Cloud | AWS IoT Core + Rules | Ingests telemetry, fans it out to S3, Lambda and SNS |
| Cloud | Lambda + API Gateway | ML inference, dashboard data API, command API |
| UI | `Dashboard.html` | Live gauges, radar, 3D robot view, motor controls |

---

## 3. Overall Workflow

| Step | What happens | Where |
|:---:|---|---|
| **1. Sense** | Read 4 distances, temperature, humidity, motor RPM, encoder position, motor current | Nano, UNO Q MCU |
| **2. Process at the edge** | Check readings; if any distance is **< 15 cm**, the motor relay is forced OFF | Nano |
| **3. Transmit** | Nano → ESP32 (UART JSON) → UNO Q (UDP) → AWS IoT Core (MQTT over TLS) | ESP32, UNO Q |
| **4. Update the twin** | Dashboard polls `GET /data` every 1 s and redraws machine state | Dashboard |
| **5. Analyse** | Every message is stored in S3; logistic-regression demo classifies temperature HOT/COLD | AWS |
| **6. Decide** | Classifier issues `ML:FAN:ON` / `ML:FAN:OFF`; alert rule fires on fan/motor events | AWS |
| **7. Act** | Command travels back down the chain to the relay or motor driver | All layers |
| **8. Verify** | New sensor readings confirm the action; dashboard and email reflect it | Dashboard, SNS |

**Timing at a glance**

| Item | Period |
|---|---|
| Nano ultrasonic scan + interlock | 250 ms |
| Nano DHT11 read | 2 s |
| Nano telemetry → ESP32 | 1 s |
| UNO Q MCU RPM / DHT update | 1 s |
| UNO Q MCU current sampling | 100 ms |
| UNO Q telemetry → AWS | 10 s |
| Dashboard refresh | 1 s |

---

## 4. Repository Structure

```
Source Code/
├── Arduino Nano/
│   └── Arduino Nano.ino          # SARM sensors, obstacle interlock, relay control, UART JSON telemetry
├── ESP32/
│   └── ESP32.ino                 # UART <-> UDP bridge (Nano <-> UNO Q); needs secrets.h (you create it)
├── UNO Q/                        # Arduino App Lab app ("5G Project")
│   ├── app.yaml                  # App Lab app manifest
│   ├── sketch/
│   │   ├── sketch.ino            # STM32 MCU: motor driver, encoder, ACS712, DHT11, Bridge RPC endpoints
│   │   └── sketch.yaml           # Board platform + library list
│   └── python/
│       ├── main.py               # Linux MPU: AWS MQTT client, UDP listener, command router (app entry point)
│       ├── udp_relay.py          # Runs on the UNO Q HOST: forwards UDP between ESP32 and the app container
│       └── motor_cli.py          # Optional: interactive motor test tool (ON / OFF / PWM:n)
├── AWS/
│   └── Digital-twin-tempelate-template-*.yaml   # CloudFormation export of the AWS account (reference)
└── Dashboard/
    └── Dashboard.html            # Single-file web dashboard (Three.js 3D view)
```

---

## 5. Hardware Requirements

| Hardware | Qty | Used for |
|---|:---:|---|
| Arduino UNO Q | 1 | Edge gateway (Linux MPU) + motor controller (STM32 MCU) |
| Arduino Nano | 1 | SARM sensor acquisition and local safety logic |
| ESP32-S3 dev board | 1 | Wi-Fi bridge |
| HC-SR04 ultrasonic sensor | 4 | Front, rear, left, right distance |
| DHT11 temperature/humidity sensor | 2 | One on the Nano, one on the UNO Q |
| 2-channel relay module (active-LOW by default) | 1 | Fan and SARM motor enable |
| DC gear motor with quadrature encoder (7 PPR × 55:1 = 385 PPR) | 1 | UNO Q conveyor/base drive |
| H-bridge motor driver (IN1 / IN2 / PWM type) | 1 | Drives the UNO Q motor |
| ACS712 current sensor (20 A version assumed) + resistor divider | 1 | Motor current |
| Fan, SARM manipulator robot, power supplies, jumper wires | — | Representative factory assets |
| Wi-Fi network (or 5G CPE / hotspot) | 1 | ESP32 and UNO Q must be on the **same network** |

---

## 6. Wiring / Pin Map

### 6.1 Arduino Nano

| Nano pin | Connects to |
|---|---|
| D2 (SoftwareSerial **RX**) | ESP32-S3 **GPIO17** (TX) |
| D3 (SoftwareSerial **TX**) | ESP32-S3 **GPIO18** (RX) — *use a voltage divider (5 V → 3.3 V)* |
| D4 / D5 | Front HC-SR04 TRIG / ECHO |
| D6 / D7 | Rear HC-SR04 TRIG / ECHO |
| D8 / D9 | Left HC-SR04 TRIG / ECHO |
| D10 / D11 | Right HC-SR04 TRIG / ECHO |
| A0 | DHT11 data |
| D12 | Fan relay IN |
| D13 | Motor relay IN |
| GND | Common ground with the ESP32 |

### 6.2 ESP32-S3

| ESP32 pin | Connects to |
|---|---|
| GPIO18 (RX) | Nano D3 (through the divider) |
| GPIO17 (TX) | Nano D2 |
| GND | Nano GND |

### 6.3 Arduino UNO Q

| UNO Q pin | Connects to |
|---|---|
| D2 | Encoder channel A (interrupt) |
| D3 | Encoder channel B |
| D4 / D5 | Motor driver IN1 / IN2 |
| D9 | Motor driver PWM / EN |
| D6 | DHT11 data |
| A0 | ACS712 OUT **through a resistor divider** (the code assumes ratio `(R1+R2)/R2 = 1.57`) |

> ⚠️ The UNO Q and ESP32-S3 are **3.3 V** devices. Never feed a 5 V signal directly into their pins.

---

## 7. Software Dependencies

### 7.1 Tools

| Tool | Used for |
|---|---|
| [Arduino IDE 2.x](https://www.arduino.cc/en/software) | Flashing the Nano and the ESP32-S3 |
| [Arduino App Lab](https://docs.arduino.cc/software/app-lab/) | Building and running the UNO Q app |
| AWS account (region used: `eu-north-1`) | IoT Core, Lambda, API Gateway, S3, SNS |
| Any modern web browser | Opening the dashboard |

### 7.2 Board packages

| Board | Package (Boards Manager) |
|---|---|
| Arduino Nano | *Arduino AVR Boards* (built in) |
| ESP32-S3 | *esp32* by Espressif Systems |
| Arduino UNO Q | `arduino:zephyr` (installed by App Lab) |

### 7.3 Arduino libraries

| Library | Nano | ESP32 | UNO Q |
|---|:---:|:---:|:---:|
| DHT sensor library (Adafruit) 1.4.7 | ✅ | | ✅ |
| Adafruit Unified Sensor 1.1.15 | ✅ | | ✅ |
| SoftwareSerial (built in) | ✅ | | |
| ArduinoJson **v7.x** (uses `JsonDocument`) | | ✅ | |
| WiFi, WiFiUdp (part of the ESP32 core) | | ✅ | |
| Arduino_RouterBridge (Bridge 1.7.0) | | | ✅ |
| DHT11 and DHT22 Soldered Boards 1.0.0 | | | ✅ (listed in `sketch.yaml`) |

### 7.4 Python packages (UNO Q)

| Package | Where | Notes |
|---|---|---|
| `arduino.app_utils` | App container | Provided by App Lab (`App`, `Bridge`) |
| `paho-mqtt` | App container | MQTT client for AWS IoT. Add it to `UNO Q/python/requirements.txt` |
| Python 3 standard library (`socket`, `ssl`, `json`, `threading`, `subprocess`) | Container and host | No install needed |
| `docker` CLI | UNO Q **host** | Used by `udp_relay.py` to find the container's IP |

### 7.5 Dashboard

| Dependency | Notes |
|---|---|
| Three.js r128 | Loaded from the cdnjs CDN — the browser needs internet access |

### 7.6 AWS resources

| Resource | Name in this project | Purpose |
|---|---|---|
| IoT Thing | `ESP32_Device` | Device identity; its certificate is used by the UNO Q |
| IoT Policy | — | Lets the device connect, publish and subscribe |
| IoT Rule | `StoreSensorDataToS3` | `SELECT * FROM 'esp32s3/pub'` → S3 `sensor-data/${timestamp()}.json` |
| IoT Rule | `HotColdMLRule` | `SELECT * FROM 'esp32s3/pub'` → Lambda `HotColdInference` |
| IoT Rule | `SensorAlerts` | `... WHERE fan_on = true OR motor_allowed = true` → SNS |
| S3 bucket | `iot-sensor-data-<suffix>` | Telemetry history |
| Lambda | `HotColdInference` | Logistic-regression HOT/COLD classifier; env `CONTROL_MODE=ML`; sends fan commands |
| Lambda | `GetSensorData` | Backs `GET /data` for the dashboard |
| Lambda | `SendCommand` | Backs `POST /command`; publishes to `esp32s3/sub` |
| API Gateway (HTTP API) | `SARM-Dashboard-API`, stage `prod` | Routes `GET /data`, `POST /command`; CORS `*` |
| SNS topic | `IoTAlerts` | Email alerts for FAN_ON / MOTOR_ON / MOTOR_OFF |

All Lambdas: Python runtime, handler `lambda_function.lambda_handler`, 128 MB, 3 s timeout.

---

## 8. Configuration — Values You Must Change

The code contains values from the original deployment. Replace them with your own before running.

| File | Variable | What to put |
|---|---|---|
| `ESP32/secrets.h` *(create it)* | `WIFI_SSID`, `WIFI_PASSWORD` | Your Wi-Fi / 5G hotspot credentials |
| `ESP32/ESP32.ino` | `UNO_Q_IP` | IP address of the UNO Q on your network |
| `UNO Q/python/main.py` | `AWS_ENDPOINT` | Your IoT Core endpoint (*IoT Core → Settings → Device data endpoint*) |
| `UNO Q/python/main.py` | `UNO_Q_HOST_IP` | Same UNO Q IP as above |
| `UNO Q/python/main.py` | `CA_PATH`, `CERT_PATH`, `KEY_PATH` | Only change if you name the certificate files differently |
| `UNO Q/python/udp_relay.py` | `CONTAINER_NAME` | Name of the running app container (check with `docker ps`) |
| `Dashboard/Dashboard.html` | `API_URL` | Your API Gateway invoke URL, including `/prod` |
| `Arduino Nano/Arduino Nano.ino` | `RELAY_ACTIVE_LOW` | `false` if your relay module is active-HIGH |
| `Arduino Nano/Arduino Nano.ino` | `MOTOR_STOP_DISTANCE_CM` | Obstacle threshold (default **15 cm**) |
| `UNO Q/sketch/sketch.ino` | `ACS712_SENSITIVITY`, `DIVIDER_RATIO`, `ADC_VREF` | Match your ACS712 version and divider |

**Fixed ports and topics** (change on both sides if you change them):

| Item | Value |
|---|---|
| UDP uplink (ESP32 → UNO Q) | `5005` |
| UDP downlink (UNO Q → ESP32) | `5006` |
| MQTT port | `8883` (TLS 1.2) |
| Publish topic (telemetry) | `esp32s3/pub` |
| Subscribe topic (commands) | `esp32s3/sub` |

---

## 9. How to Run (Step by Step)

Follow the steps **in order**: cloud first, then devices from the bottom up, then the dashboard.

### Step 1 — Set up AWS

1. **Create the IoT Thing.** *AWS IoT Core → Manage → Things → Create thing* (e.g. `ESP32_Device`). Choose *auto-generate a certificate* and download:
   - `certificate.pem.crt`
   - `private.pem.key`
   - `AmazonRootCA1.pem`
2. **Attach a policy** to the certificate that allows `iot:Connect`, `iot:Publish`, `iot:Subscribe` and `iot:Receive` on your topics.
3. **Create the S3 bucket** for telemetry history.
4. **Create the three Lambda functions** (`HotColdInference`, `GetSensorData`, `SendCommand`), each with a role that can reach S3 and/or publish to IoT as needed.
5. **Create the IoT Rules** listed in [section 7.6](#76-aws-resources).
6. **Create the SNS topic** `IoTAlerts` with an email subscription, and confirm the email AWS sends you.
7. **Create an HTTP API** in API Gateway with `GET /data` → `GetSensorData` and `POST /command` → `SendCommand`, enable CORS (`GET, POST, OPTIONS`, header `content-type`) and deploy to stage `prod`. Copy the invoke URL.

> 📄 The YAML in `AWS/` is an *IaC Generator* export of the whole AWS account. It is the reference for how the project's resources were configured, but it also contains many unrelated account resources and hard-coded account IDs, and it **does not include the Lambda source code** (the code is referenced from S3). Rebuild the resources above by hand or trim the template to only those resources before deploying it.

### Step 2 — Flash the Arduino Nano

1. Open `Arduino Nano/Arduino Nano.ino` in Arduino IDE.
2. Install *DHT sensor library* and *Adafruit Unified Sensor*.
3. Select **Tools → Board → Arduino Nano** (try *ATmega328P (Old Bootloader)* if upload fails) and the right port.
4. Upload. Open the Serial Monitor at **115200 baud**. You should see:
   ```
   SARM Nano ready.
   Fan control: AWS ML.
   Motor commands: MOTOR:ON / MOTOR:OFF.
   Distances cm | Front:200.0 | Rear:200.0 | ... | Obstacle=NO | ...
   ```

### Step 3 — Flash the ESP32-S3

1. Create `ESP32/secrets.h` next to `ESP32.ino`:
   ```cpp
   #pragma once
   #define WIFI_SSID     "your-network-name"
   #define WIFI_PASSWORD "your-network-password"
   ```
2. Set `UNO_Q_IP` in `ESP32.ino` to your UNO Q's IP address.
3. Install **ArduinoJson (v7)**. Select your ESP32-S3 board and upload.
4. Serial Monitor at **115200 baud** should show:
   ```
   ESP32-S3 <-> UNO Q Bidirectional UDP Bridge
   Wi-Fi connected.
   ESP32 IP: 192.168.x.x
   Listening for UNO Q commands on UDP port 5006
   ```

### Step 4 — Deploy the UNO Q app

1. Copy the `UNO Q` folder into Arduino App Lab as an app (it appears as **"5G Project"**).
2. Put the three AWS certificate files in `UNO Q/python/certs/` so they appear inside the container at `/app/python/certs/`:
   ```
   python/certs/AmazonRootCA1.pem
   python/certs/certificate.pem.crt
   python/certs/private.pem.key
   ```
3. Create `UNO Q/python/requirements.txt` containing:
   ```
   paho-mqtt
   ```
4. Edit `AWS_ENDPOINT` and `UNO_Q_HOST_IP` in `python/main.py`.
5. Click **Run** in App Lab. App Lab flashes `sketch.ino` to the STM32 and starts `main.py` in a container. The log should show:
   ```
   [AWS] Connected to AWS IoT Core!
   [AWS] Subscribed to esp32s3/sub
   [UDP] Listening for ESP32-S3 packets on port 5005...
   [AWS SENT] {"device_id": "UNO-Q-MPU", "motor_rpm": ...}
   ```

### Step 5 — Start the UDP relay on the UNO Q host

The app runs inside a Docker container, so the ESP32 cannot reach it directly. `udp_relay.py` runs on the UNO Q's own Linux system and passes packets in both directions.

```bash
# On the UNO Q (SSH or the board's terminal)
docker ps                                  # note the app container name, e.g. 5g-project-main-1
nano udp_relay.py                          # set CONTAINER_NAME if it differs
python3 udp_relay.py
```

Expected output:
```
[Relay UP] Listening on 0.0.0.0:5005
[Relay DOWN] Listening on 0.0.0.0:5006
[Watcher] Found 5g-project-main-1 at 172.17.0.2
[UP] 412B from ESP (192.168.x.x) -> 172.17.0.2:5005
```

> If `docker` needs root, run `sudo python3 udp_relay.py`.

### Step 6 — Open the dashboard

1. Set `API_URL` in `Dashboard/Dashboard.html` to your API Gateway URL (ending in `/prod`).
2. Double-click the file to open it in a browser (or host it on S3 / any static web server).
3. Two sections appear:
   - **SARM-Robot-01** — temperature and humidity gauges, four distances on a radar, 3D proximity view, fan and motor status, Motor ON/OFF buttons.
   - **UNO-Q-MPU** — RPM, motor current, encoder position, temperature, humidity, Motor ON/OFF and a PWM slider (−255 … 255).

### Step 7 — Demonstration checklist

| # | Action | Expected result |
|:---:|---|---|
| 1 | Power everything on | Both badges on the dashboard show "Last packet: <time>" |
| 2 | Press **SARM → Motor ON** | Motor relay clicks; status shows **ON**; an email arrives |
| 3 | Put your hand < 15 cm from any ultrasonic sensor | Motor shows **BLOCKED**, obstacle **DETECTED** — even with Wi-Fi unplugged |
| 4 | Remove your hand | Motor resumes (the request is still ON) |
| 5 | Warm the DHT11 | Classifier sends `ML:FAN:ON`; fan relay turns on; email arrives |
| 6 | **UNO-Q → Set PWM** to −150 | UNO Q motor reverses; RPM becomes negative |
| 7 | **UNO-Q → Motor OFF** | Motor stops; RPM → 0 |

### Optional — Local testing without the cloud

| Tool | How | Commands |
|---|---|---|
| STM32 serial console | App Lab serial monitor at 115200 | `ON`, `OFF`, `STOP`, `PWM:150`, `PWM:-200`, `CAL` (re-zero the current sensor with the motor off) |
| `motor_cli.py` | `docker exec -it 5g-project-main-1 python3 /app/python/motor_cli.py` | `ON`, `OFF`, `PWM:<n>`, `EXIT` |

---

## 10. Message and Command Reference

### 10.1 Telemetry from the Nano (`device_id: "SARM-Robot-01"`, every 1 s)

```json
{
  "device_id": "SARM-Robot-01",
  "temperature_c": 27.40, "humidity_pct": 61.0,
  "front_cm": 200.0, "rear_cm": 84.3, "left_cm": 200.0, "right_cm": 12.7,
  "front_valid": true, "rear_valid": true, "left_valid": true, "right_valid": true,
  "sensor_fault": false, "obstacle_detected": true,
  "fan_on": false, "fan_mode": "ML",
  "motor_requested": true, "motor_allowed": false
}
```

- No echo or a reading above 200 cm is reported as **200 cm** (treated as clear).
- `motor_allowed = motor_requested AND NOT obstacle_detected`

### 10.2 Telemetry from the UNO Q (`device_id: "UNO-Q-MPU"`, every 10 s)

```json
{
  "device_id": "UNO-Q-MPU",
  "motor_rpm": 42.5, "motor_current_a": 0.312, "encoder_pos": 18204,
  "temperature_c": 26.8, "humidity_pct": 58.0, "timestamp_s": 1763700000
}
```

**How the MCU computes the numbers**

$$
\text{RPM} = \frac{\Delta\text{counts}}{385} \times \frac{60\,000}{\Delta t_{\text{ms}}}
$$

$$
V_{\text{sensor}} = \frac{\text{ADC}}{4095} \times 3.18 \times 1.57,
\qquad
I = \frac{V_{\text{sensor}} - V_{\text{zero}}}{0.100\ \text{V/A}}
$$

Current is smoothed with $I_{\text{filt}} \leftarrow 0.85\,I_{\text{filt}} + 0.15\,I_{\text{new}}$.

*Worked example:* 160 encoder counts in 1000 ms → $\frac{160}{385} \times 60 = 24.9$ RPM.

### 10.3 Commands (published to `esp32s3/sub`)

| Payload | Goes to | Effect |
|---|---|---|
| `MOTOR:ON` / `MOTOR:OFF` | Nano (via UDP → ESP32 → UART) | Request / cancel SARM motor (still gated by the interlock) |
| `FAN:ON` / `FAN:OFF` | Nano | Fan relay |
| `ML:FAN:ON` / `ML:FAN:OFF` | Nano | Fan relay (sent by the classifier) |
| `{"nano_cmd": "MOTOR:ON"}` | Nano | Same as the plain form, wrapped in JSON |
| `{"cmd": "ON"}` / `{"cmd": "OFF"}` | UNO Q STM32 | UNO Q motor on / off |
| `{"pwm": -150}` | UNO Q STM32 | Signed speed, −255 … 255 (sign = direction) |
| `ON`, `OFF`, `STOP`, `PWM:<n>` | UNO Q STM32 | Plain-text fallback |

Dashboard `POST /command` body: `{"command": "MOTOR:ON"}` for the SARM, `{"command": "{\"cmd\":\"ON\"}"}` for the UNO Q.

### 10.4 UNO Q Bridge RPC endpoints (MCU ↔ MPU)

| Endpoint | Direction | Returns / argument |
|---|---|---|
| `set_motor_on(state)` | MPU → MCU | `1` = on, `0` = off |
| `set_motor_pwm(pwm)` | MPU → MCU | Signed PWM packed as `uint32` |
| `get_rpm` | MCU → MPU | RPM × 100 (signed) |
| `get_enc` | MCU → MPU | Encoder count (signed) |
| `get_temp` / `get_hum` | MCU → MPU | Value × 10 |
| `get_current` | MCU → MPU | Milliamps |

---

## 11. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Nano prints `Temperature=N/A` | DHT11 wiring or library | Check the data pin (A0) and that the sensor has power |
| ESP32 prints `Invalid JSON from Nano` | Garbled UART | Common GND, correct TX/RX crossing, 9600 baud on both sides |
| ESP32 forwards but UNO Q log shows nothing | Relay not running, or wrong IP | Start `udp_relay.py`; check `UNO_Q_IP`; both devices on the same network |
| `[UP] Waiting for 5g-project-main-1 to start...` | Container name differs | Run `docker ps` and update `CONTAINER_NAME` |
| `[DOWN] Dropped: waiting for first packet` | Relay hasn't heard from the ESP32 yet | It learns the ESP32's IP from the first uplink packet — wait for telemetry |
| `[AWS] Connection failed` / TLS error | Wrong endpoint, certificate files or policy | Check `AWS_ENDPOINT`, cert paths, and that the policy is attached and the certificate is *Active* |
| `ModuleNotFoundError: paho` | Dependency missing | Add `paho-mqtt` to `python/requirements.txt` and re-run the app |
| Dashboard stays on "Waiting…" | `API_URL` wrong or CORS blocked | Check the browser console (F12); confirm CORS on the API |
| Motor never turns on | An ultrasonic sensor reads < 15 cm | Look at the Nano debug line; clear the path or adjust the threshold |
| Relay works backwards | Relay polarity | Toggle `RELAY_ACTIVE_LOW` |
| Current reads non-zero with motor off | Zero offset drift | Send `CAL` on the STM32 serial console |

---

## 12. Known Limitations and Scope

- **Network:** the working SARM link uses **Wi-Fi**. Running it over 5G and measuring 5G latency, throughput and reliability is the proposed next step.
- **ML model:** the HOT/COLD logistic-regression classifier is a demo trained on **synthetic data**.
- **Threshold:** the code uses a **15 cm** obstacle threshold (`MOTOR_STOP_DISTANCE_CM`); the poster mentions 20 cm. Set the value you want.
- **Data path:** in this code the ESP32 sends data to the UNO Q over UDP, and the UNO Q is the only device that talks to AWS. Both devices publish on the same `esp32s3/pub` topic, told apart by `device_id`.
- **Lambda source:** the Lambda function code is not in this repository; only its configuration is in the AWS template.
- **Security:** the API has no authentication and CORS allows any origin. The exported IoT policy allows `iot:*` on all resources. Tighten both before using this outside a lab. **Never commit** `secrets.h` or the certificate files — add `certs/` and `secrets.h` to `.gitignore`.

---

## 13. Acknowledgements and References

We thank our supervisor **Dr. Salil Kashyap** and Teaching Assistants **Aditya Gupta**, **Abhay Saran** and **Asif Hoque** for their guidance, and the Department of Electronics and Electrical Engineering, IIT Guwahati, for its support.

1. M. Grieves, *"Digital Twin: Manufacturing Excellence through Virtual Factory Replication,"* White Paper, 2014.
2. 3GPP TS 22.104, *"Service requirements for cyber-physical control applications in vertical domains."*
3. AWS IoT Core Documentation — <https://docs.aws.amazon.com/iot>
4. Arduino UNO Q Documentation — <https://docs.arduino.cc/hardware/uno-q>
