"""UNO Q application settings. Run everything from inside the uno_q/ folder."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# Settings may come from environment variables (SSH runs) OR from local_settings.json
# next to this file (needed when the app is started by App Lab, which does not see your
# shell's `export` lines). Environment variables win if both are set.
_local = {}
try:
    with open(os.path.join(HERE, "local_settings.json")) as _f:
        _local = json.load(_f)
except (OSError, ValueError):
    pass


def _get(name, default):
    return os.environ.get(name, _local.get(name, default))


# --- MQTT: set MQTT_HOST to your LAPTOP's IP (e.g. 192.168.1.23) ------
BROKER_HOST = _get("MQTT_HOST", "localhost")
BROKER_PORT = int(_get("MQTT_PORT", "1883"))

# --- Shared secret: MUST match mec_server/config.py (prototype only) --
SHARED_KEY = _get("COLDCHAIN_KEY", "change-me-demo-key").encode("utf-8")

DEVICE_ID = "uno_q_edge_01"
TELEMETRY_TOPIC = f"coldchain/{DEVICE_ID}/telemetry"
COMMAND_TOPIC = f"coldchain/{DEVICE_ID}/commands"
EVENTS_TOPIC = f"coldchain/{DEVICE_ID}/events"

SAMPLE_PERIOD_S = 3.0

# --- Command policy ---------------------------------------------------
COMMAND_MAX_AGE_S = 30         # reject commands whose timestamp is older/newer than this
MAX_MESSAGE_BYTES = 4096

# --- Local fail-safe (used only when the MEC goes silent) -------------
MEC_TIMEOUT_S = 30             # no valid command for this long -> fail-safe
FAILSAFE_TEMP_C = 7.0          # at/above this, run cooling locally
FAILSAFE_PWM = 200

# --- Hardware / simulation -------------------------------------------
def _flag(name, default):
    return str(_get(name, default)).strip().lower() in ("1", "true", "yes", "on")


# Can also be set in local_settings.json / environment, e.g. "SENSOR_SIMULATION": false
SENSOR_SIMULATION = _flag("SENSOR_SIMULATION", False)     # False = real DS18B20 via the Bridge (App Lab only)
COOLING_SIMULATION = _flag("COOLING_SIMULATION", False)    # True until the MOSFET driver is wired (Step 6)
SIM_START_C = 4.5
SIM_HEAT_RATE = 0.25           # deg C gained per sample from the environment
SIM_COOL_RATE = 0.80           # deg C removed per sample at PWM 255
SIM_DOOR_OPEN_EVERY = 40       # every N samples a "door opening" spike
SIM_DOOR_SPIKE_C = 2.5

# --- Files ------------------------------------------------------------
STATE_DIR = _get("COLDCHAIN_STATE_DIR", os.path.join(HERE, "state"))
NONCE_FILE = os.path.join(STATE_DIR, "last_nonce.json")
TELEMETRY_SEQ_FILE = os.path.join(STATE_DIR, "telemetry_seq.json")
