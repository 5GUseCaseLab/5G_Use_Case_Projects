"""MEC server settings. Run everything from inside the mec_server/ folder."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# --- MQTT -------------------------------------------------------------
BROKER_HOST = os.environ.get("MQTT_HOST", "localhost")   # broker runs on this laptop
BROKER_PORT = int(os.environ.get("MQTT_PORT", "1883"))

# --- Shared secret (PROTOTYPE ONLY: never commit a real key) ----------
# Must be IDENTICAL on the UNO Q. Override with: set/export COLDCHAIN_KEY=...
SHARED_KEY = os.environ.get("COLDCHAIN_KEY", "change-me-demo-key").encode("utf-8")

# --- Devices / topics -------------------------------------------------
DEVICE_ID = "uno_q_edge_01"
ALLOWED_DEVICES = {DEVICE_ID}
TELEMETRY_SUB = "coldchain/+/telemetry"
EVENTS_SUB = "coldchain/+/events"
COMMAND_TOPIC_FMT = "coldchain/{device_id}/commands"

# --- Security policy --------------------------------------------------
MAX_CLOCK_SKEW_S = 30          # allowed |now - timestamp| for telemetry/events
MAX_MESSAGE_BYTES = 4096
TEMP_MIN_PLAUSIBLE = -60.0
TEMP_MAX_PLAUSIBLE = 150.0

# --- Control policy (example project thresholds, not a medical spec) --
SAFE_LOW_C = 7.0               # below this -> LOW alarm, cooling forced off
SAFE_HIGH_C = 13.0              # above this -> HIGH alarm
COOL_ON_C = 8.5                # start cooling at/above this
COOL_OFF_C = 6.5               # stop cooling at/below this (hysteresis band)
PWM_MIN_ON = 120
PWM_MAX = 255
PWM_GAIN_PER_C = 60            # extra PWM per degree above COOL_ON_C
COMMAND_HEARTBEAT_S = 10       # resend current command at least this often

# --- Files ------------------------------------------------------------
STATE_DIR = os.path.join(HERE, "state")
LOG_DIR = os.path.join(HERE, "logs")
TELEMETRY_STATE_FILE = os.path.join(STATE_DIR, "telemetry_seq.json")
COMMAND_STATE_FILE = os.path.join(STATE_DIR, "command_nonce.json")
TELEMETRY_CSV = os.path.join(LOG_DIR, "telemetry.csv")
SERVER_LOG = os.path.join(LOG_DIR, "server.log")
