// =====================================================================
//  ESP32  -  slave arm. Receives "base,shoulder,elbow" UDP packets from
//  the UNO Q and drives 3 servos.
//
//  Library: ESP32Servo (Arduino IDE -> Library Manager -> "ESP32Servo")
//  Board:   "ESP32 Dev Module"
//
//  Wiring (each servo):
//     brown / black  -> external supply (-)   AND  ESP32 GND  (common ground!)
//     red            -> external supply (+)   5 V-6 V, 2-3 A
//     orange / yellow-> GPIO 25 / 26 / 27
//  Do NOT power the servos from the ESP32's 5V / VIN / 3V3 pin.
// =====================================================================
#include <WiFi.h>
#include <WiFiUdp.h>
#include <ESP32Servo.h>

const char* WIFI_SSID = "YOUR_WIFI_NAME";      // 2.4 GHz network shared with the Uno Q
const char* WIFI_PASS = "YOUR_WIFI_PASSWORD";
const uint16_t UDP_PORT = 5006;

const int SERVO_PINS[3] = {25, 26, 27};     // base, shoulder, elbow
const int PULSE_MIN_US = 500;                // MG90S pulse range
const int PULSE_MAX_US = 2400;

// keep these the same as LIMITS in main.py (second safety net)
const int LIMIT_LO[3] = {0, 15, 10};
const int LIMIT_HI[3] = {180, 165, 170};
const int START_POS[3] = {90, 90, 90};       // where the arm goes at power-up

const unsigned long LINK_TIMEOUT_MS = 500;  // no packet for this long -> "link lost"

Servo servos[3];
WiFiUDP udp;
char buf[64];
int current[3];
unsigned long lastPacketMs = 0;
bool linkUp = false;
unsigned long packets = 0;

// ---- prints WHY Wi-Fi failed ----
void onWiFiEvent(WiFiEvent_t event, WiFiEventInfo_t info) {
  if (event == ARDUINO_EVENT_WIFI_STA_DISCONNECTED) {
    int r = info.wifi_sta_disconnected.reason;
    Serial.printf("\n[WiFi] disconnect reason %d: ", r);
    switch (r) {
      case 201:           Serial.println("hotspot NOT FOUND (5 GHz / range / name)"); break;
      case 15: case 204:  Serial.println("WRONG PASSWORD"); break;
      case 202: case 2:   Serial.println("AUTH FAILED (password or WPA3-only)"); break;
      case 210: case 211: Serial.println("SECURITY MISMATCH (set hotspot to WPA2)"); break;
      default:            Serial.println("other"); break;
    }
  }
}

void connectWiFi() {
  while (true) {
    WiFi.disconnect(true);
    delay(200);
    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);                   // modem sleep adds 50-100 ms latency spikes
    WiFi.setTxPower(WIFI_POWER_8_5dBm);     // fixes connect issues on many boards
    WiFi.begin(WIFI_SSID, WIFI_PASS);

    Serial.printf("Connecting to '%s'", WIFI_SSID);
    unsigned long t0 = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - t0 < 10000) {
      delay(300);
      Serial.print(".");
    }
    if (WiFi.status() == WL_CONNECTED) break;
    Serial.println("\nNo luck in 10 s, retrying...");
  }
  Serial.println();
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());           // put this IP in main.py (ESP32_IP)
}

void setup() {
  Serial.begin(115200);
  delay(500);
  WiFi.onEvent(onWiFiEvent);

  // attach servos first so they hold a known position while Wi-Fi connects
  for (int i = 0; i < 3; i++) {
    servos[i].setPeriodHertz(50);
    servos[i].attach(SERVO_PINS[i], PULSE_MIN_US, PULSE_MAX_US);
    current[i] = START_POS[i];
    servos[i].write(current[i]);
  }

  connectWiFi();
  udp.begin(UDP_PORT);
  Serial.printf("Listening for angles on UDP port %u\n", UDP_PORT);
}

void loop() {
  // read every queued packet, keep only the newest
  int len;
  bool gotNew = false;
  int a[3];
  while ((len = udp.parsePacket()) > 0) {
    int n = udp.read(buf, sizeof(buf) - 1);
    if (n <= 0) continue;
    buf[n] = '\0';
    if (sscanf(buf, "%d,%d,%d", &a[0], &a[1], &a[2]) == 3) gotNew = true;
  }

  if (gotNew) {
    for (int i = 0; i < 3; i++) {
      int target = constrain(a[i], LIMIT_LO[i], LIMIT_HI[i]);
      if (target != current[i]) {
        servos[i].write(target);
        current[i] = target;
      }
    }
    lastPacketMs = millis();
    packets++;
    if (!linkUp) {
      linkUp = true;
      Serial.println("Link UP - following master arm");
    }
  }

  // failsafe: no data -> hold last position and report it
  if (linkUp && millis() - lastPacketMs > LINK_TIMEOUT_MS) {
    linkUp = false;
    Serial.println("Link LOST - holding position");
  }

  // auto-reconnect if the router drops us
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Wi-Fi dropped, reconnecting...");
    connectWiFi();
    udp.begin(UDP_PORT);
  }

  // status line once a second
  static unsigned long lastStatus = 0;
  if (millis() - lastStatus > 1000) {
    lastStatus = millis();
    Serial.printf("pkts/s=%lu  angles=%d,%d,%d  link=%s\n",
                  packets, current[0], current[1], current[2], linkUp ? "up" : "down");
    packets = 0;
  }
}
