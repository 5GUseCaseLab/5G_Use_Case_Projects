// =====================================================================
//  ESP32  -  slave arm.   ***  MEASUREMENT VERSION  ***
//  Receives "seq,t_us,base,shoulder,elbow" UDP packets from the UNO Q,
//  drives 3 servos, and echoes every packet straight back so the UNO Q
//  can measure the round-trip latency.
//  (Still accepts the old "base,shoulder,elbow" format, without stats.)
//
//  Prints every second:  packets/s, data rate, lost, reordered, biggest
//  gap between packets, jitter (RFC 3550), processing time, angles.
//  Prints every 10 s:    totals for the whole run.
//
//  Library: ESP32Servo   |   Board: "ESP32 Dev Module"
//  Servo wiring: brown -> supply(-) AND ESP32 GND | red -> supply(+) 5-6 V
//                orange -> GPIO 25 / 26 / 27
// =====================================================================
#include <WiFi.h>
#include <WiFiUdp.h>
#include <ESP32Servo.h>

const char* WIFI_SSID = "YOUR_WIFI_NAME";      // 2.4 GHz network shared with the Uno Q
const char* WIFI_PASS = "YOUR_WIFI_PASSWORD";
const uint16_t UDP_PORT = 5006;

const int SERVO_PINS[3] = {25, 26, 27};     // base, shoulder, elbow
const int PULSE_MIN_US = 500;
const int PULSE_MAX_US = 2400;

const int LIMIT_LO[3] = {0, 15, 10};
const int LIMIT_HI[3] = {180, 165, 170};
const int START_POS[3] = {90, 90, 90};

const unsigned long LINK_TIMEOUT_MS = 500;

Servo servos[3];
WiFiUDP udp;
char buf[96];
char out[112];
int current[3];
unsigned long lastPacketMs = 0;
bool linkUp = false;

// ---------------- measurement state ----------------
// whole run
unsigned long rxTotal = 0, lostTotal = 0, reorderTotal = 0, dupTotal = 0;
unsigned long outages = 0, longestOutageMs = 0, linkLostAtMs = 0;
unsigned long maxGapTotalUs = 0;
// this second
unsigned long winPkts = 0, winBytes = 0, winLost = 0, winReorder = 0;
unsigned long winProcSumUs = 0, winProcMaxUs = 0, winMaxGapUs = 0;
// sequence + jitter tracking
bool haveSeq = false;
unsigned long lastSeq = 0;
bool haveTransit = false;
unsigned long lastTransit = 0;
float jitterUs = 0;                 // RFC 3550 inter-arrival jitter
unsigned long lastArrivalUs = 0;

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
    WiFi.setSleep(false);                   // avoids latency spikes
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
  Serial.printf("Wi-Fi signal (RSSI): %d dBm\n", WiFi.RSSI());
}

// Handles one received packet: stats -> servos -> echo
void handlePacket(int n) {
  unsigned long tArrive = micros();
  buf[n] = '\0';

  unsigned long seq = 0, tUs = 0;
  int a[3];
  bool measured = (sscanf(buf, "%lu,%lu,%d,%d,%d", &seq, &tUs, &a[0], &a[1], &a[2]) == 5);
  if (!measured && sscanf(buf, "%d,%d,%d", &a[0], &a[1], &a[2]) != 3) return;   // junk

  bool apply = true;
  if (measured) {
    // biggest gap between two packets
    if (lastArrivalUs != 0) {
      unsigned long gap = tArrive - lastArrivalUs;
      if (gap > winMaxGapUs) winMaxGapUs = gap;
      if (gap > maxGapTotalUs) maxGapTotalUs = gap;
    }
    lastArrivalUs = tArrive;

    // master restarted (sequence jumped far back) -> start counting again
    if (haveSeq && seq < lastSeq && lastSeq - seq > 1000) {
      haveSeq = false;
      haveTransit = false;
    }
    if (!haveSeq) {
      haveSeq = true;
      lastSeq = seq;
    } else if (seq > lastSeq) {
      unsigned long missing = seq - lastSeq - 1;   // gap in numbering = lost packets
      lostTotal += missing;
      winLost += missing;
      lastSeq = seq;
    } else if (seq == lastSeq) {
      dupTotal++;
      apply = false;
    } else {                                       // older packet arriving late
      reorderTotal++;
      winReorder++;
      if (lostTotal) lostTotal--;                  // it was counted lost, it isn't
      if (winLost) winLost--;
      apply = false;                               // never move the arm backwards in time
    }

    // RFC 3550 jitter. The two clocks aren't synced, but only the CHANGE
    // in transit time is used, so the offset cancels out.
    unsigned long transit = tArrive - tUs;
    if (haveTransit) {
      long d = (long)(transit - lastTransit);
      if (d < 0) d = -d;
      jitterUs += ((float)d - jitterUs) / 16.0f;
    }
    lastTransit = transit;
    haveTransit = true;
  }

  if (apply) {
    for (int i = 0; i < 3; i++) {
      int target = constrain(a[i], LIMIT_LO[i], LIMIT_HI[i]);
      if (target != current[i]) {
        servos[i].write(target);
        current[i] = target;
      }
    }
  }

  rxTotal++;
  winPkts++;
  winBytes += n;
  lastPacketMs = millis();
  if (!linkUp) {
    linkUp = true;
    if (linkLostAtMs != 0) {
      unsigned long outage = millis() - linkLostAtMs;
      if (outage > longestOutageMs) longestOutageMs = outage;
      Serial.printf("Link UP again after %lu ms\n", outage);
    } else {
      Serial.println("Link UP - following master arm");
    }
  }

  unsigned long procUs = micros() - tArrive;
  winProcSumUs += procUs;
  if (procUs > winProcMaxUs) winProcMaxUs = procUs;

  // echo straight back to whoever sent it
  if (measured) {
    int m = snprintf(out, sizeof(out), "A,%lu,%lu,%lu,%d,%d,%d,%lu,%lu",
                     seq, tUs, procUs, current[0], current[1], current[2],
                     rxTotal, lostTotal);
    udp.beginPacket(udp.remoteIP(), udp.remotePort());
    udp.write((const uint8_t*)out, m);
    udp.endPacket();
  }
}

void setup() {
  Serial.begin(115200);
  delay(500);
  WiFi.onEvent(onWiFiEvent);

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
  int len;
  while ((len = udp.parsePacket()) > 0) {
    int n = udp.read(buf, sizeof(buf) - 1);
    if (n > 0) handlePacket(n);
  }

  // failsafe: no data -> hold last position and report it
  if (linkUp && millis() - lastPacketMs > LINK_TIMEOUT_MS) {
    linkUp = false;
    outages++;
    linkLostAtMs = lastPacketMs;
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
  if (millis() - lastStatus >= 1000) {
    float secs = (millis() - lastStatus) / 1000.0f;
    if (lastStatus == 0) secs = 1.0f;
    lastStatus = millis();
    Serial.printf("pkts/s=%lu  kbit/s=%.2f  lost=%lu  reord=%lu  maxGap=%.1fms  "
                  "jitter=%.2fms  proc=%luus(max %lu)  angles=%d,%d,%d  rssi=%d  link=%s\n",
                  winPkts, winBytes * 8 / secs / 1000.0f, winLost, winReorder,
                  winMaxGapUs / 1000.0f, jitterUs / 1000.0f,
                  winPkts ? winProcSumUs / winPkts : 0UL, winProcMaxUs,
                  current[0], current[1], current[2], WiFi.RSSI(), linkUp ? "up" : "down");
    winPkts = winBytes = winLost = winReorder = 0;
    winProcSumUs = winProcMaxUs = winMaxGapUs = 0;
  }

  // whole-run totals every 10 s
  static unsigned long lastTotals = 0;
  if (millis() - lastTotals >= 10000) {
    lastTotals = millis();
    unsigned long expected = rxTotal + lostTotal;
    Serial.println("---------------------------------------------------------------");
    Serial.printf("TOTALS  received %lu  lost %lu  delivery %.2f%%  reordered %lu  dup %lu\n",
                  rxTotal, lostTotal, expected ? 100.0f * rxTotal / expected : 0.0f,
                  reorderTotal, dupTotal);
    Serial.printf("        jitter %.2f ms  biggest gap %.1f ms  link outages %lu (longest %lu ms)\n",
                  jitterUs / 1000.0f, maxGapTotalUs / 1000.0f, outages, longestOutageMs);
    Serial.println("---------------------------------------------------------------");
  }
}
