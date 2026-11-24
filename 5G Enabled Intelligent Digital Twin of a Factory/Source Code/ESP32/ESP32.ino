#include <WiFi.h>
#include <WiFiUdp.h>
#include <ArduinoJson.h>
#include "secrets.h"

// Arduino UNO Q destination & local listen port
const char* UNO_Q_IP = "10.153.9.109";
const uint16_t UNO_Q_UDP_PORT = 5005;
const uint16_t ESP_LISTEN_PORT = 5006;

// ESP32-S3 UART pins
#define NANO_RX_PIN 18
#define NANO_TX_PIN 17
#define UART_BAUD 9600

WiFiUDP udp;
String uartLine;
char udpPacketBuffer[256];

unsigned long lastReconnectAttempt = 0;
const unsigned long RECONNECT_INTERVAL_MS = 5000;

bool connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) return true;

  Serial.println("Starting Wi-Fi connection...");
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 20000) {
    delay(500);
    Serial.print(".");
  }

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("\nWi-Fi connection timed out.");
    return false;
  }

  Serial.println("\nWi-Fi connected.");
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());

  udp.begin(ESP_LISTEN_PORT);
  Serial.print("Listening for UNO Q commands on UDP port ");
  Serial.println(ESP_LISTEN_PORT);
  return true;
}

// Uplink: Nano UART -> UNO Q UDP (5005)
void readNanoUART() {
  while (Serial2.available() > 0) {
    char c = (char)Serial2.read();

    if (c == '\n') {
      uartLine.trim();
      if (uartLine.length() == 0) {
        uartLine = "";
        continue;
      }

      JsonDocument doc;
      DeserializationError error = deserializeJson(doc, uartLine);
      if (error) {
        Serial.print("Invalid JSON from Nano: ");
        Serial.println(error.c_str());
        uartLine = "";
        continue;
      }

      if (WiFi.status() == WL_CONNECTED) {
        udp.beginPacket(UNO_Q_IP, UNO_Q_UDP_PORT);
        udp.print(uartLine);
        udp.endPacket();
        Serial.println("[UP] Forwarded Nano JSON to UNO Q");
      }
      uartLine = "";
    }
    else if (c != '\r') {
      if (uartLine.length() < 900) uartLine += c;
      else uartLine = "";
    }
  }
}

// Downlink: UNO Q UDP (5006) -> Nano UART
void readUNOQCommands() {
  int packetSize = udp.parsePacket();
  if (packetSize <= 0) return;

  int len = udp.read(udpPacketBuffer, sizeof(udpPacketBuffer) - 1);
  if (len <= 0) return;
  udpPacketBuffer[len] = '\0';

  String cmd = String(udpPacketBuffer);
  cmd.trim();

  Serial.print("[DOWN] Received from UNO Q: ");
  Serial.println(cmd);

  if (cmd == "FAN:ON" ||
      cmd == "FAN:OFF" ||
      cmd == "ML:FAN:ON" ||
      cmd == "ML:FAN:OFF" ||
      cmd == "MOTOR:ON" ||
      cmd == "MOTOR:OFF") {
    Serial2.println(cmd);
    Serial.println("Command forwarded to Arduino Nano.");
  } else {
    Serial.println("Unknown command rejected.");
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial2.begin(UART_BAUD, SERIAL_8N1, NANO_RX_PIN, NANO_TX_PIN);
  Serial.println("\nESP32-S3 <-> UNO Q Bidirectional UDP Bridge");

  WiFi.setAutoReconnect(true);
  connectWiFi();
}

void loop() {
  readNanoUART();
  readUNOQCommands();

  if (WiFi.status() != WL_CONNECTED) {
    unsigned long now = millis();
    if (now - lastReconnectAttempt >= RECONNECT_INTERVAL_MS) {
      lastReconnectAttempt = now;
      connectWiFi();
    }
  }

  delay(2);
}