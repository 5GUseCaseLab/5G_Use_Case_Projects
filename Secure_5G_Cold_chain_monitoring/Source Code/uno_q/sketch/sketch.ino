#include <OneWire.h>
#include <DallasTemperature.h>
#include "Arduino_RouterBridge.h"

// ---------------- Pins ----------------
const int ONE_WIRE_BUS = 2;     // DS18B20 data (with 4.7k pull-up to 3.3V)
const int COOLING_PIN  = 5;     // <-- CHANGE to the PWM-capable (~) pin wired to your driver

// ---------------- Driver polarity ----------------
// false = direct MOSFET drive (HIGH = cooling ON)
// true  = NPN inverter stage in front of the MOSFET (LOW = cooling ON)
const bool INVERTED_DRIVER = false;

// ---------------- Safety limits (enforced HERE, independent of the laptop and Python) ----------------
const int MAX_PWM = 255;                    // hard cap on output; lower it to protect the Peltier/supply
const unsigned long WATCHDOG_MS = 15000;    // no valid update for this long -> cooling OFF

OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);

volatile int currentPwm = 0;
volatile unsigned long lastUpdateMs = 0;

// Single place that touches the output pin.
void writeCooling(int pwm) {
  pwm = constrain(pwm, 0, MAX_PWM);
  currentPwm = pwm;
  analogWrite(COOLING_PIN, INVERTED_DRIVER ? 255 - pwm : pwm);
}

// ---- Exposed to Python via Bridge ----
float get_temperature() {
  sensors.requestTemperatures();
  return sensors.getTempCByIndex(0);        // -127.0 means "sensor disconnected"
}

int set_cooling_pwm(int pwm) {
  if (pwm < 0 || pwm > 255) {               // invalid request: fail safe
    writeCooling(0);
    return -1;
  }
  writeCooling(pwm);
  lastUpdateMs = millis();
  return currentPwm;                         // value actually applied (after the cap)
}

void setup() {
  pinMode(COOLING_PIN, OUTPUT);
  writeCooling(0);                           // FIRST thing: output OFF at every boot/reset

  Serial.begin(9600);
  delay(3000);                               // allow USB serial to settle

  sensors.begin();
  Bridge.begin();
  Bridge.provide("get_temperature", get_temperature);
  Bridge.provide("set_cooling_pwm", set_cooling_pwm);
  Serial.println("Ready: get_temperature, set_cooling_pwm");
}

void loop() {
  // Watchdog: if Python stops refreshing the output (crash, Bridge down), switch cooling OFF.
  if (currentPwm > 0 && (millis() - lastUpdateMs) > WATCHDOG_MS) {
    writeCooling(0);
    Serial.println("WATCHDOG: no update, cooling OFF");
  }
  delay(100);
}
