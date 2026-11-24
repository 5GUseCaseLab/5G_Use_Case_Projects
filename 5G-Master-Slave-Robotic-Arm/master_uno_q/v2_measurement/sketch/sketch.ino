// =====================================================================
//  UNO Q  -  MCU side (STM32)  -  reads the 3 master-arm potentiometers
//  and pushes the raw ADC values to the Linux/Python side over the Bridge.
//
//  MEASUREMENT VERSION - also sends how long the ADC reads took (us),
//  and answers a "ping" so Python can time the Bridge hop.
//
//  Wiring (each pot):  outer leg -> 3V3   |  other outer leg -> GND
//                      middle leg (wiper) -> A2 / A3 / A4
//  !! Power the pots from 3V3, NOT 5V. The ADC only reads 0-3.3 V, and
//  !! A0/A1 are not even 5 V tolerant.
// =====================================================================
#include <Arduino_RouterBridge.h>

const int POT_PINS[3] = {A2, A3, A4};    // base, shoulder, elbow
const unsigned long PERIOD_MS = 20;      // 50 Hz
const int SAMPLES = 4;                   // average 4 readings per pot (kills ADC noise)

// Python calls this and times the round trip -> Bridge latency
int ping() {
  return 1;
}

void setup() {
  Bridge.begin();
  analogReadResolution(12);              // values 0..4095
  Bridge.provide("ping", ping);
}

void loop() {
  static unsigned long last = 0;
  unsigned long now = millis();
  if (now - last < PERIOD_MS) return;
  last = now;

  unsigned long t0 = micros();           // --- start of ADC timing
  int raw[3];
  for (int i = 0; i < 3; i++) {
    analogRead(POT_PINS[i]);             // throw-away read: lets the ADC settle after switching channel
    long sum = 0;
    for (int k = 0; k < SAMPLES; k++) sum += analogRead(POT_PINS[i]);
    raw[i] = sum / SAMPLES;
  }
  int adcUs = (int)(micros() - t0);      // --- time spent reading all 3 pots

  // fire-and-forget message to python/main.py -> on_pots(base, shoulder, elbow, adc_us)
  Bridge.notify("pots", raw[0], raw[1], raw[2], adcUs);
}
