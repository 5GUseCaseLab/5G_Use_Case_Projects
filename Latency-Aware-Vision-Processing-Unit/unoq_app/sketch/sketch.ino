// MCU side (STM32U585). Receives an alert level from Python over the Router Bridge.
// 0 = clear, 1 = caution (slow blink), 2 = brake (fast blink).
// Replace the blink logic with a real brake / buzzer / CAN output on your vehicle.
#include <Arduino_RouterBridge.h>

volatile int level = 0;
unsigned long lastToggle = 0;
bool ledOn = false;

void set_alert(int l) { level = l; }

void led(bool on) { digitalWrite(LED_BUILTIN, on ? LOW : HIGH); }  // onboard LEDs are active-low

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  led(false);
  Bridge.begin();
  Bridge.provide("set_alert", set_alert);
}

void loop() {
  unsigned long now = millis();
  if (level == 0) { led(false); ledOn = false; return; }
  unsigned long period = (level == 1) ? 400 : 100;
  if (now - lastToggle >= period) {
    lastToggle = now;
    ledOn = !ledOn;
    led(ledOn);
  }
}
