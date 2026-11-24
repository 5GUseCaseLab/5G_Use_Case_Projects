#include <Arduino_RouterBridge.h>

const int TRIGGER_PIN = 2;               // button between D2 and GND (internal pull-up)
const unsigned long DEBOUNCE_MS = 30;

int lastRead = HIGH, lastStable = HIGH;
unsigned long tChange = 0;
bool latched = false;                    // press waiting to be collected by Python

// Returns 1 once per press, then clears. Python polls slowly, so latching
// here means a short press is never missed between polls.
int read_trigger() {
  int v = latched ? 1 : 0;
  latched = false;
  return v;
}

void setup() {
  pinMode(TRIGGER_PIN, INPUT_PULLUP);
  Bridge.begin();
  Bridge.provide_safe("read_trigger", read_trigger);  // runs between loop() passes, so no race on `latched`
}

void loop() {
  int r = digitalRead(TRIGGER_PIN);
  if (r != lastRead) { lastRead = r; tChange = millis(); }
  if (millis() - tChange > DEBOUNCE_MS && r != lastStable) {
    lastStable = r;
    if (r == LOW) latched = true;        // falling edge = button pressed
  }
}