#include <Arduino_RouterBridge.h>

const int BUZZER_PIN = 4;
const int LED_CHANNELS[] = {2, 3, 5, 6, 7, 8, 9, 13}; // D0-D7
const int RED_PIN = 10;
const int GREEN_PIN = 11;
const int BLUE_PIN = 12;

void set_pin(int pin, int state) {
  digitalWrite(pin, state ? HIGH : LOW);
}

void setup() {
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, HIGH);
  delay(2000);
  digitalWrite(BUZZER_PIN, LOW);
  for (int i = 0; i < 8; i++) {
    pinMode(LED_CHANNELS[i], OUTPUT);
    digitalWrite(LED_CHANNELS[i], HIGH); // OFF (active-LOW)
  }

  pinMode(RED_PIN, OUTPUT);
  pinMode(GREEN_PIN, OUTPUT);
  pinMode(BLUE_PIN, OUTPUT);

  digitalWrite(RED_PIN, HIGH);
  digitalWrite(GREEN_PIN, HIGH);
  digitalWrite(BLUE_PIN, HIGH);

  Bridge.begin();
  Bridge.provide("set_pin", set_pin);
}

void loop() {}