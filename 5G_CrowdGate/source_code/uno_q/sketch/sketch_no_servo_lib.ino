// CrowdGate gate controller - same as sketch.ino but WITHOUT the Servo library (servo pulses are sent by hand).
#include <Arduino_RouterBridge.h>

const int LED_G = 4, LED_Y = 5, LED_R = 6, BUZZER = 7, SERVO_PIN = 9;
const int BARRIER_OPEN = 90, BARRIER_CLOSED = 0;   // swap these if the barrier moves the wrong way

int state = 3;                 // 0 GREEN, 1 YELLOW, 2 RED, 3 FAIL-SAFE
unsigned long lastServo = 0;

void setState(int s) { state = s; }

void setup() {
  pinMode(LED_G, OUTPUT); pinMode(LED_Y, OUTPUT); pinMode(LED_R, OUTPUT);
  pinMode(BUZZER, OUTPUT); pinMode(SERVO_PIN, OUTPUT);
  Bridge.begin();
  Bridge.provide_safe("set_state", setState);
}

void loop() {
  unsigned long now = millis();
  bool blink = ((now / 400) % 2) == 0;

  digitalWrite(LED_G, state == 0 ? HIGH : LOW);
  digitalWrite(LED_Y, (state == 1 && blink) ? HIGH : LOW);
  digitalWrite(LED_R, (state == 2 || (state == 3 && blink)) ? HIGH : LOW);

  bool beep = (state >= 2) && ((now % 1000) < 200);
  digitalWrite(BUZZER, beep ? HIGH : LOW);

  if (now - lastServo >= 20) {                              // servo pulse every 20 ms
    lastServo = now;
    int angle = (state <= 1) ? BARRIER_OPEN : BARRIER_CLOSED;
    int us = map(angle, 0, 180, 544, 2400);
    digitalWrite(SERVO_PIN, HIGH); delayMicroseconds(us); digitalWrite(SERVO_PIN, LOW);
  }
}
