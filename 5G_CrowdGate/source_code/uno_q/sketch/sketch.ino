// CrowdGate gate controller - runs on the Arduino UNO Q's microcontroller (STM32).
// Needs the sketch library "Arduino_RouterBridge" (add it in Arduino App Lab) and the Servo library.
// If Servo.h is not available on your board, use sketch_no_servo_lib.ino instead.
// State codes sent from the Python side: 0=GREEN 1=YELLOW 2=RED 3=FAIL-SAFE (link lost / camera failure)
#include <Arduino_RouterBridge.h>
#include <Servo.h>

const int LED_G = 4, LED_Y = 5, LED_R = 6, BUZZER = 7, SERVO_PIN = 9;
const int BARRIER_OPEN = 90, BARRIER_CLOSED = 0;   // servo angles - adjust to your mount

Servo barrier;
int state = 3;                                      // start in fail-safe until the MEC answers

void setState(int s) { state = s; }                 // called from Python: Bridge.call("set_state", code)

void setup() {
  pinMode(LED_G, OUTPUT); pinMode(LED_Y, OUTPUT); pinMode(LED_R, OUTPUT);
  pinMode(BUZZER, OUTPUT);
  barrier.attach(SERVO_PIN);
  Bridge.begin();
  Bridge.provide_safe("set_state", setState);
}

void loop() {
  unsigned long now = millis();
  bool blink = ((now / 400) % 2) == 0;

  digitalWrite(LED_G, state == 0 ? HIGH : LOW);
  digitalWrite(LED_Y, (state == 1 && blink) ? HIGH : LOW);                     // yellow blinks
  digitalWrite(LED_R, (state == 2 || (state == 3 && blink)) ? HIGH : LOW);     // red steady / fail-safe blinks

  bool beep = (state >= 2) && ((now % 1000) < 200);                            // short beep every second
  digitalWrite(BUZZER, beep ? HIGH : LOW);

  barrier.write(state <= 1 ? BARRIER_OPEN : BARRIER_CLOSED);                   // open in GREEN/YELLOW
}
