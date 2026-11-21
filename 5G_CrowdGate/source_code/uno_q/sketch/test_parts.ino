// Parts test: green LED (D4), yellow LED (D5), red LED (D6), buzzer (D7), servo (D9). No network needed.
const int pins[] = {4, 5, 6, 7};

void servoMove(int angle) {          // holds the servo at "angle" for about 0.6 s (no library needed)
  int us = map(angle, 0, 180, 544, 2400);
  for (int i = 0; i < 30; i++) {
    digitalWrite(9, HIGH); delayMicroseconds(us);
    digitalWrite(9, LOW);  delay(20);
  }
}

void setup() {
  for (int i = 0; i < 4; i++) { pinMode(pins[i], OUTPUT); digitalWrite(pins[i], LOW); }
  pinMode(9, OUTPUT);
}

void loop() {
  for (int i = 0; i < 4; i++) {
    digitalWrite(pins[i], HIGH); delay(500);
    digitalWrite(pins[i], LOW);  delay(300);
  }
  servoMove(90);
  servoMove(0);
}
