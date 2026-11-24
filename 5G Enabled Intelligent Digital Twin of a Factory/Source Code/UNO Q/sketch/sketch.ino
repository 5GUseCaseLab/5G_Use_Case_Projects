#include <Arduino_RouterBridge.h>
#include <DHT.h>

// ============================================================
// Pin Configuration (Arduino UNO Q)
// ============================================================
#define DHT_PIN 6
#define DHTTYPE DHT11
DHT dht(DHT_PIN, DHTTYPE);

#define MOTOR_IN1 4
#define MOTOR_IN2 5
#define MOTOR_PWM 9

#define ENCA 2
#define ENCB 3

#define CURRENT_PIN A0  // ACS712 OUT (via resistor divider)

// ============================================================
// Motor, Encoder & Current Sensor Settings
// ============================================================
const float TOTAL_PPR = 385.0; // 7 base pulses * 55 gear ratio

const float ADC_VREF           = 3.18;    // UNO Q STM32 ADC reference voltage
const float ADC_MAX            = 4095.0; // 12-bit ADC resolution
const float ACS712_SENSITIVITY = 0.100;  // 0.185 (5A), 0.100 (20A), 0.066 (30A)
const float DIVIDER_RATIO      = 1.57;    // Set to (R1 + R2) / R2 (e.g., 2.0 for 10k/10k)

volatile long encoderCount = 0;
long previousEncoderCount = 0;
float currentRPM = 0.0;
float temperatureC = 0.0;
float humidityPct = 0.0;

float zeroCurrentVolts = 2.5;
float filteredSignedAmps = 0.0;
float currentAmps = 0.0;

bool motorEnabled = false;
int targetPWM = 150;

unsigned long lastCalcTime = 0;
unsigned long lastCurrentTime = 0;

// ============================================================
// Encoder Interrupt Service Routine
// ============================================================
void readEncoder() {
  if (digitalRead(ENCB) > 0) {
    encoderCount++;
  } else {
    encoderCount--;
  }
}

// ============================================================
// Current Sensor Helper
// ============================================================
float readAveragedCurrentVolts(int samples) {
  long sum = 0;
  for (int i = 0; i < samples; i++) {
    sum += analogRead(CURRENT_PIN);
  }
  float avgAdc = (float)sum / (float)samples;
  return (avgAdc / ADC_MAX) * ADC_VREF * DIVIDER_RATIO;
}

// ============================================================
// Motor Control Helper
// ============================================================
void applyMotorOutput() {
  if (!motorEnabled || targetPWM == 0) {
    digitalWrite(MOTOR_IN1, LOW);
    digitalWrite(MOTOR_IN2, LOW);
    analogWrite(MOTOR_PWM, 0);
    return;
  }

  if (targetPWM > 0) {
    digitalWrite(MOTOR_IN1, HIGH);
    digitalWrite(MOTOR_IN2, LOW);
    analogWrite(MOTOR_PWM, targetPWM);
  } else {
    digitalWrite(MOTOR_IN1, LOW);
    digitalWrite(MOTOR_IN2, HIGH);
    analogWrite(MOTOR_PWM, -targetPWM);
  }
}

// ============================================================
// Bridge RPC Handlers: Motor Control (MPU -> MCU)
// ============================================================
// Pass 1 for ON, 0 for OFF
uint32_t set_motor_on(uint32_t state) {
  if (state > 0) {
    if (targetPWM == 0) targetPWM = 150;
    motorEnabled = true;
  } else {
    motorEnabled = false;
  }
  applyMotorOutput();
  return motorEnabled ? 1 : 0;
}

// Pass signed PWM (-255 to 255) as uint32_t
uint32_t set_motor_pwm(uint32_t raw_pwm) {
  int32_t signed_pwm = (int32_t)raw_pwm;
  targetPWM = constrain(signed_pwm, -255, 255);
  motorEnabled = (targetPWM != 0);
  applyMotorOutput();
  return (uint32_t)targetPWM;
}

// ============================================================
// Bridge RPC Handlers: Telemetry (MCU -> MPU)
// ============================================================
uint32_t get_rpm(uint32_t val) {
  return (uint32_t)((int32_t)(currentRPM * 100.0));
}

uint32_t get_enc(uint32_t val) {
  return (uint32_t)encoderCount;
}

uint32_t get_temp(uint32_t val) {
  return (uint32_t)(temperatureC * 10.0);
}

uint32_t get_hum(uint32_t val) {
  return (uint32_t)(humidityPct * 10.0);
}

// Returns motor current in milliamps (divide by 1000.0 in Python for Amps)
uint32_t get_current(uint32_t val) {
  return (uint32_t)(currentAmps * 1000.0f);
}

// ============================================================
// Serial Command Parser (Local Debugging)
// ============================================================
void handleSerialCommands() {
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    cmd.toUpperCase();

    if (cmd.length() == 0) return;

    if (cmd == "ON") {
      set_motor_on(1);
      Serial.print("Motor ON | PWM: ");
      Serial.println(targetPWM);
    }
    else if (cmd == "OFF" || cmd == "STOP") {
      set_motor_on(0);
      Serial.println("Motor OFF");
    }
    else if (cmd.startsWith("PWM:")) {
      int val = cmd.substring(4).toInt();
      set_motor_pwm((uint32_t)val);
      Serial.print("PWM set to: ");
      Serial.println(targetPWM);
    }
    else if (cmd == "CAL") {
      set_motor_on(0);
      delay(250);
      zeroCurrentVolts = readAveragedCurrentVolts(200);
      filteredSignedAmps = 0.0;
      currentAmps = 0.0;
      Serial.print("ACS712 Zero Calibrated (V): ");
      Serial.println(zeroCurrentVolts, 3);
    }
  }
}

// ============================================================
// Setup
// ============================================================
void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  dht.begin();

  pinMode(MOTOR_IN1, OUTPUT);
  pinMode(MOTOR_IN2, OUTPUT);
  pinMode(MOTOR_PWM, OUTPUT);
  applyMotorOutput();

  pinMode(ENCA, INPUT_PULLUP);
  pinMode(ENCB, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ENCA), readEncoder, RISING);

  // Calibrate 0 A baseline on startup while motor is OFF
  delay(250);
  zeroCurrentVolts = readAveragedCurrentVolts(200);

  if (!Bridge.begin()) {
    Serial.println("Bridge initialization failed!");
    return;
  }

  // Register motor control endpoints
  Bridge.provide("set_motor_on", set_motor_on);
  Bridge.provide("set_motor_pwm", set_motor_pwm);

  // Register telemetry endpoints
  Bridge.provide("get_rpm", get_rpm);
  Bridge.provide("get_enc", get_enc);
  Bridge.provide("get_temp", get_temp);
  Bridge.provide("get_hum", get_hum);
  Bridge.provide("get_current", get_current);

  unsigned long now = millis();
  lastCalcTime = now;
  lastCurrentTime = now;

  Serial.print("UNO Q MCU Ready. ACS712 Zero: ");
  Serial.print(zeroCurrentVolts, 3);
  Serial.println(" V");
}

// ============================================================
// Main Loop
// ============================================================
void loop() {
  handleSerialCommands();

  unsigned long now = millis();

  // Sample and smooth motor current every 100 ms
  if (now - lastCurrentTime >= 100) {
    lastCurrentTime = now;
    float volts = readAveragedCurrentVolts(32);
    float instSignedAmps = (volts - zeroCurrentVolts) / ACS712_SENSITIVITY;
    filteredSignedAmps = 0.85f * filteredSignedAmps + 0.15f * instSignedAmps;
    currentAmps = fabs(filteredSignedAmps);
  }

  if (now - lastCalcTime >= 1000) {
    unsigned long timeDelta = now - lastCalcTime;
    lastCalcTime = now;

    noInterrupts();
    long currentCount = encoderCount;
    interrupts();

    long countDelta = currentCount - previousEncoderCount;
    currentRPM = ((float)countDelta / TOTAL_PPR) * (60000.0 / (float)timeDelta);
    previousEncoderCount = currentCount;

    float newTemp = dht.readTemperature();
    float newHum = dht.readHumidity();
    if (!isnan(newTemp)) temperatureC = newTemp;
    if (!isnan(newHum)) humidityPct = newHum;
  }
}