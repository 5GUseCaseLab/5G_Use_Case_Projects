#include <SoftwareSerial.h>
#include <DHT.h>
#include <math.h>

// ============================================================
// UART: Arduino Nano <-> ESP32-S3
// Nano D2 = RX, Nano D3 = TX
// ============================================================
SoftwareSerial espSerial(2, 3);

// ============================================================
// HC-SR04 sensors: Front, Rear, Left, Right
// ============================================================
const byte trigPins[4] = {4, 6, 8, 10};
const byte echoPins[4] = {5, 7, 9, 11};

const char *sensorNames[4] = {
  "Front", "Rear", "Left", "Right"
};

// ============================================================
// DHT sensor
// Use DHT11 or change to DHT22 for a DHT22 sensor.
// ============================================================
#define DHT_PIN A0
#define DHTTYPE DHT11

DHT dht(DHT_PIN, DHTTYPE);

// ============================================================
// Relay pins
// ============================================================
const byte FAN_RELAY_PIN = 12;
const byte MOTOR_RELAY_PIN = 13;

// Set true for active-LOW relay modules.
// Change to false if your relay module is active HIGH.
const bool RELAY_ACTIVE_LOW = true;

const byte RELAY_ON =
    RELAY_ACTIVE_LOW ? LOW : HIGH;

const byte RELAY_OFF =
    RELAY_ACTIVE_LOW ? HIGH : LOW;

// ============================================================
// Ultrasonic settings
// ============================================================
const float MOTOR_STOP_DISTANCE_CM = 15.0;
const float MAX_DISTANCE_CM = 200.0;

const unsigned long ECHO_TIMEOUT_US = 25000UL;

// ============================================================
// Timing
// ============================================================
const unsigned long SENSOR_INTERVAL_MS = 250;
const unsigned long DHT_INTERVAL_MS = 2000;
const unsigned long TELEMETRY_INTERVAL_MS = 1000;
const unsigned long DEBUG_INTERVAL_MS = 500;

// ============================================================
// Sensor and system state
// ============================================================
float distanceCm[4] = {
  200.0, 200.0, 200.0, 200.0
};

bool distanceValid[4] = {
  true, true, true, true
};

float temperatureC = NAN;
float humidityPct = NAN;

bool fanOn = false;

bool motorRequested = false;
bool motorAllowed = false;

bool obstacleDetected = false;
bool sensorFault = false;

unsigned long lastSensorRead = 0;
unsigned long lastDHTRead = 0;
unsigned long lastTelemetry = 0;
unsigned long lastDebug = 0;

// ============================================================
// FAN CONTROL
// The fan remains controlled by AWS ML commands.
// No manual fan mode is introduced.
// ============================================================
void setFan(bool turnOn) {
  fanOn = turnOn;

  digitalWrite(
    FAN_RELAY_PIN,
    turnOn ? RELAY_ON : RELAY_OFF
  );
}

// ============================================================
// Read one ultrasonic sensor
//
// No echo       -> 200 cm
// Distance >200 -> 200 cm
// Distance <=200 -> actual measured distance
//
// No echo is deliberately treated as clear space.
// ============================================================
float readDistance(
    byte trigPin,
    byte echoPin,
    bool &valid
) {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(3);

  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);

  digitalWrite(trigPin, LOW);

  unsigned long duration = pulseIn(
    echoPin,
    HIGH,
    ECHO_TIMEOUT_US
  );

  // No echo: treat the distance as 200 cm.
  if (duration == 0) {
    valid = true;
    return MAX_DISTANCE_CM;
  }

  float cm = (duration * 0.0343) / 2.0;

  // An implausible negative or zero result is treated as clear.
  if (cm <= 0.0) {
    valid = true;
    return MAX_DISTANCE_CM;
  }

  // Cap every reading above 200 cm at 200 cm.
  if (cm > MAX_DISTANCE_CM) {
    valid = true;
    return MAX_DISTANCE_CM;
  }

  // Any positive measured distance is retained,
  // including distances below the sensor's nominal range.
  valid = true;
  return cm;
}

// ============================================================
// Read all four ultrasonic sensors
//
// A sensor detects an obstacle only if its reported distance
// is strictly less than 15 cm.
// ============================================================
void readAllDistances() {
  bool anyObstacle = false;

  for (byte i = 0; i < 4; i++) {
    distanceCm[i] = readDistance(
      trigPins[i],
      echoPins[i],
      distanceValid[i]
    );

    if (distanceCm[i] < MOTOR_STOP_DISTANCE_CM) {
      anyObstacle = true;
    }

    // Reduce cross-talk between ultrasonic sensors.
    delay(35);
  }

  obstacleDetected = anyObstacle;

  // No-echo readings are intentionally considered valid.
  sensorFault = false;
}

// ============================================================
// Read temperature and humidity
// ============================================================
void readEnvironment() {
  float newTemperature = dht.readTemperature();
  float newHumidity = dht.readHumidity();

  if (!isnan(newTemperature)) {
    temperatureC = newTemperature;
  }

  if (!isnan(newHumidity)) {
    humidityPct = newHumidity;
  }
}

// ============================================================
// Motor control
//
// MOTOR:ON  -> enables automatic motor operation.
// MOTOR:OFF -> disables the motor.
//
// If ANY ultrasonic sensor reports <15 cm, stop the motor.
// No echo is reported as 200 cm and does not block operation.
// ============================================================
void updateMotor() {
  obstacleDetected = false;

  for (byte i = 0; i < 4; i++) {
    if (distanceCm[i] < MOTOR_STOP_DISTANCE_CM) {
      obstacleDetected = true;
    }
  }

  sensorFault = false;

  motorAllowed =
      motorRequested &&
      !obstacleDetected;

  digitalWrite(
    MOTOR_RELAY_PIN,
    motorAllowed ? RELAY_ON : RELAY_OFF
  );
}

// ============================================================
// Process commands received from AWS via ESP32 UART
//
// Fan:
// FAN:ON, FAN:OFF, ML:FAN:ON, ML:FAN:OFF
//
// Motor:
// MOTOR:ON, MOTOR:OFF
//
// No acknowledgement messages are sent.
// ============================================================
void handleCommand(String command) {
  command.trim();
  command.toUpperCase();

  if (command == "FAN:ON" ||
      command == "ML:FAN:ON") {
    setFan(true);
  }
  else if (command == "FAN:OFF" ||
           command == "ML:FAN:OFF") {
    setFan(false);
  }
  else if (command == "MOTOR:ON") {
    motorRequested = true;
    updateMotor();
  }
  else if (command == "MOTOR:OFF") {
    motorRequested = false;
    updateMotor();
  }

  // Unknown commands are ignored.
}

// ============================================================
// Read newline-terminated commands from ESP32
// ============================================================
void readESPCommands() {
  static String commandBuffer = "";

  while (espSerial.available() > 0) {
    char c = (char)espSerial.read();

    if (c == '\n') {
      commandBuffer.trim();

      if (commandBuffer.length() > 0) {
        handleCommand(commandBuffer);
      }

      commandBuffer = "";
    }
    else if (c != '\r') {
      if (commandBuffer.length() < 80) {
        commandBuffer += c;
      }
      else {
        commandBuffer = "";
      }
    }
  }
}

// ============================================================
// Send telemetry to ESP32 as one JSON line
// ============================================================
void publishTelemetry() {
  espSerial.print(F("{\"device_id\":\"SARM-Robot-01\""));

  espSerial.print(F(",\"temperature_c\":"));

  if (isnan(temperatureC)) {
    espSerial.print(F("null"));
  } else {
    espSerial.print(temperatureC, 2);
  }

  espSerial.print(F(",\"humidity_pct\":"));

  if (isnan(humidityPct)) {
    espSerial.print(F("null"));
  } else {
    espSerial.print(humidityPct, 1);
  }

  espSerial.print(F(",\"front_cm\":"));
  espSerial.print(distanceCm[0], 1);

  espSerial.print(F(",\"rear_cm\":"));
  espSerial.print(distanceCm[1], 1);

  espSerial.print(F(",\"left_cm\":"));
  espSerial.print(distanceCm[2], 1);

  espSerial.print(F(",\"right_cm\":"));
  espSerial.print(distanceCm[3], 1);

  espSerial.print(F(",\"front_valid\":"));
  espSerial.print(distanceValid[0] ? F("true") : F("false"));

  espSerial.print(F(",\"rear_valid\":"));
  espSerial.print(distanceValid[1] ? F("true") : F("false"));

  espSerial.print(F(",\"left_valid\":"));
  espSerial.print(distanceValid[2] ? F("true") : F("false"));

  espSerial.print(F(",\"right_valid\":"));
  espSerial.print(distanceValid[3] ? F("true") : F("false"));

  espSerial.print(F(",\"sensor_fault\":"));
  espSerial.print(sensorFault ? F("true") : F("false"));

  espSerial.print(F(",\"obstacle_detected\":"));
  espSerial.print(obstacleDetected ? F("true") : F("false"));

  espSerial.print(F(",\"fan_on\":"));
  espSerial.print(fanOn ? F("true") : F("false"));

  espSerial.print(F(",\"fan_mode\":\"ML\""));

  espSerial.print(F(",\"motor_requested\":"));
  espSerial.print(motorRequested ? F("true") : F("false"));

  espSerial.print(F(",\"motor_allowed\":"));
  espSerial.print(motorAllowed ? F("true") : F("false"));

  espSerial.println(F("}"));
}

// ============================================================
// USB Serial Monitor diagnostics
// ============================================================
void printDebugStatus() {
  Serial.print(F("Distances cm | "));

  for (byte i = 0; i < 4; i++) {
    Serial.print(sensorNames[i]);
    Serial.print(':');
    Serial.print(distanceCm[i], 1);

    if (i < 3) {
      Serial.print(F(" | "));
    }
  }

  Serial.print(F(" | Obstacle="));
  Serial.print(obstacleDetected ? F("YES") : F("NO"));

  Serial.print(F(" | Requested="));
  Serial.print(motorRequested ? F("ON") : F("OFF"));

  Serial.print(F(" | Allowed="));
  Serial.print(motorAllowed ? F("ON") : F("OFF"));

  Serial.print(F(" | Fan="));
  Serial.print(fanOn ? F("ON") : F("OFF"));

  Serial.print(F(" | Temperature="));

  if (isnan(temperatureC)) {
    Serial.print(F("N/A"));
  } else {
    Serial.print(temperatureC, 1);
  }

  Serial.println(F(" C"));
}

// ============================================================
// Setup
// ============================================================
void setup() {
  Serial.begin(115200);
  espSerial.begin(9600);

  dht.begin();

  for (byte i = 0; i < 4; i++) {
    pinMode(trigPins[i], OUTPUT);
    pinMode(echoPins[i], INPUT);

    digitalWrite(trigPins[i], LOW);
  }

  pinMode(FAN_RELAY_PIN, OUTPUT);
  pinMode(MOTOR_RELAY_PIN, OUTPUT);

  // Both relays start OFF.
  digitalWrite(FAN_RELAY_PIN, RELAY_OFF);
  digitalWrite(MOTOR_RELAY_PIN, RELAY_OFF);

  fanOn = false;
  motorRequested = false;
  motorAllowed = false;

  lastSensorRead = millis() - SENSOR_INTERVAL_MS;
  lastDHTRead = millis() - DHT_INTERVAL_MS;
  lastTelemetry = millis();
  lastDebug = millis();

  Serial.println(F("SARM Nano ready."));
  Serial.println(F("Fan control: AWS ML."));
  Serial.println(F("Motor commands: MOTOR:ON / MOTOR:OFF."));
  Serial.println(F("Ultrasonic maximum reported distance: 200 cm."));
  Serial.println(F("Motor stops when any reading is below 15 cm."));
  Serial.println(F("No echo is treated as 200 cm."));
}

// ============================================================
// Main loop
// ============================================================
void loop() {
  readESPCommands();

  unsigned long now = millis();

  if (now - lastSensorRead >= SENSOR_INTERVAL_MS) {
    lastSensorRead = now;

    readAllDistances();
    updateMotor();
  }

  if (now - lastDHTRead >= DHT_INTERVAL_MS) {
    lastDHTRead = now;
    readEnvironment();
  }

  if (now - lastTelemetry >= TELEMETRY_INTERVAL_MS) {
    lastTelemetry = now;
    publishTelemetry();
  }

  if (now - lastDebug >= DEBUG_INTERVAL_MS) {
    lastDebug = now;
    printDebugStatus();
  }
}