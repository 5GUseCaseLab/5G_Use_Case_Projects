/*
  PROJECT: 5G-Based Flash Flood Early Warning System
  CONTROLLER: Arduino UNO Q

  Sensors:
  Water Level   -> A1
  Soil Moisture -> A2
  Rain Sensor   -> A0

  Communication:
  Arduino MCU -> UNO Q Bridge -> Python UDP sender -> server
*/

#include <Arduino_RouterBridge.h>

const int WATER_PIN = A1;
const int SOIL_PIN  = A2;
const int RAIN_PIN  = A0;

// Sampling configuration
const int SAMPLE_COUNT = 5;
const unsigned long SAMPLE_INTERVAL = 200;

// Calibration values (adjust after testing your sensors)
const int WATER_LOW  = 0;
const int WATER_HIGH = 1023;

const int SOIL_DRY = 800;
const int SOIL_WET = 350;

const int RAIN_DRY = 850;
const int RAIN_WET = 350;

// Flood alert thresholds (%)
const float WATER_WARNING        = 55.0;
const float WATER_CRITICAL       = 80.0;
const float SOIL_SATURATED_LIMIT = 70.0;  // High moisture = saturated soil (runoff risk)
const float RAIN_LIMIT           = 35.0;

// Event confirmation & hysteresis
const int CONFIRM_SAMPLES = 3;
const float HYSTERESIS    = 5.0;

unsigned long lastSampleTime = 0;
unsigned long sequenceNumber = 0;

uint32_t clockMillis() {
  return millis();
}

int waterWarnCount = 0;
int waterCritCount = 0;
int soilSatCount   = 0;
int rainStateCount = 0;

bool waterWarningActive  = false;
bool waterCriticalActive = false;
bool soilSaturatedActive = false;
bool rainActive          = false;


// Read sensor using averaging filter
int readAverage(int pin) {
  long total = 0;
  for (int i = 0; i < SAMPLE_COUNT; i++) {
    total += analogRead(pin);
    delay(2);
  }
  return total / SAMPLE_COUNT;
}


// Convert raw sensor reading into 0-100 percentage
float normalizeValue(int raw, int lowValue, int highValue) {
  if (lowValue == highValue) {
    return 0;
  }
  float percentage = ((float)(raw - lowValue) / (highValue - lowValue)) * 100.0;
  return constrain(percentage, 0.0, 100.0);
}


// Confirm event for consecutive samples and clear when condition stops
bool confirmEvent(bool condition, int &counter, bool currentState) {
  if (condition) {
    counter++;
    if (counter >= CONFIRM_SAMPLES) {
      return true;
    }
    return currentState;
  } else {
    counter = 0;
    return false;
  }
}


// Sensor event detection with hysteresis
void updateSensorStates(float water, float soil, float rain) {
  bool warningCondition = waterWarningActive
      ? (water > (WATER_WARNING - HYSTERESIS))
      : (water >= WATER_WARNING);
  waterWarningActive = confirmEvent(warningCondition, waterWarnCount, waterWarningActive);

  bool criticalCondition = waterCriticalActive
      ? (water > (WATER_CRITICAL - HYSTERESIS))
      : (water >= WATER_CRITICAL);
  waterCriticalActive = confirmEvent(criticalCondition, waterCritCount, waterCriticalActive);

  bool soilCondition = soilSaturatedActive
      ? (soil > (SOIL_SATURATED_LIMIT - HYSTERESIS))
      : (soil >= SOIL_SATURATED_LIMIT);
  soilSaturatedActive = confirmEvent(soilCondition, soilSatCount, soilSaturatedActive);

  bool rainCondition = rainActive
      ? (rain > (RAIN_LIMIT - HYSTERESIS))
      : (rain >= RAIN_LIMIT);
  rainActive = confirmEvent(rainCondition, rainStateCount, rainActive);
}


// Determine priority and flood-risk event: SAFE, WARNING, or CRITICAL
String getEvent(int &priority) {
  // CRITICAL: Water reaches critical level, OR all three flood factors are active together
  if (waterCriticalActive || (waterWarningActive && rainActive && soilSaturatedActive)) {
    priority = 0;
    return "CRITICAL";
  }

  // WARNING: Any flood precursor (elevated water, active rain, or saturated soil)
  if (waterWarningActive || rainActive || soilSaturatedActive) {
    priority = 1;
    return "WARNING";
  }

  // SAFE: All sensors below flood thresholds
  priority = 3;
  return "SAFE";
}


void setup() {
  Serial.begin(115200);

  Bridge.begin();
  Bridge.provide("clock_millis", clockMillis);

  pinMode(WATER_PIN, INPUT);
  pinMode(SOIL_PIN, INPUT);
  pinMode(RAIN_PIN, INPUT);

  delay(1000);
  Serial.println("SYSTEM_STARTED");
}


void loop() {
  unsigned long currentTime = millis();

  if (currentTime - lastSampleTime >= SAMPLE_INTERVAL) {
    lastSampleTime = currentTime;

    int waterRaw = readAverage(WATER_PIN);
    int soilRaw  = readAverage(SOIL_PIN);
    int rainRaw  = readAverage(RAIN_PIN);

    float waterPercent = normalizeValue(waterRaw, WATER_LOW, WATER_HIGH);
    float soilPercent  = normalizeValue(soilRaw, SOIL_DRY, SOIL_WET);
    float rainPercent  = normalizeValue(rainRaw, RAIN_DRY, RAIN_WET);

    updateSensorStates(waterPercent, soilPercent, rainPercent);

    int priority = 3;
    String event = getEvent(priority);

    sequenceNumber++;

    // Keep arguments compact: Bridge messages have a 256-byte limit.
    Bridge.notify("sensor_sample",
                  (uint32_t)sequenceNumber, (uint32_t)currentTime,
                  waterRaw, soilRaw, rainRaw,
                  waterPercent, soilPercent, rainPercent,
                  priority, event);

    Serial.print("{\"seq\":");
    Serial.print(sequenceNumber);
    Serial.print(",\"water_raw\":");
    Serial.print(waterRaw);
    Serial.print(",\"soil_raw\":");
    Serial.print(soilRaw);
    Serial.print(",\"rain_raw\":");
    Serial.print(rainRaw);
    Serial.print(",\"water\":");
    Serial.print(waterPercent, 1);
    Serial.print(",\"soil\":");
    Serial.print(soilPercent, 1);
    Serial.print(",\"rain\":");
    Serial.print(rainPercent, 1);
    Serial.print(",\"priority\":");
    Serial.print(priority);
    Serial.print(",\"event\":\"");
    Serial.print(event);
    Serial.println("\"}");
  }
}
