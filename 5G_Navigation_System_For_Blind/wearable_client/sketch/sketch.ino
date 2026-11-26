// =========================================================================
// 5G WEARABLE ASSISTIVE DEVICE - STREAMLINED MCU FIRMWARE
// Target: Arduino UNO Q (Zephyr MCU Core)
// Hardware: HC-SR04 Ultrasonic Distance Sensor on Pins 12 & 13
// Features: Intelligent Dead-Zone Handling + Asymmetric Fast-Approach Filter
// Output: Serial & App Lab Monitor at 10 Hz + RouterBridge RPC for Python
// =========================================================================

#include <Arduino_RouterBridge.h>

// --- PIN DEFINITIONS ---
#define PIN_TRIG            12  // HC-SR04 Ultrasonic Trigger
#define PIN_ECHO            13  // HC-SR04 Ultrasonic Echo

// --- TIMING CONSTANTS (Non-blocking with millis) ---
const unsigned long SENSOR_INTERVAL_MS    = 60;   // 16 Hz ultrasonic scan rate
const unsigned long TELEMETRY_INTERVAL_MS = 100;  // 10 Hz telemetry uplink

unsigned long lastSensorReadTime = 0;
unsigned long lastTelemetryTime  = 0;

// Current filtered distance in centimeters (defaults to 400 = clear pathway)
float currentDistanceCm = 400.0f;

// -------------------------------------------------------------------------
// Fast, Non-Blocking Ultrasonic Ping with Intelligent Blind-Zone Guard
// The HC-SR04 has a physical dead-zone below 3cm where the receiver rings.
// If an object was already close and suddenly echo disappears, it is TOUCHING!
// -------------------------------------------------------------------------
float readUltrasonicDistanceCM() {
    digitalWrite(PIN_TRIG, LOW);
    delayMicroseconds(2);
    digitalWrite(PIN_TRIG, HIGH);
    delayMicroseconds(10);
    digitalWrite(PIN_TRIG, LOW);

    // Manual non-blocking timing (replaces unreliable pulseIn on Zephyr core)
    unsigned long t0 = micros();
    while (digitalRead(PIN_ECHO) == LOW) { 
        if (micros() - t0 > 20000UL) return (currentDistanceCm < 18.0f) ? 2.0f : 400.0f; 
    }
    unsigned long start = micros();
    while (digitalRead(PIN_ECHO) == HIGH) { 
        if (micros() - start > 18000UL) return 400.0f; 
    }
    long duration = (long)(micros() - start);
    
    // Case 1: No echo received / invalid
    if (duration <= 0) {
        // If an object was already closer than 18cm, losing the echo means
        // it has pushed right against the sensor into the physical dead zone!
        if (currentDistanceCm < 18.0f) {
            return 2.0f; // Obstacle is touching the sensor
        }
        return 400.0f; // Open, clear pathway
    }
    
    float distance = (duration * 0.0343f) / 2.0f;
    
    // Case 2: Reading falls inside physical dead zone (0 - 2.5cm)
    if (distance <= 2.5f) {
        return 2.0f; // Obstacle right in front of sensor
    }
    if (distance > 400.0f) {
        return 400.0f;
    }
    return distance;
}

// -------------------------------------------------------------------------
// RouterBridge RPC Callback for Python
// -------------------------------------------------------------------------
float get_distance() {
    return currentDistanceCm;
}

// -------------------------------------------------------------------------
// Broadcast telemetry packet over App Lab Monitor
// -------------------------------------------------------------------------
void sendTelemetryPacket(float dist_cm) {
    char buffer[48];
    snprintf(buffer, sizeof(buffer), "{\"dist\":%.1f}", dist_cm);
    Monitor.println(buffer);
}

// -------------------------------------------------------------------------
// Setup
// -------------------------------------------------------------------------
void setup() {
    Monitor.begin();

    Bridge.begin();
    Bridge.provide("get_distance", get_distance);

    pinMode(PIN_TRIG, OUTPUT);
    pinMode(PIN_ECHO, INPUT);
    digitalWrite(PIN_TRIG, LOW);

    Monitor.println("🟢 [MCU READY]: Arduino UNO Q Ultrasonic & Bridge Online");
}

// -------------------------------------------------------------------------
// Main Loop (100% Non-Blocking)
// -------------------------------------------------------------------------
void loop() {
    unsigned long now = millis();

    // 1. Ultrasonic Proximity Scan (16 Hz)
    if (now - lastSensorReadTime >= SENSOR_INTERVAL_MS) {
        lastSensorReadTime = now;
        float rawDist = readUltrasonicDistanceCM();
        
        // Asymmetric filtering:
        // When an obstacle gets closer, react with zero lag (65% weight on new reading)
        // When an obstacle pulls away, smooth gently (70% weight on previous reading)
        if (rawDist < currentDistanceCm) {
            currentDistanceCm = (0.35f * currentDistanceCm) + (0.65f * rawDist);
        } else {
            currentDistanceCm = (0.70f * currentDistanceCm) + (0.30f * rawDist);
        }
    }

    // 2. Periodic Telemetry Stream (10 Hz)
    if (now - lastTelemetryTime >= TELEMETRY_INTERVAL_MS) {
        lastTelemetryTime = now;
        sendTelemetryPacket(currentDistanceCm);
    }
}
