/*
  Smart Security System - ESP32 Side
  ------------------------------------
  কাজ করে:
    - Laptop (Python/OpenCV) থেকে WiFi HTTP কমান্ড রিসিভ করা (door open / unknown alert)
    - PIR Motion Sensor দিয়ে মোশন ডিটেকশন
    - DHT11 দিয়ে তাপমাত্রা ও আর্দ্রতা
    - Flame Sensor দিয়ে আগুন শনাক্তকরণ
    - LDR দিয়ে আলো/অন্ধকার বোঝা এবং Relay দিয়ে অটো লাইট কন্ট্রোল
    - Servo Motor দিয়ে ডোর লক/আনলক
    - Buzzer দিয়ে অ্যালার্ম
    - Blynk অ্যাপে লাইভ ডেটা পাঠানো
    - Firebase Realtime Database-এ visitor/event log রাখা

  প্রয়োজনীয় Arduino Library (Library Manager থেকে ইনস্টল করো):
    - DHT sensor library (Adafruit)
    - ESP32Servo
    - Blynk (Blynk by Volodymyr Shymanskyy)
    - Firebase ESP32 Client (Mobizt)
*/

#include <WiFi.h>
#include <WebServer.h>
#include <DHT.h>
#include <ESP32Servo.h>

#define BLYNK_TEMPLATE_ID   "YOUR_TEMPLATE_ID"
#define BLYNK_TEMPLATE_NAME "Smart Security"
#define BLYNK_AUTH_TOKEN    "YOUR_BLYNK_AUTH_TOKEN"
#include <BlynkSimpleEsp32.h>

#include <Firebase_ESP_Client.h>
#include "addons/TokenHelper.h"
#include "addons/RTDBHelper.h"

// ---------------------- WiFi & Cloud CONFIG ----------------------
char wifi_ssid[] = "YOUR_WIFI_NAME";
char wifi_pass[] = "YOUR_WIFI_PASSWORD";
char blynk_auth[] = BLYNK_AUTH_TOKEN;

#define FIREBASE_HOST "YOUR_PROJECT_ID-default-rtdb.firebaseio.com"
#define FIREBASE_API_KEY "YOUR_FIREBASE_API_KEY"
#define FIREBASE_USER_EMAIL "YOUR_EMAIL"
#define FIREBASE_USER_PASSWORD "YOUR_PASSWORD"

FirebaseData fbdo;
FirebaseAuth fbAuth;
FirebaseConfig fbConfig;
bool firebaseReady = false;

// ---------------------- PIN CONFIG ----------------------
#define PIR_PIN     13
#define DHT_PIN     14
#define DHT_TYPE    DHT11
#define FLAME_PIN   27
#define LDR_PIN     34      // Analog pin
#define RELAY_PIN   26
#define BUZZER_PIN  25
#define SERVO_PIN   33

DHT dht(DHT_PIN, DHT_TYPE);
Servo doorServo;
WebServer server(80);

// ---------------------- STATE ----------------------
const int LDR_DARK_THRESHOLD = 2000;   // এর নিচে হলে অন্ধকার ধরা হবে (0-4095)
const int DOOR_LOCKED_ANGLE = 0;
const int DOOR_OPEN_ANGLE   = 90;

unsigned long lastSensorRead = 0;
const unsigned long SENSOR_INTERVAL = 2000;

unsigned long doorOpenedAt = 0;
bool doorIsOpen = false;
const unsigned long DOOR_AUTO_CLOSE_MS = 5000;

// ---------------------- SETUP ----------------------
void setup() {
  Serial.begin(115200);

  pinMode(PIR_PIN, INPUT);
  pinMode(FLAME_PIN, INPUT);
  pinMode(RELAY_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW);
  digitalWrite(BUZZER_PIN, LOW);

  dht.begin();
  doorServo.attach(SERVO_PIN);
  doorServo.write(DOOR_LOCKED_ANGLE);

  connectWiFi();
  setupFirebase();

  // Blynk শুরু করা (non-blocking connect)
  Blynk.config(blynk_auth);
  Blynk.connect();

  // HTTP endpoints - Laptop থেকে কমান্ড রিসিভ করার জন্য
  server.on("/door_open", HTTP_GET, handleDoorOpen);
  server.on("/unknown_alert", HTTP_GET, handleUnknownAlert);
  server.on("/status", HTTP_GET, handleStatus);
  server.begin();

  Serial.println("[i] ESP32 সার্ভার চালু হয়েছে।");
  Serial.print("[i] এই IP address টা Python কোডে বসাও: ");
  Serial.println(WiFi.localIP());
}

// ---------------------- LOOP ----------------------
void loop() {
  server.handleClient();
  Blynk.run();

  if (millis() - lastSensorRead > SENSOR_INTERVAL) {
    lastSensorRead = millis();
    readSensorsAndAct();
  }

  // Door auto-close after few seconds
  if (doorIsOpen && millis() - doorOpenedAt > DOOR_AUTO_CLOSE_MS) {
    doorServo.write(DOOR_LOCKED_ANGLE);
    doorIsOpen = false;
    Serial.println("[i] দরজা auto-close হলো।");
  }
}

// ---------------------- WiFi ----------------------
void connectWiFi() {
  WiFi.begin(wifi_ssid, wifi_pass);
  Serial.print("[i] WiFi সংযোগ করা হচ্ছে");
  while (WiFi.status() != WL_CONNECTED) {
    delay(400);
    Serial.print(".");
  }
  Serial.println("\n[+] WiFi সংযুক্ত হয়েছে।");
}

// ---------------------- Firebase ----------------------
void setupFirebase() {
  fbConfig.api_key = FIREBASE_API_KEY;
  fbConfig.database_url = FIREBASE_HOST;
  fbAuth.user.email = FIREBASE_USER_EMAIL;
  fbAuth.user.password = FIREBASE_USER_PASSWORD;
  fbConfig.token_status_callback = tokenStatusCallback;

  Firebase.begin(&fbConfig, &fbAuth);
  Firebase.reconnectWiFi(true);
  firebaseReady = true;
}

void logToFirebase(const String &eventType, const String &details) {
  if (!firebaseReady) return;
  String path = "/logs/" + String(millis());
  FirebaseJson json;
  json.set("event", eventType);
  json.set("details", details);
  json.set("timestamp", millis());
  Firebase.RTDB.setJSON(&fbdo, path.c_str(), &json);
}

// ---------------------- HTTP Handlers (Laptop -> ESP32) ----------------------
void handleDoorOpen() {
  Serial.println("[>] Laptop থেকে কমান্ড: চেনা মুখ, দরজা খোলো।");
  doorServo.write(DOOR_OPEN_ANGLE);
  doorIsOpen = true;
  doorOpenedAt = millis();

  digitalWrite(BUZZER_PIN, HIGH);
  delay(150);
  digitalWrite(BUZZER_PIN, LOW);

  logToFirebase("known_face", "Door opened for recognized person");
  Blynk.virtualWrite(V0, "Door Opened - Known Person");

  server.send(200, "text/plain", "door opened");
}

void handleUnknownAlert() {
  Serial.println("[>] Laptop থেকে কমান্ড: অচেনা মুখ শনাক্ত হয়েছে!");

  digitalWrite(BUZZER_PIN, HIGH);
  delay(400);
  digitalWrite(BUZZER_PIN, LOW);
  delay(150);
  digitalWrite(BUZZER_PIN, HIGH);
  delay(400);
  digitalWrite(BUZZER_PIN, LOW);

  logToFirebase("unknown_face", "Unknown person detected at door");
  Blynk.virtualWrite(V0, "ALERT - Unknown Person!");
  Blynk.logEvent("unknown_person");

  server.send(200, "text/plain", "alert received");
}

void handleStatus() {
  String msg = "ESP32 is online. Door: " + String(doorIsOpen ? "open" : "locked");
  server.send(200, "text/plain", msg);
}

// ---------------------- Sensors ----------------------
void readSensorsAndAct() {
  // PIR Motion
  int motion = digitalRead(PIR_PIN);
  if (motion == HIGH) {
    Serial.println("[i] মোশন ডিটেক্ট হয়েছে।");
    Blynk.virtualWrite(V1, 1);
  } else {
    Blynk.virtualWrite(V1, 0);
  }

  // DHT11
  float temp = dht.readTemperature();
  float hum = dht.readHumidity();
  if (!isnan(temp) && !isnan(hum)) {
    Serial.printf("[i] Temp: %.1f C, Humidity: %.1f %%\n", temp, hum);
    Blynk.virtualWrite(V2, temp);
    Blynk.virtualWrite(V3, hum);
  }

  // Flame Sensor (সাধারণত LOW = আগুন শনাক্ত, মডিউল অনুযায়ী পাল্টাতে পারে)
  int flame = digitalRead(FLAME_PIN);
  if (flame == LOW) {
    Serial.println("[!] আগুন শনাক্ত হয়েছে! Alarm চালু।");
    digitalWrite(BUZZER_PIN, HIGH);
    logToFirebase("fire_alert", "Flame sensor triggered");
    Blynk.logEvent("fire_alert");
  } else {
    digitalWrite(BUZZER_PIN, LOW);
  }

  // LDR -> Auto Light via Relay
  int ldrValue = analogRead(LDR_PIN);
  Blynk.virtualWrite(V4, ldrValue);
  if (ldrValue < LDR_DARK_THRESHOLD) {
    digitalWrite(RELAY_PIN, HIGH);   // অন্ধকার -> লাইট অন
  } else {
    digitalWrite(RELAY_PIN, LOW);    // আলো আছে -> লাইট অফ
  }
}

// ---------------------- Blynk: App থেকে ম্যানুয়াল কন্ট্রোল ----------------------
// Blynk App-এ Virtual Pin V5 - একটা Button দিয়ে ম্যানুয়ালি লাইট অন/অফ করা যাবে
BLYNK_WRITE(V5) {
  int value = param.asInt();
  digitalWrite(RELAY_PIN, value ? HIGH : LOW);
}

// Virtual Pin V6 - ম্যানুয়ালি দরজা খোলা/বন্ধ করা
BLYNK_WRITE(V6) {
  int value = param.asInt();
  doorServo.write(value ? DOOR_OPEN_ANGLE : DOOR_LOCKED_ANGLE);
  doorIsOpen = value;
  if (value) doorOpenedAt = millis();
}
