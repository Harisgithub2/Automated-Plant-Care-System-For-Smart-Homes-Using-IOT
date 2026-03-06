#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClient.h>
#include <ArduinoJson.h>
#include <DHT.h>

/* ================= WIFI ================= */
const char* ssid     = "Airtel_?";
const char* password = "Sairam@1";

/* ================= FLASK SERVER ================= */
const char* POST_DATA_URL   = "http://192.168.1.8:5000/data";
const char* GET_PUMP_URL    = "http://192.168.1.8:5000/api/pump/status";

/* ================= PINS ================= */
#define DHTPIN    D4
#define DHTTYPE   DHT11

#define SOIL1_PIN D5
#define SOIL2_PIN D6
#define WATER_PIN A0

#define RELAY1_PIN D2
#define RELAY2_PIN D1

/* ================= RELAY ================= */
#define RELAY_ON  LOW
#define RELAY_OFF HIGH
#define WATER_THRESHOLD 250

/* ================= OBJECTS ================= */
DHT dht(DHTPIN, DHTTYPE);

/* ================= STATE ================= */
bool pump1ON = false;
bool pump2ON = false;
bool manualOverride = false;
bool serverConnected = false;

/* ========================================================= */
/* ================= WIFI CONNECTION ======================== */
/* ========================================================= */

void connectWiFi() {

  if (WiFi.status() == WL_CONNECTED) return;

  Serial.println("Connecting to WiFi...");
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  unsigned long start = millis();

  while (WiFi.status() != WL_CONNECTED && millis() - start < 15000) {
    delay(500);
    Serial.print(".");
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi Connected ✅");
    Serial.print("ESP IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\nWiFi Connection Failed ❌");
  }
}

/* ========================================================= */
/* ================= CHECK SERVER =========================== */
/* ========================================================= */

bool checkServerConnection() {

  if (WiFi.status() != WL_CONNECTED) return false;

  WiFiClient client;

  if (client.connect("192.168.1.8", 5000)) {
    client.stop();
    return true;
  } else {
    return false;
  }
}

/* ========================================================= */
/* ================= GET PUMP COMMAND ======================= */
/* ========================================================= */

void getPumpCommand() {

  if (WiFi.status() != WL_CONNECTED) return;

  WiFiClient client;
  HTTPClient http;

  http.setTimeout(5000);
  http.begin(client, GET_PUMP_URL);

  int httpCode = http.GET();

  if (httpCode == HTTP_CODE_OK) {

    serverConnected = true;

    String payload = http.getString();

    StaticJsonDocument<256> doc;
    DeserializationError error = deserializeJson(doc, payload);

    if (!error) {
      manualOverride = doc["manual_override"];

      if (manualOverride) {
        pump1ON = (String(doc["pump1_state"]) == "ON");
        pump2ON = (String(doc["pump2_state"]) == "ON");
      }
    }

  } else {
    serverConnected = false;
  }

  http.end();
}

/* ========================================================= */
/* ================= SEND SENSOR DATA ======================= */
/* ========================================================= */

void sendData(float temp, float hum, bool soil1Dry, bool soil2Dry, bool waterAvailable) {

  if (WiFi.status() != WL_CONNECTED) return;

  WiFiClient client;
  HTTPClient http;

  http.setTimeout(5000);
  http.begin(client, POST_DATA_URL);
  http.addHeader("Content-Type", "application/json");

  StaticJsonDocument<256> doc;

  doc["temperature"] = temp;
  doc["humidity"] = hum;
  doc["soil1"] = soil1Dry ? "DRY" : "WET";
  doc["soil2"] = soil2Dry ? "DRY" : "WET";
  doc["water"] = waterAvailable ? "AVAILABLE" : "NOT AVAILABLE";
  doc["pump1"] = pump1ON ? "ON" : "OFF";
  doc["pump2"] = pump2ON ? "ON" : "OFF";

  String json;
  serializeJson(doc, json);

  int httpCode = http.POST(json);

  if (httpCode == HTTP_CODE_OK) {
    serverConnected = true;
  } else {
    serverConnected = false;
  }

  http.end();
}

/* ========================================================= */
/* ================= SETUP ================================= */
/* ========================================================= */

void setup() {

  Serial.begin(9600);
  delay(1000);

  pinMode(SOIL1_PIN, INPUT);
  pinMode(SOIL2_PIN, INPUT);
  pinMode(RELAY1_PIN, OUTPUT);
  pinMode(RELAY2_PIN, OUTPUT);

  digitalWrite(RELAY1_PIN, RELAY_OFF);
  digitalWrite(RELAY2_PIN, RELAY_OFF);

  dht.begin();
  connectWiFi();

  Serial.println("\nSMART IRRIGATION SYSTEM READY 🌱");
}

/* ========================================================= */
/* ================= LOOP ================================== */
/* ========================================================= */

void loop() {

  connectWiFi();

  serverConnected = checkServerConnection();
  getPumpCommand();

  /* -------- READ SENSORS -------- */

  float temperature = dht.readTemperature();
  float humidity    = dht.readHumidity();

  if (isnan(temperature) || isnan(humidity)) {
    Serial.println("DHT Sensor Error ❌");
    delay(2000);
    return;
  }

  int soil1State = digitalRead(SOIL1_PIN);
  int soil2State = digitalRead(SOIL2_PIN);
  int waterValue = analogRead(WATER_PIN);

  bool soil1Dry = (soil1State == HIGH);
  bool soil2Dry = (soil2State == HIGH);
  bool waterAvailable = (waterValue >= WATER_THRESHOLD);

  /* -------- AUTO MODE -------- */

  if (!manualOverride) {
    pump1ON = (soil1Dry && waterAvailable);
    pump2ON = (soil2Dry && waterAvailable);
  }

  /* -------- APPLY RELAYS -------- */

  digitalWrite(RELAY1_PIN, pump1ON ? RELAY_ON : RELAY_OFF);
  digitalWrite(RELAY2_PIN, pump2ON ? RELAY_ON : RELAY_OFF);

  sendData(temperature, humidity, soil1Dry, soil2Dry, waterAvailable);

  /* ================= OUTPUT ================= */

  Serial.println("\n=================================");
  Serial.println(" SMART IRRIGATION SYSTEM STATUS ");
  Serial.println("=================================");

  Serial.print("WiFi Status   : ");
  Serial.println(WiFi.status() == WL_CONNECTED ? "CONNECTED" : "NOT CONNECTED");

  Serial.print("Server Status : ");
  Serial.println(serverConnected ? "CONNECTED" : "NOT REACHABLE");

  Serial.print("Mode          : ");
  Serial.println(manualOverride ? "MANUAL" : "AUTO");

  Serial.print("Temperature   : ");
  Serial.print(temperature);
  Serial.println(" °C");

  Serial.print("Humidity      : ");
  Serial.print(humidity);
  Serial.println(" %");

  Serial.print("Water Value   : ");
  Serial.println(waterValue);

  Serial.print("Soil1         : ");
  Serial.println(soil1Dry ? "DRY" : "WET");

  Serial.print("Soil2         : ");
  Serial.println(soil2Dry ? "DRY" : "WET");

  Serial.print("Pump1         : ");
  Serial.println(pump1ON ? "ON" : "OFF");

  Serial.print("Pump2         : ");
  Serial.println(pump2ON ? "ON" : "OFF");

  Serial.println("=================================\n");

  delay(5000);
}
