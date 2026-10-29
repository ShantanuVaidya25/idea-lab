/*
  Smart Classroom - ESP32 Controller
  ----------------------------------
  Four DHT11 sensors, one per classroom region (report section 3.2):
      Region 1 -> GPIO 4, Region 2 -> GPIO 16, Region 3 -> GPIO 21, Region 4 -> GPIO 22
  Fewer sensors are fine: an unconnected region reports null.
  4-channel relay on GPIO 17, 5, 18, 19.
  Talks to the PC over USB serial (115200) using JSON Lines.

  PC -> ESP32:  {"type":"relay","channel":1,"state":1}
                {"type":"all_off"}
                {"type":"ping"}
  ESP32 -> PC:  {"type":"telemetry","temps":[29.0,28.0,null,null],"hums":[55,54,null,null],
                 "temp_c":28.5,"humidity":55,"sensor_ok":true,"relay1":0,"relay2":0,"relay3":0,"relay4":0}
                (temp_c = average of the regions that answered, kept for older software)
                {"type":"ack","channel":1,"state":1}
                {"type":"error","msg":"..."}

  Library needed (Arduino IDE > Library Manager):
    "DHT sensor library" by Adafruit (+ "Adafruit Unified Sensor")
*/
#include <DHT.h>

// ---------------- CONFIG (change here if your wiring differs) ----------------
#define DHT_TYPE       DHT11
const int NUM_REGIONS = 4;
const int DHT_PINS[NUM_REGIONS] = {4, 16, 21, 22};      // Region 1..4
const int RELAY_PINS[4] = {17, 5, 18, 19};
// Most blue 4-channel relay modules are ACTIVE-LOW (LOW = relay ON).
// If your relays turn ON at startup, flip this value.
const bool RELAY_ACTIVE_LOW = true;
const unsigned long TELEMETRY_MS = 5000;   // sample every 5 s, as in the report (DHT11 max ~1 Hz)
// -----------------------------------------------------------------------------

DHT dht[NUM_REGIONS] = {DHT(DHT_PINS[0], DHT_TYPE), DHT(DHT_PINS[1], DHT_TYPE),
                        DHT(DHT_PINS[2], DHT_TYPE), DHT(DHT_PINS[3], DHT_TYPE)};
int relayState[4] = {0, 0, 0, 0};
unsigned long lastTelemetry = 0;
String rxBuffer = "";

void writeRelay(int idx, int state) {
  relayState[idx] = state ? 1 : 0;
  bool level = RELAY_ACTIVE_LOW ? !relayState[idx] : relayState[idx];
  digitalWrite(RELAY_PINS[idx], level ? HIGH : LOW);
}

void allOff() {
  for (int i = 0; i < 4; i++) writeRelay(i, 0);
}

// tiny JSON helpers (no extra library needed for our simple messages)
bool hasType(const String& s, const char* type) {
  String key = String("\"type\":\"") + type + "\"";
  String compact = s; compact.replace(" ", "");
  return compact.indexOf(key) >= 0;
}

long getInt(const String& s, const char* key, long fallback) {
  String compact = s; compact.replace(" ", "");
  String k = String("\"") + key + "\":";
  int p = compact.indexOf(k);
  if (p < 0) return fallback;
  return compact.substring(p + k.length()).toInt();
}

void handleLine(const String& line) {
  if (hasType(line, "relay")) {
    long ch = getInt(line, "channel", -1);
    long st = getInt(line, "state", -1);
    if (ch < 1 || ch > 4 || (st != 0 && st != 1)) {
      Serial.println("{\"type\":\"error\",\"msg\":\"bad relay command\"}");
      return;
    }
    writeRelay(ch - 1, st);
    Serial.printf("{\"type\":\"ack\",\"channel\":%ld,\"state\":%ld}\n", ch, st);
    lastTelemetry = 0;  // send fresh telemetry right away
  } else if (hasType(line, "all_off")) {
    allOff();
    Serial.println("{\"type\":\"ack\",\"all_off\":true}");
  } else if (hasType(line, "ping")) {
    Serial.println("{\"type\":\"pong\"}");
  } else {
    Serial.println("{\"type\":\"error\",\"msg\":\"unknown command\"}");
  }
}

void printValue(float v, int decimals) {
  if (isnan(v)) Serial.print("null"); else Serial.print(v, decimals);
}

void sendTelemetry() {
  float t[NUM_REGIONS], h[NUM_REGIONS];
  float sum = 0; int n = 0; float firstHum = NAN;
  for (int i = 0; i < NUM_REGIONS; i++) {
    t[i] = dht[i].readTemperature();
    h[i] = dht[i].readHumidity();
    if (!isnan(t[i])) { sum += t[i]; n++; }
    if (isnan(firstHum) && !isnan(h[i])) firstHum = h[i];
  }
  Serial.print("{\"type\":\"telemetry\",\"temps\":[");
  for (int i = 0; i < NUM_REGIONS; i++) { if (i) Serial.print(","); printValue(t[i], 1); }
  Serial.print("],\"hums\":[");
  for (int i = 0; i < NUM_REGIONS; i++) { if (i) Serial.print(","); printValue(h[i], 0); }
  Serial.print("],\"temp_c\":");
  printValue(n ? sum / n : NAN, 1);
  Serial.print(",\"humidity\":");
  printValue(firstHum, 0);
  Serial.printf(",\"sensor_ok\":%s,\"relay1\":%d,\"relay2\":%d,\"relay3\":%d,\"relay4\":%d}\n",
                n ? "true" : "false", relayState[0], relayState[1], relayState[2], relayState[3]);
}

void setup() {
  // SAFE STARTUP: set the OFF level before enabling outputs
  for (int i = 0; i < 4; i++) {
    digitalWrite(RELAY_PINS[i], RELAY_ACTIVE_LOW ? HIGH : LOW);
    pinMode(RELAY_PINS[i], OUTPUT);
  }
  allOff();
  Serial.begin(115200);
  for (int i = 0; i < NUM_REGIONS; i++) dht[i].begin();
  Serial.println("{\"type\":\"boot\",\"msg\":\"esp32 ready, relays OFF\"}");
}

void loop() {
  // non-blocking serial read
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n') {
      rxBuffer.trim();
      if (rxBuffer.length()) handleLine(rxBuffer);
      rxBuffer = "";
    } else if (rxBuffer.length() < 200) {
      rxBuffer += c;
    }
  }
  // periodic telemetry using millis() (no delay())
  unsigned long now = millis();
  if (lastTelemetry == 0 || now - lastTelemetry >= TELEMETRY_MS) {
    lastTelemetry = now == 0 ? 1 : now;
    sendTelemetry();
  }
}
