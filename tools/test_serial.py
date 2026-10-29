"""Hardware check: reads the DHT11 and clicks each relay once.

Usage:  python tools/test_serial.py          (auto-detects the ESP32)
        python tools/test_serial.py COM7
Close the Arduino Serial Monitor and the dashboard first (only one program can use the port).
"""
import sys
import time
from pathlib import Path

import serial

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.esp32 import build_relay_command, find_esp32_port, parse_line  # noqa: E402

port = sys.argv[1] if len(sys.argv) > 1 else find_esp32_port()
if not port:
    sys.exit("No ESP32 found. Plug it in (data cable), install the driver, or pass the port: python tools/test_serial.py COM7")
print(f"Using {port}")
ser = serial.Serial(port, 115200, timeout=1)
time.sleep(2)                                   # the ESP32 resets when the port opens

temps, relay_seen, bad_json = [], {}, 0


def read_for(seconds: float) -> None:
    global bad_json
    end = time.time() + seconds
    while time.time() < end:
        line = ser.readline().decode(errors="ignore").strip()
        if not line:
            continue
        msg = parse_line(line)
        print("ESP32 ->", line, "" if msg else "   (not JSON)")
        if not msg:
            bad_json += 1
        elif msg["type"] == "telemetry":
            relay_seen.update({ch: msg[f"relay{ch}"] for ch in range(1, 5)})
            if msg["sensor_ok"]:
                temps.append(msg["temp_c"])


print("\n-- 1. Telemetry / DHT11 (6 s) --")
read_for(6)
print(f"   DHT11: {'OK, ' + str(temps[-1]) + ' C' if temps else 'NO VALID READING - check DATA wire on GPIO 4, 3.3V and GND'}")

print("\n-- 2. Relays (listen for the clicks, watch the LEDs) --")
results = {}
for ch in (1, 2, 3, 4):
    print(f"\nRelay {ch} ON");  ser.write(build_relay_command(ch, 1).encode()); read_for(1.5)
    on_ok = relay_seen.get(ch) == 1
    print(f"Relay {ch} OFF"); ser.write(build_relay_command(ch, 0).encode()); read_for(1.5)
    results[ch] = on_ok and relay_seen.get(ch) == 0
ser.write(b'{"type":"all_off"}\n')
ser.close()

print("\n== Summary ==")
print(f"DHT11 temperature : {'PASS' if temps else 'FAIL'}")
for ch, ok in results.items():
    print(f"Relay {ch} command   : {'PASS (ESP32 confirmed the new state)' if ok else 'FAIL (no confirmation)'}")
print("If the relays clicked ON at startup / reversed, flip RELAY_ACTIVE_LOW in the firmware.")
