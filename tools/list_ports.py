"""List serial ports and show which one is your ESP32 (Bluetooth 'COM' ports are ignored by default)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.esp32 import list_serial_ports  # noqa: E402

ports = list_serial_ports()
if not any(p["likely_esp32"] for p in ports):
    print("No ESP32 USB port found. Plug in the ESP32 with a DATA USB cable and install the CP210x / CH340 driver.")
for p in ports:
    tag = "  <-- ESP32 (use this one)" if p["likely_esp32"] else ("  (Bluetooth, ignore)" if p["bluetooth"] else "")
    print(f"{p['device']:8s} {p['description']}{tag}")
