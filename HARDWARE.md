# Connecting the ESP32, DHT11 and 4-channel relay

Do the steps in order. Each step has a quick check so you know it worked before moving on.

> ⚠️ Use **LEDs / low-voltage demo loads only**. Do **not** connect 230 V mains.

## 1. Wiring

The report divides the room into 4 quadrants with one DHT11 in the middle of each. Fewer sensors work too: an
unconnected region shows "No sensor" and the mean ΔT uses the regions that are connected.


| Part | Pin | ESP32 |
|---|---|---|
| DHT11 Region 1 | DATA | **GPIO 4** |
| DHT11 Region 2 | DATA | **GPIO 16** |
| DHT11 Region 3 | DATA | **GPIO 21** |
| DHT11 Region 4 | DATA | **GPIO 22** |
| every DHT11 | VCC / GND | 3.3V / GND |
| Relay module | VCC | VIN (5V from USB) |
| | GND | GND |
| | IN1 (Light 1) | **GPIO 17** |
| | IN2 (Light 2) | **GPIO 5** |
| | IN3 (Fan) | **GPIO 18** |
| | IN4 (AC demo LED) | **GPIO 19** |

Each demo load: 5V → relay **COM**, relay **NO** → LED + 220 Ω resistor → GND.

## 2. Install the USB driver (only if Windows does not show a COM port)

Plug the ESP32 in with a **data** USB cable (some cables only charge). Look at the chip next to the USB socket:
- **CP2102 / CP210x** → "CP210x Universal Windows Driver" from Silicon Labs
- **CH340 / CH9102** → "CH341SER" driver from WCH

Check: `python tools/list_ports.py` shows a line ending with `<-- ESP32 (use this one)`.
(The Bluetooth "COM3 to COM6" ports are not your ESP32 and are ignored.)

## 3. Upload the firmware (Arduino IDE)

1. **File → Preferences → Additional boards manager URLs**, add:
   `https://espressif.github.io/arduino-esp32/package_esp32_index.json`
2. **Tools → Board → Boards Manager** → install **esp32 by Espressif Systems**.
3. **Tools → Manage Libraries** → install **DHT sensor library** (Adafruit) and **Adafruit Unified Sensor**.
4. Open `firmware/esp32_controller/esp32_controller.ino`.
5. **Tools → Board → ESP32 Dev Module**, **Tools → Port →** your ESP32's COM port, click **Upload**.
   If it says "Connecting......", hold the **BOOT** button on the board until the upload starts.
6. **Tools → Serial Monitor** at **115200**. You should see a line every 2 s:
   `{"type":"telemetry","temp_c":29.0,"sensor_ok":true,"relay1":0,...}`
7. **Close the Serial Monitor** (only one program can use the port).

## 4. Hardware self-test (no dashboard needed)

```powershell
python tools/test_serial.py
```

It auto-detects the ESP32, reads the DHT11 for 6 s, clicks each relay on/off and prints a **PASS / FAIL summary**:

```
DHT11 temperature : PASS
Relay 1 command   : PASS (ESP32 confirmed the new state)
...
```

| Result | Fix |
|---|---|
| `DHT11: NO VALID READING` | DATA wire to GPIO 4, VCC to 3.3V, GND; try another DHT11 |
| Relays click ON when the board starts / ON and OFF reversed | set `RELAY_ACTIVE_LOW = false` in the firmware and re-upload |
| `No ESP32 found` | step 2 (driver / data cable); close Serial Monitor; try another USB port |
| A relay does not click | check IN wire to the right GPIO, relay VCC on VIN (5V) |

## 5. Use it in the dashboard

1. Start the app (`run_demo.bat`) and open http://localhost:8501.
2. Sidebar → **🔌 ESP32 → Port → 🔎 Auto-detect ESP32 → Connect ESP32**.
   (or pick the COM port with ✅ next to it). **🔄 Rescan** finds a board you plugged in later.
3. The top row shows **ESP32: Connected COMx** and the real temperature (no "SIMULATED" note).
4. **Appliances** tab: the buttons switch the real relays. Check that Light 1 / Light 2 / Fan / AC click and the LEDs follow.
5. To start connected every time, set in `config.yaml`:
   ```yaml
   esp32:
     port: "auto"      # or "COM7"
   ```

## 6. Demo checklist

- [ ] `tools/test_serial.py` all PASS
- [ ] Dashboard shows **Connected COMx**, temperature updates every ~2 s
- [ ] Relay buttons switch the LEDs (turns **Manual override** on, so automation stays out of the way)
- [ ] Turn **Manual override OFF** and **Automation ON**; switch Light 1/2 and Fan ON
- [ ] Walk out of the camera view → after the vacancy time the lights/fan relays switch **OFF** by themselves
- [ ] Camera sees a fan/light ON and the room is empty → **Turn OFF** popup after ~8 s
- [ ] AC LED ON, room empty → temperature drops → **Turn OFF the AC** alert (the AC relay is never switched automatically)

Tip for the demo: in the dashboard Settings box set *Vacancy confirmation* to 20 s and *AC observation* to 30 s (restart to apply).

## Safety behaviour (already built in)

- All relays are **OFF** when the ESP32 boots.
- If the camera or YOLO fails, **nothing is switched** (a failure is never treated as an empty room).
- Any relay button turns **Manual override** on, and automation stops changing relays.
- At least 10 s between switches of the same relay.
- If a relay is commanded but the ESP32 reports a different state, it is logged as **RELAY_FAULT** (see the Logs page).

## Alternative: upload from VS Code (PlatformIO)

1. Install **VS Code** and its **PlatformIO IDE** extension (Extensions panel → search "PlatformIO IDE" → Install). Restart VS Code when it asks.
2. **File → Open Folder →** the project's **`firmware`** folder (the one containing `platformio.ini`, not the whole project).
   PlatformIO downloads the ESP32 support and the DHT libraries by itself the first time (a few minutes, needs internet).
3. Plug the ESP32 in (data cable, driver installed - see step 2 above).
4. Use the buttons in the blue bar at the bottom of VS Code:
   - ✔ **Build** - checks that the firmware compiles
   - → **Upload** - flashes the ESP32 (hold the **BOOT** button if it says "Connecting......")
   - 🔌 **Serial Monitor** - shows the JSON lines (close it before starting the dashboard)
5. Or in the VS Code terminal, from the `firmware` folder:
   ```
   pio run                      # build
   pio run -t upload            # upload
   pio device monitor           # serial monitor (Ctrl+C to quit)
   ```
   If it picks the wrong port, add `upload_port = COM7` under `[env:esp32dev]` in `platformio.ini`.
