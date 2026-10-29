"""ESP32 communication over USB serial (JSON Lines) + a simulator."""
from __future__ import annotations

import json
import logging
import random
import threading
import time
from typing import Any

log = logging.getLogger("esp32")


# ----------------------------------------------------------- protocol helpers
def parse_line(line: str) -> dict[str, Any] | None:
    """Parse one JSON line from the ESP32. Returns None for junk/invalid lines."""
    line = line.strip()
    if not line.startswith("{"):
        return None
    try:
        msg = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(msg, dict) or "type" not in msg:
        return None
    if msg["type"] == "telemetry":
        valid = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and -20 <= v <= 80
        # one temperature per classroom region (up to 4); older firmware sends only temp_c
        raw = msg.get("temps")
        if isinstance(raw, list) and raw:
            temps = [float(v) if valid(v) else None for v in raw[:4]]
        else:
            temps = [float(msg["temp_c"]) if valid(msg.get("temp_c")) and msg.get("sensor_ok", True) else None]
        msg["temps"] = temps
        good = [v for v in temps if v is not None]
        t = msg.get("temp_c")
        ok = bool(msg.get("sensor_ok", False)) and bool(good)
        if not valid(t):
            t = round(sum(good) / len(good), 2) if good else None
        if not ok:
            t = None              # never trust an impossible reading
        msg["temp_c"] = t
        msg["sensor_ok"] = ok
        h = msg.get("humidity")
        msg["humidity"] = h if isinstance(h, (int, float)) and 0 <= h <= 100 else None
        for ch in range(1, 5):
            msg[f"relay{ch}"] = 1 if msg.get(f"relay{ch}") else 0
    return msg


def build_relay_command(channel: int, state: int) -> str:
    if channel not in (1, 2, 3, 4):
        raise ValueError(f"channel must be 1-4, got {channel}")
    if state not in (0, 1):
        raise ValueError(f"state must be 0 or 1, got {state}")
    return json.dumps({"type": "relay", "channel": channel, "state": state}) + "\n"


# ---------------------------------------------------------------- port discovery
# USB-serial chips found on ESP32 dev boards: Silicon Labs CP210x, WCH CH340/CH9102, FTDI, Espressif native USB, Prolific
ESP32_USB_VIDS = {0x10C4, 0x1A86, 0x0403, 0x303A, 0x067B}
ESP32_KEYWORDS = ("cp210", "ch340", "ch910", "usb-serial", "usb serial", "silicon labs", "esp32", "ftdi")


def classify_port(device: str, description: str, hwid: str, vid: int | None) -> dict[str, Any]:
    """Describe one serial port. Bluetooth 'COM' ports are never an ESP32 USB cable."""
    bluetooth = hwid.upper().startswith("BTHENUM") or "bluetooth" in (description or "").lower()
    likely = not bluetooth and (vid in ESP32_USB_VIDS or any(k in (description or "").lower() for k in ESP32_KEYWORDS))
    return {"device": device, "description": description or "", "bluetooth": bluetooth, "likely_esp32": likely}


def list_serial_ports() -> list[dict[str, Any]]:
    """All serial ports, ESP32-looking USB ones first."""
    from serial.tools import list_ports
    ports = [classify_port(p.device, p.description, p.hwid or "", p.vid) for p in list_ports.comports()]
    return sorted(ports, key=lambda p: (not p["likely_esp32"], p["bluetooth"], p["device"]))


def find_esp32_port() -> str | None:
    """COM port of the first USB-serial device that looks like an ESP32 (None if nothing is plugged in)."""
    try:
        for p in list_serial_ports():
            if p["likely_esp32"]:
                return p["device"]
    except Exception as e:  # pyserial missing / driver trouble: behave as "not found"
        log.warning("port scan failed: %s", e)
    return None


# ------------------------------------------------------------------- links
class BaseLink:
    simulated = False

    def __init__(self, timeout_s: float = 10):
        self.timeout_s = timeout_s
        self.latest: dict[str, Any] = {}
        self.last_rx: float = 0.0
        self.error: str = ""

    @property
    def connected(self) -> bool:
        return self.last_rx > 0 and time.time() - self.last_rx < self.timeout_s

    def set_relay(self, channel: int, state: int) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class SerialLink(BaseLink):
    def __init__(self, port: str, baud: int = 115200, timeout_s: float = 10):
        super().__init__(timeout_s)
        self.port, self.baud = port, baud
        self._ser = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True, name="serial").start()

    def _open(self) -> None:
        import serial
        # serial_for_url opens normal ports ('COM7') and also test URLs like 'socket://127.0.0.1:5000'
        self._ser = serial.serial_for_url(self.port, baudrate=self.baud, timeout=1)
        time.sleep(2)  # ESP32 resets when port opens
        self.error = ""
        log.info("Opened %s", self.port)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                if self._ser is None or not self._ser.is_open:
                    self._open()
                raw = self._ser.readline().decode("utf-8", errors="ignore")
                msg = parse_line(raw) if raw else None
                if msg and msg["type"] == "telemetry":
                    self.latest = msg
                    self.last_rx = time.time()
                elif msg and msg["type"] == "error":
                    log.warning("ESP32 error: %s", msg)
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"
                log.warning("serial problem: %s (retry in 3s)", self.error)
                self._close_port()
                self._stop.wait(3)

    def set_relay(self, channel: int, state: int) -> None:
        cmd = build_relay_command(channel, state)
        with self._lock:
            if self._ser is None or not self._ser.is_open:
                raise ConnectionError("ESP32 not connected")
            self._ser.write(cmd.encode())

    def _close_port(self) -> None:
        try:
            if self._ser:
                self._ser.close()
        except Exception:
            pass
        self._ser = None

    def close(self) -> None:
        self._stop.set()
        self._close_port()


class SimulatedLink(BaseLink):
    """Fake ESP32. Every telemetry message carries "simulated": true."""
    simulated = True

    def __init__(self, timeout_s: float = 10, start_temp: float = 29.0):
        super().__init__(timeout_s)
        self.temp = start_temp
        self.region_offsets = [0.0, 0.4, -0.3, 0.8]     # small natural differences between regions
        self.region_temps = [start_temp + o for o in self.region_offsets]
        self.relays = {1: 0, 2: 0, 3: 0, 4: 0}
        self.ac_running = False          # toggled from dashboard to test AC inference
        self.sensor_fault = False        # toggled from dashboard to test alerts
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True, name="sim-esp32").start()

    def _run(self) -> None:
        while not self._stop.is_set():
            # AC pulls temperature down to 22 C, otherwise drifts back to 29 C
            cooling = self.ac_running or self.relays[4]
            for i, off in enumerate(self.region_offsets):
                target = (22.0 + i * 0.7) if cooling else 29.0 + off   # region 1 is nearest the AC
                rate = (0.02 - i * 0.003) if cooling else 0.01
                self.region_temps[i] += (target - self.region_temps[i]) * rate + random.uniform(-0.05, 0.05)
            temps = [None if self.sensor_fault else round(v, 1) for v in self.region_temps]
            self.temp = sum(self.region_temps) / len(self.region_temps)
            self.latest = {
                "type": "telemetry", "simulated": True, "temps": temps,
                "temp_c": None if self.sensor_fault else round(self.temp, 1),
                "sensor_ok": not self.sensor_fault,
                **{f"relay{c}": s for c, s in self.relays.items()},
            }
            self.last_rx = time.time()
            self._stop.wait(1)

    def set_relay(self, channel: int, state: int) -> None:
        build_relay_command(channel, state)  # same validation as real link
        self.relays[channel] = state

    def close(self) -> None:
        self._stop.set()
