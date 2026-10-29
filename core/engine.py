"""Engine: ties vision + ESP32 + logic + alerts together in one background loop.

Detection  -> Occupancy state -> Desired relay state -> Relay command -> ESP32 reported state
(the three relay concepts are kept separate: desired / commanded / reported)
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any

from core.alerts import AlertManager, build_alert_manager
from core.esp32 import BaseLink, SerialLink, SimulatedLink, find_esp32_port
from core.logic import (AC_ON_RESULTS, ACInference, ACMonitor, ACState, ACStatus, OccupancyMachine, State, SwitchGuard,
                        desired_relay_state)
from core.storage import Storage

log = logging.getLogger("engine")
RELAY_FAULT_AFTER_S = 5
CAMERA_RECOVER_S = 10
PRESENT_CONFIRM_S = 1.5   # a person must be seen this long to count as present


def _fmt_regions(temps) -> str:
    return ", ".join(f"R{i + 1}={'-' if v is None else v}" for i, v in enumerate(temps or []))


def _action_text(reason: str, on: list[str], off: list[str]) -> str:
    """Alert body that says exactly what to do: what to turn off, and what is already off."""
    text = f"{reason}\nTURN OFF: {', '.join(on)}"
    if off:
        text += f"\nAlready off: {', '.join(off)}"
    return text


def _split_appliances(apps: list[dict], frame_w: int) -> tuple[list[str], list[str]]:
    """Names of camera-seen fans/lights that are ON and OFF (with left/centre/right if there are several)."""
    apps = sorted((a for a in apps if a.get("state") in ("ON", "OFF")), key=lambda a: a.get("bbox", [0])[0])
    count = {c: sum(1 for a in apps if a["cls"] == c) for c in {a["cls"] for a in apps}}
    on, off = [], []
    for a in apps:
        name = a["cls"].capitalize()
        if count[a["cls"]] > 1 and a.get("bbox"):
            x = (a["bbox"][0] + a["bbox"][2]) / 2
            name += " (" + ("left" if x < frame_w / 3 else "right" if x > 2 * frame_w / 3 else "centre") + ")"
        (on if a["state"] == "ON" else off).append(name)
    return on, off


class Engine:
    def __init__(self, cfg: dict, link: BaseLink | None = None, vision=None,
                 storage: Storage | None = None, alerts: AlertManager | None = None, start_thread: bool = True):
        self.cfg = cfg
        o, a = cfg["occupancy"], cfg["ac_inference"]
        self.link = link or self._make_link(cfg)
        self.vision = vision or self._make_vision(cfg)
        self.storage = storage or Storage(cfg["logging"]["db_path"])
        self.alerts = alerts or build_alert_manager(cfg)
        self.alerts.on_alert.append(lambda al: self.storage.log_event(al.code, al.level, al.message))

        self.occ = OccupancyMachine(o["vacancy_confirm_s"], o["occupied_confirm_frames"], o["detection_fail_timeout_s"],
                                    o.get("vacancy_confirm_frames", 0))
        self._last_vis_ts: float | None = None       # camera frame the occupancy logic last saw
        self._vacant_since: float | None = None      # when the room became VACANT
        # post-vacancy regional delta T (report section 3.4); delta_threshold_c = delta_thresh of the report
        self.ac = ACInference(a["observation_s"], a.get("delta_threshold_c", a["cooling_threshold_c"]),
                              a["warming_threshold_c"], mode=a.get("delta_mode", "absolute"))
        self.acmon = ACMonitor(**cfg.get("ac_monitor", {}))   # continuous AC ON/OFF from the temperature trend
        self.guard = SwitchGuard(cfg["automation"]["min_switch_interval_s"])
        self.relay_cfg: dict[int, dict] = cfg["relays"]

        self.automation_enabled: bool = bool(cfg["automation"].get("enabled", False))
        self.manual_override = False
        self.commanded: dict[int, int] = {ch: 0 for ch in self.relay_cfg}   # what we sent
        self.desired: dict[int, int] = {ch: 0 for ch in self.relay_cfg}     # what policy wants
        self._cmd_time: dict[int, float] = {}
        self.history: deque[tuple[float, float]] = deque(maxlen=1800)
        self.region_history: deque[tuple[float, list]] = deque(maxlen=1800)   # (time, [T1..T4])
        self.state = State.INITIALIZING
        self._prev_occ = State.INITIALIZING
        self._last_db = 0.0
        self._empty_since: float | None = None      # when the camera last started seeing nobody
        self._cam_ok_since: float | None = None     # when the camera last became healthy
        self._ac_rx: float = 0.0                    # time of the sensor reading the AC logic last saw
        self._seen_since: float | None = None       # when a person was first seen in the current run
        self._app_state: dict[int, str] = {}        # last logged fan/light states
        self._started = time.time()
        self._lock = threading.RLock()
        self._stop = threading.Event()

        for w in getattr(self.vision, "warnings", []):
            self.storage.log_event("VISION", "INFO", w)
        self.storage.log_event("SYSTEM", "INFO",
                               f"Started. ESP32={'SIMULATED' if self.link.simulated else cfg['esp32']['port']}, "
                               f"automation={'ON' if self.automation_enabled else 'OFF'}")
        if start_thread:
            threading.Thread(target=self._loop, daemon=True, name="engine").start()

    # --------------------------------------------------------------- factories
    @staticmethod
    def _make_link(cfg: dict) -> BaseLink:
        e = cfg["esp32"]
        port = e.get("port")
        if str(port).lower() == "auto":              # find the ESP32 USB cable by itself
            port = find_esp32_port()
            if port:
                log.info("ESP32 auto-detected on %s", port)
            else:
                log.warning("ESP32 not found (plug in the USB cable): using the simulator")
        if port:
            return SerialLink(port, e.get("baud", 115200), e.get("telemetry_timeout_s", 10))
        return SimulatedLink(e.get("telemetry_timeout_s", 10))

    @staticmethod
    def _make_vision(cfg: dict):
        from core.vision import SimulatedVision, Vision
        return SimulatedVision(cfg) if cfg["vision"].get("simulate") else Vision(cfg)

    # -------------------------------------------------------------------- loop
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.step(time.time())
            except Exception as e:
                log.exception("engine step failed")
                self.storage.log_event("ERROR", "CRITICAL", f"Engine error: {e}")
            self._stop.wait(0.5)

    def step(self, now: float) -> None:
        with self._lock:
            vis = self.vision.snapshot()
            tel = self.link.latest or {}
            connected = self.link.connected
            temp = tel.get("temp_c") if connected else None              # mean of the regions
            region_temps = tel.get("temps") if connected else None        # one per classroom region
            sensor_ok = bool(tel.get("sensor_ok")) and connected
            reported = {ch: int(tel.get(f"relay{ch}", 0)) for ch in self.relay_cfg} if connected else dict(self.commanded)

            # 1. occupancy
            new_frame = (not vis.ts) or vis.ts != self._last_vis_ts      # no timestamp: count every result
            self._last_vis_ts = vis.ts
            occ = self.occ.update(vis.person_count if vis.ok else None, now, fresh=new_frame)
            if occ == State.VACANT:
                self._vacant_since = self._vacant_since or now
            else:
                self._vacant_since = None
            if occ != self._prev_occ:
                self.storage.log_event("STATE", "INFO", f"{self._prev_occ.value} -> {occ.value}")
                if occ == State.VACANT:
                    self.alerts.raise_alert("VACANCY_CONFIRMED", "INFO", "Classroom confirmed VACANT.", now)
                if occ == State.OCCUPIED:
                    self.alerts.clear("ENERGY_WASTE")
                self._prev_occ = occ

            for a in vis.appliances or []:           # log fan / light ON-OFF changes (visible on the Logs page)
                key = a.get("id", a["cls"])
                if self._app_state.get(key) != a["state"] and a["state"] != "UNKNOWN":
                    if key in self._app_state or a["state"] == "ON":
                        self.storage.log_event("APPLIANCE", "INFO", f"{a['cls'].capitalize()} #{key} is now {a['state']} "
                                               f"(signal {a.get('metric', 0) * 100:.0f}%)")
                    self._app_state[key] = a["state"]

            # 2. AC inference (only while vacant; never switches the real AC)
            rx = getattr(self.link, "last_rx", None)          # time of the newest sensor reading
            rx_prev, self._ac_rx = self._ac_rx, rx
            new_reading = rx is None or rx != rx_prev
            if new_reading and temp is not None and sensor_ok:
                self.acmon.add(temp, now)
            ac_change = self.acmon.update(now)                # "ON" / "OFF" when the AC state changes
            if occ == State.VACANT:
                if self.ac.status == ACStatus.IDLE:
                    self.ac.start(region_temps or temp, sensor_ok, now)
                    self.storage.log_event("AC", "INFO", "AC observation started, T_init per region: "
                                           + _fmt_regions(region_temps or [temp]))
                else:
                    before = self.ac.status
                    # a NEW sensor reading arrives every ~2 s; repeats of the same one must not count twice
                    fresh = rx is None or (not connected) or rx != rx_prev
                    self.ac.update(region_temps or temp, sensor_ok, now, fresh)
                    if before == ACStatus.OBSERVING and self.ac.status != ACStatus.OBSERVING:
                        per = "; ".join(f"R{i} {r['init']}->{r['final']} (dT {r['delta']})"
                                        for i, r in self.ac.regions.items())
                        self.storage.log_event("AC", "INFO", f"AC result {self.ac.status.value}: mean dT="
                                               f"{self.ac.delta_t} C over {len(self.ac.regions)} region(s), "
                                               f"threshold {self.ac.threshold_c} C. {per}")
            elif self.ac.status != ACStatus.IDLE:
                self.ac.reset()

            # 3. display state
            if self.manual_override:
                self.state = State.MANUAL_OVERRIDE
            elif occ == State.VACANT and self.ac.status == ACStatus.OBSERVING:
                self.state = State.AC_OBSERVATION
            else:
                self.state = occ

            # 4. desired state + automatic relay commands
            for ch, rc in self.relay_cfg.items():
                self.desired[ch] = desired_relay_state(occ, rc, reported[ch])
            off_after = self.cfg["automation"].get("off_after_vacant_s", 0)
            settled = occ == State.OCCUPIED or (occ == State.VACANT and now - (self._vacant_since or now) >= off_after)
            if self.automation_enabled and not self.manual_override and connected and settled:
                turned_off = []
                for ch in self.relay_cfg:
                    if self.desired[ch] != reported[ch] and self.guard.can_switch(ch, now):
                        if self._send(ch, self.desired[ch], now, source="AUTO") and self.desired[ch] == 0:
                            turned_off.append(self.relay_cfg[ch]["name"])
                if turned_off:
                    self.alerts.raise_alert("AUTO_SHUTDOWN", "INFO",
                                            f"Room vacant - switched OFF: {', '.join(turned_off)}", now)

            # 5. alert rules
            self._check_alerts(now, vis, tel, connected, temp, sensor_ok, occ, reported, ac_change)

            # 6. history + DB log
            if temp is not None and sensor_ok:
                self.history.append((now, temp))
                if region_temps and new_reading:
                    self.region_history.append((now, list(region_temps)))
            if now - self._last_db >= self.cfg["logging"].get("log_interval_s", 5):
                self._last_db = now
                status = ", ".join(f"{self.relay_cfg[c]['name']}={'ON' if reported[c] else 'OFF'}" for c in self.relay_cfg)
                seen = ", ".join(f"{a['cls'].capitalize()} {a.get('id', '')}={a['state']}" for a in vis.appliances or [])
                self.storage.log_status(vis.person_count, self.state.value, temp, sensor_ok,
                                        self.link.simulated, status, self.ac.status.value,
                                        humidity=tel.get("humidity") if connected else None,
                                        visual_status=seen, ac_monitor=self.acmon.state.value,
                                        region_temps=region_temps)

    def _check_alerts(self, now, vis, tel, connected, temp, sensor_ok, occ, reported, ac_change=None) -> None:
        A = self.alerts
        grace = now - self._started > self.cfg["esp32"].get("telemetry_timeout_s", 10)

        # hardware / camera faults
        if not connected and grace:
            A.raise_alert("ESP32_DISCONNECTED", "CRITICAL",
                          f"No data from ESP32. {self.link.error or 'Check USB cable / COM port.'}", now)
        else:
            A.clear("ESP32_DISCONNECTED")
        if connected and not sensor_ok:
            A.raise_alert("SENSOR_FAULT", "WARNING", "DHT11 temperature sensor is not giving valid readings.", now)
        else:
            A.clear("SENSOR_FAULT")
        if not vis.ok and vis.error and not getattr(self.vision, "loading", False):
            self._cam_ok_since = None
            A.raise_alert("CAMERA_FAULT", "CRITICAL", f"Vision problem: {vis.error}. Automation paused.", now)
        else:
            # clear only after the camera has been healthy for a while, otherwise a flickering camera
            # clears + re-raises the alert (and re-sends the notification) every few seconds
            if self._cam_ok_since is None:
                self._cam_ok_since = now
            if now - self._cam_ok_since >= CAMERA_RECOVER_S:
                A.clear("CAMERA_FAULT")

        # temperature
        if temp is not None and sensor_ok and temp >= self.cfg["alerts"].get("high_temp_c", 35):
            A.raise_alert("HIGH_TEMPERATURE", "WARNING", f"Room temperature is {temp:.1f} °C.", now)
        else:
            A.clear("HIGH_TEMPERATURE")

        # energy waste: vacant room but non-AC loads still ON (relay state reported by the ESP32)
        if occ == State.VACANT:
            non_ac = [c for c in self.relay_cfg if self.relay_cfg[c].get("kind") != "ac"]
            on = [self.relay_cfg[c]["name"] for c in non_ac if reported[c] == 1]
            off = [self.relay_cfg[c]["name"] for c in non_ac if reported[c] == 0]
            if on:
                A.raise_alert("ENERGY_WASTE", "WARNING", _action_text("The classroom is EMPTY.", on, off), now,
                              title=f"Turn OFF: {', '.join(on)} - room is empty")
            else:
                A.clear("ENERGY_WASTE")

        # camera sees a fan/light ON and nobody in the room (never while someone is present)
        # "empty" = no person for a while. A person only counts as PRESENT after being seen continuously for
        # PRESENT_CONFIRM_S, so one spurious detection can't restart the countdown.
        if vis.ok and vis.person_count == 0:
            self._seen_since = None
            if self._empty_since is None:
                self._empty_since = now
        elif vis.ok:
            if self._seen_since is None:
                self._seen_since = now
            if now - self._seen_since >= PRESENT_CONFIRM_S:
                self._empty_since = None
        else:                                   # no valid detection: we don't know, so never "empty"
            self._empty_since = self._seen_since = None
        delay = self.cfg["alerts"].get("visual_waste_delay_s", 8)
        on_v, off_v = _split_appliances(vis.appliances or [], vis.frame.shape[1] if vis.frame is not None else 640)
        if on_v and self._empty_since is not None and now - self._empty_since >= delay:
            A.raise_alert("VISUAL_ENERGY_WASTE", "WARNING",
                          _action_text(f"Nobody has been in the room for {int(now - self._empty_since)} s.", on_v, off_v),
                          now, title=f"Turn OFF: {', '.join(on_v)} - room is empty")
        else:
            A.clear("VISUAL_ENERGY_WASTE")

        # AC switched ON / OFF (from the temperature trend) and AC left ON in an empty room
        info = self.acmon.info(now)
        room_empty = occ == State.VACANT or (self._empty_since is not None and now - self._empty_since >= delay)
        if ac_change == "ON":
            self.storage.log_event("AC", "INFO", f"AC detected ON: temperature {info['slope_c_min']:+.2f} C/min, "
                                   f"dropped {info['drop_c']:.1f} C")
            if not room_empty:      # an empty room gets the stronger 'Turn OFF the AC' alert below instead
                A.raise_alert("AC_ON", "WARNING",
                              f"The temperature is falling {abs(info['slope_c_min']):.2f} C/min "
                              f"({info['drop_c']:.1f} C in the last {int(self.acmon.window_s / 60)} min), "
                              f"so the AC has been switched ON.", now, title="AC switched ON")
        elif ac_change == "OFF":
            self.storage.log_event("AC", "INFO", f"AC detected OFF: temperature {info['slope_c_min']:+.2f} C/min")
            A.clear("AC_ON")
            A.clear("AC_LEFT_ON")
            A.raise_alert("AC_OFF", "INFO", "The room is warming up again, so the AC has been switched OFF.",
                          now, title="AC switched OFF")
        ac_on = self.acmon.state == ACState.ON or self.ac.status in AC_ON_RESULTS
        if ac_on and room_empty:
            how = (f"the temperature is falling {abs(info['slope_c_min']):.2f} C/min"
                   if self.acmon.state == ACState.ON else
                   f"the mean regional temperature change was {abs(self.ac.delta_t or 0):.1f} C in "
                   f"{int(self.ac.observation_s)} s (threshold {self.ac.threshold_c} C)")
            A.raise_alert("AC_LEFT_ON", "CRITICAL",
                          f"The room is empty but {how}, so the AC is probably still running.\n"
                          f"TURN OFF: the AC (by hand - this system never switches the AC itself).",
                          now, title="Turn OFF the AC - room is empty")
        elif not room_empty or self.acmon.state == ACState.OFF:
            A.clear("AC_LEFT_ON")

        # relay fault: commanded state not reflected by ESP32 after a few seconds
        if connected:
            bad = [self.relay_cfg[c]["name"] for c in self.relay_cfg
                   if c in self._cmd_time and now - self._cmd_time[c] > RELAY_FAULT_AFTER_S
                   and reported[c] != self.commanded[c]]
            if bad:
                A.raise_alert("RELAY_FAULT", "CRITICAL", f"Relay not responding: {', '.join(bad)}.", now)
            else:
                A.clear("RELAY_FAULT")

    # ---------------------------------------------------------------- commands
    def _send(self, ch: int, state: int, now: float, source: str) -> bool:
        try:
            self.link.set_relay(ch, state)
        except Exception as e:
            self.storage.log_event("RELAY", "WARNING", f"{source} command failed for relay {ch}: {e}")
            return False
        self.commanded[ch] = state
        self._cmd_time[ch] = now
        self.guard.mark(ch, now)
        self.storage.log_event("RELAY", "INFO",
                               f"{source}: {self.relay_cfg[ch]['name']} (relay {ch}) -> {'ON' if state else 'OFF'}")
        return True

    def manual_set(self, ch: int, state: int) -> bool:
        """Dashboard toggle. Turns manual override ON so automation won't fight the user."""
        with self._lock:
            if not self.manual_override:
                self.manual_override = True
                self.storage.log_event("OVERRIDE", "INFO", "Manual override ON")
            return self._send(ch, state, time.time(), source="MANUAL")

    def set_manual_override(self, on: bool) -> None:
        with self._lock:
            self.manual_override = on
            self.storage.log_event("OVERRIDE", "INFO", f"Manual override {'ON' if on else 'OFF'}")

    def set_automation(self, on: bool) -> None:
        with self._lock:
            self.automation_enabled = on
            self.storage.log_event("AUTOMATION", "INFO", f"Automation {'ENABLED' if on else 'DISABLED'}")

    # ------------------------------------------------------- live reconfiguration
    def set_vision(self, simulate: bool, source=0, model_path: str | None = None,
                   confidence: float | None = None) -> None:
        """Swap simulated <-> webcam+YOLO at runtime (no restart). Runs the heavy load in the vision thread."""
        with self._lock:
            v = self.cfg["vision"]
            v["simulate"] = simulate
            v["source"] = source
            if model_path:
                v["model_path"] = model_path
            if confidence is not None:
                v["confidence"] = confidence
            self.vision.close()
            self.vision = self._make_vision(self.cfg)
            self.occ.reset_detection()
            self.storage.log_event("VISION", "INFO", f"Vision switched to "
                                   f"{'SIMULATED' if simulate else f'camera {source} + {v['model_path']}'}")
            for w in getattr(self.vision, "warnings", []):
                self.storage.log_event("VISION", "INFO", w)

    def set_esp32_port(self, port: str) -> None:
        """Connect to a COM port (or '' for the simulator) at runtime."""
        with self._lock:
            self.link.close()
            self.cfg["esp32"]["port"] = port
            self.link = self._make_link(self.cfg)
            port = getattr(self.link, "port", "")
            self._started = time.time()          # restart the disconnect grace period
            self.commanded = {ch: 0 for ch in self.relay_cfg}
            self._cmd_time.clear()
            self.storage.log_event("SYSTEM", "INFO", f"ESP32 link -> {port or 'SIMULATED'}"
                                   + (" (auto-detected)" if str(self.cfg["esp32"]["port"]).lower() == "auto" and port else ""))

    # ---------------------------------------------------------------- snapshot
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = time.time()
            tel = self.link.latest or {}
            return {
                "state": self.state.value,
                "occ_state": self.occ.state.value,
                "vacant_in": self.occ.seconds_to_vacant(now),
                "empty_frames": self.occ.empty_frames(),
                "connected": self.link.connected,
                "simulated_esp": self.link.simulated,
                "link_error": self.link.error,
                "link_port": getattr(self.link, "port", ""),
                "temp": tel.get("temp_c") if self.link.connected else None,
                "sensor_ok": bool(tel.get("sensor_ok")) and self.link.connected,
                "reported": {ch: int(tel.get(f"relay{ch}", 0)) for ch in self.relay_cfg},
                "commanded": dict(self.commanded),
                "desired": dict(self.desired),
                "ac": {"status": self.ac.status.value, "t_initial": self.ac.t_initial,
                       "t_final": self.ac.t_final, "delta_t": self.ac.delta_t,
                       "seconds_left": self.ac.seconds_left(now),
                       "delta_live": self.ac.live_delta(now),
                       "samples": sum(len(v) for v in self.ac.samples.values()),
                       "regions": self.ac.live_regions(now), "mode": self.ac.mode,
                       "threshold": self.ac.threshold_c},
                "region_temps": tel.get("temps") if self.link.connected else None,
                "humidity": tel.get("humidity") if self.link.connected else None,
                "acmon": self.acmon.info(now),
                "automation": self.automation_enabled,
                "manual_override": self.manual_override,
                "history": list(self.history),
                "region_history": list(self.region_history),
                "active_alerts": list(self.alerts.active.values()),
                "alert_history": list(self.alerts.history),
            }

    def close(self) -> None:
        self._stop.set()
        self.link.close()
        self.vision.close()
