"""Run:  python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.alerts import AlertManager  # noqa: E402
from core.engine import Engine  # noqa: E402
from core.esp32 import build_relay_command, parse_line  # noqa: E402
from core.logic import (ACInference, ACStatus, OccupancyMachine, State, SwitchGuard,  # noqa: E402
                        classify_delta, desired_relay_state)
from core.regions import count_by_region  # noqa: E402
from core.vision import VisionResult, resolve_classes  # noqa: E402


# ---------------------------------------------------------------- occupancy
def test_starts_initializing():
    assert OccupancyMachine().state == State.INITIALIZING


def test_person_marks_occupied_after_debounce():
    m = OccupancyMachine(vacancy_confirm_s=300, occupied_confirm_frames=2)
    assert m.update(1, 0) == State.INITIALIZING
    assert m.update(1, 1) == State.OCCUPIED


def test_vacancy_timer_and_cancel():
    m = OccupancyMachine(vacancy_confirm_s=300, occupied_confirm_frames=1)
    m.update(2, 0)
    assert m.update(0, 10) == State.VACANCY_PENDING
    assert m.update(0, 200) == State.VACANCY_PENDING
    assert m.update(1, 250) == State.OCCUPIED            # person returns -> cancelled
    m.update(0, 260)
    assert m.update(0, 559) == State.VACANCY_PENDING
    assert m.update(0, 560) == State.VACANT


def test_detection_failure_never_vacant():
    m = OccupancyMachine(vacancy_confirm_s=5, occupied_confirm_frames=1, fail_timeout_s=30)
    m.update(1, 0)
    assert m.update(None, 10) == State.OCCUPIED           # holds
    assert m.update(None, 31) == State.ERROR              # then ERROR, not VACANT


# ------------------------------------------------------------- AC inference
@pytest.mark.parametrize("delta,expected", [
    (-2.0, ACStatus.COOLING_OBSERVED), (-1.0, ACStatus.COOLING_OBSERVED),
    (-0.5, ACStatus.INCONCLUSIVE), (0.0, ACStatus.INCONCLUSIVE),
    (1.0, ACStatus.NO_SIGNIFICANT_COOLING_OBSERVED),
])
def test_classify_delta(delta, expected):
    assert classify_delta(delta, 1.0, 1.0) == expected


def test_ac_observation_cycle():
    ac = ACInference(observation_s=120, mode="signed")
    assert ac.start(29.0, True, 0) == ACStatus.OBSERVING
    assert ac.update(28.5, True, 60) == ACStatus.OBSERVING
    assert ac.update(27.0, True, 120) == ACStatus.COOLING_OBSERVED
    assert ac.delta_t == -2.0


def test_ac_dropped_readings_are_tolerated_but_a_dead_sensor_is_not():
    ac = ACInference(observation_s=120)
    ac.start(29.0, True, 0)
    assert ac.update(None, False, 50) == ACStatus.OBSERVING          # one dropped DHT11 read is normal
    assert ac.update(28.0, True, 52) == ACStatus.OBSERVING           # and it recovers
    ac2 = ACInference(observation_s=120, max_invalid_s=15)
    ac2.start(29.0, True, 0)
    ac2.update(None, False, 50)
    assert ac2.update(None, False, 70) == ACStatus.SENSOR_ERROR      # nothing valid for 20 s -> real sensor fault
    assert ACInference().start(None, False, 0) == ACStatus.SENSOR_ERROR


def _dht11_stream(ac, true_temp, start=0, end=120, step=2.0, seed=1):
    """Feed DHT11-like readings: whole degrees, +-1 C noise, 5% dropped reads. true_temp(t) -> real temperature."""
    import random
    rng = random.Random(seed)
    t = start
    while t <= end:
        if rng.random() < 0.05:
            ac.update(None, False, t)
        else:
            ac.update(float(round(true_temp(t) + rng.uniform(-0.6, 0.6))), True, t)
        t += step


def test_delta_t_with_realistic_dht11_noise():
    cooling = ACInference(observation_s=120, mode="signed")
    cooling.start(29.0, True, 0)
    _dht11_stream(cooling, lambda t: 29.0 - 2.5 * t / 120)               # real room cools by 2.5 C (AC running)
    assert cooling.status == ACStatus.COOLING_OBSERVED and cooling.delta_t <= -1.0

    steady = ACInference(observation_s=120, mode="signed")
    steady.start(29.0, True, 0)
    _dht11_stream(steady, lambda t: 29.0, seed=7)                         # AC is off, temperature flat
    assert steady.status in (ACStatus.INCONCLUSIVE, ACStatus.NO_SIGNIFICANT_COOLING_OBSERVED)
    assert steady.status != ACStatus.COOLING_OBSERVED                     # noise alone must not raise the alarm


def test_ac_live_delta_and_duplicate_readings():
    ac = ACInference(observation_s=120, mode="signed")
    ac.start(29.0, True, 0)
    for t in range(2, 100, 2):
        ac.update(27.0 if t > 60 else 29.0, True, float(t))
    assert ac.live_delta(100.0) == -2.0 and ac.status == ACStatus.OBSERVING
    n = len(ac.samples[1])
    ac.update(27.0, True, 101.0, fresh=False)                             # the same reading seen again
    assert len(ac.samples[1]) == n


def test_humidity_parsing():
    assert parse_line('{"type":"telemetry","temp_c":29.0,"humidity":55,"sensor_ok":true}')["humidity"] == 55
    assert parse_line('{"type":"telemetry","temp_c":29.0,"humidity":null,"sensor_ok":true}')["humidity"] is None
    assert parse_line('{"type":"telemetry","temp_c":29.0,"humidity":300,"sensor_ok":true}')["humidity"] is None


# ------------------------------------------------------------ relay policy
def test_policy():
    auto = {"auto": True, "occupied_state": 1}
    assert desired_relay_state(State.VACANT, auto, 1) == 0
    assert desired_relay_state(State.OCCUPIED, auto, 0) == 1
    assert desired_relay_state(State.VACANCY_PENDING, auto, 1) == 1
    assert desired_relay_state(State.ERROR, auto, 1) == 1
    assert desired_relay_state(State.VACANT, {"auto": False}, 1) == 1


def test_switch_guard():
    g = SwitchGuard(10)
    assert g.can_switch(1, 0)
    g.mark(1, 0)
    assert not g.can_switch(1, 5)
    assert g.can_switch(1, 10)


# ------------------------------------------------------------------ serial
def test_parse_telemetry():
    m = parse_line('{"type":"telemetry","temp_c":29.0,"sensor_ok":true,"relay1":1,"relay2":0,"relay3":0,"relay4":0}')
    assert m["temp_c"] == 29.0 and m["sensor_ok"] and m["relay1"] == 1


@pytest.mark.parametrize("line", ["", "garbage", "{bad json", '{"no_type":1}', "ets Jun  8 2016 boot"])
def test_parse_junk(line):
    assert parse_line(line) is None


def test_parse_impossible_temp_marked_invalid():
    m = parse_line('{"type":"telemetry","temp_c":null,"sensor_ok":true}')
    assert m["sensor_ok"] is False and m["temp_c"] is None
    assert parse_line('{"type":"telemetry","temp_c":500,"sensor_ok":true}')["sensor_ok"] is False


def test_relay_command_validation():
    assert build_relay_command(1, 1) == '{"type": "relay", "channel": 1, "state": 1}\n'
    for ch, st in [(0, 1), (5, 0), (1, 2), (1, -1)]:
        with pytest.raises(ValueError):
            build_relay_command(ch, st)


# ------------------------------------------------------------------ vision
def test_region_counting():
    regions = {"R1": [0, 0, .5, .5], "R2": [.5, 0, 1, .5], "R3": [0, .5, .5, 1], "R4": [.5, .5, 1, 1]}
    boxes = [(10, 10, 50, 100), (400, 300, 500, 480), (600, 10, 620, 200)]
    assert count_by_region(boxes, regions, 640, 480) == {"R1": 1, "R2": 1, "R3": 0, "R4": 1}


def test_class_mapping_no_fake_classes():
    m = resolve_classes({0: "person", 1: "bicycle"}, {"person": ["person"], "fan": ["fan"]})
    assert m == {"person": [0], "fan": []}


# ------------------------------------------------------------------ alerts
def test_alert_cooldown_and_clear():
    a = AlertManager(cooldown_s=300)
    sent = []
    a.notifiers.append(sent.append)
    assert a.raise_alert("X", "WARNING", "m", now=0)
    assert not a.raise_alert("X", "WARNING", "m", now=100)   # suppressed
    assert "X" in a.active
    a.clear("X")
    assert "X" not in a.active
    assert a.raise_alert("X", "WARNING", "m", now=101)       # can fire again after clear


# --------------------------------------------------- engine (integration)
class FakeVision:
    warnings: list = []
    def __init__(self): self.count = 3
    def snapshot(self): return VisionResult(ok=True, person_count=self.count)
    def close(self): pass


class FakeLink:
    simulated, error = True, ""
    def __init__(self): self.relays = {1: 1, 2: 1, 3: 1, 4: 0}; self.temp = 29.0; self.connected = True
    @property
    def latest(self): return {"type": "telemetry", "temp_c": self.temp, "sensor_ok": True,
                              **{f"relay{c}": s for c, s in self.relays.items()}}
    def set_relay(self, ch, st): self.relays[ch] = st
    def close(self): pass


def make_engine(tmp_path, automation):
    cfg = {
        "esp32": {"port": "", "telemetry_timeout_s": 10},
        "vision": {"simulate": True, "regions": {}},
        "occupancy": {"vacancy_confirm_s": 60, "occupied_confirm_frames": 1, "detection_fail_timeout_s": 30},
        "ac_inference": {"observation_s": 120, "cooling_threshold_c": 1.0, "warming_threshold_c": 1.0},
        "automation": {"enabled": automation, "min_switch_interval_s": 10},
        "relays": {1: {"name": "Light 1", "kind": "light", "auto": True, "occupied_state": 1},
                   2: {"name": "Light 2", "kind": "light", "auto": True, "occupied_state": 1},
                   3: {"name": "Fan", "kind": "fan", "auto": True, "occupied_state": 1},
                   4: {"name": "AC", "kind": "ac", "auto": False, "occupied_state": 0}},
        "alerts": {"cooldown_s": 300, "high_temp_c": 35, "telegram": {}},
        "logging": {"db_path": str(tmp_path / "t.db"), "log_interval_s": 5},
    }
    v, l = FakeVision(), FakeLink()
    return Engine(cfg, link=l, vision=v, start_thread=False), v, l


def test_engine_automation_off_only_alerts(tmp_path):
    e, v, l = make_engine(tmp_path, automation=False)
    e.step(0); v.count = 0; e.step(1); e.step(62)
    assert e.state in (State.VACANT, State.AC_OBSERVATION)
    assert l.relays == {1: 1, 2: 1, 3: 1, 4: 0}            # nothing switched
    assert "ENERGY_WASTE" in e.alerts.active


def test_engine_auto_shutdown_restore_and_ac_alert(tmp_path):
    e, v, l = make_engine(tmp_path, automation=True)
    e.step(0); v.count = 0; e.step(1); e.step(62)
    assert l.relays == {1: 0, 2: 0, 3: 0, 4: 0}
    assert e.state == State.AC_OBSERVATION
    l.temp = 27.0; e.step(62 + 121)
    assert e.ac.status == ACStatus.AC_ACTIVE and "AC_LEFT_ON" in e.alerts.active
    v.count = 2; e.step(200)
    assert e.state == State.OCCUPIED and l.relays == {1: 1, 2: 1, 3: 1, 4: 0}


def test_engine_manual_override_blocks_auto(tmp_path):
    e, v, l = make_engine(tmp_path, automation=True)
    e.step(0)
    e.manual_set(1, 1)
    v.count = 0; e.step(1); e.step(62)
    assert e.state == State.MANUAL_OVERRIDE and l.relays[1] == 1


# ---------------------------------------------------------------- fan / light ON-OFF
def _bg(seed=0):
    return np.random.default_rng(seed).integers(90, 110, (240, 320), dtype=np.uint8)


def _run_tracker(cls, make_frame, n=20):
    from core.appliances import ApplianceTracker
    tr = ApplianceTracker({"state_window_s": 2.0})
    src = "glow" if cls == "light" else "model"          # lights are found by their glow
    tr.update_detections([(cls, 0.9, (100, 60, 220, 180), src)], 0.0)
    for i in range(n):
        tr.update_frame(make_frame(i), i * 0.1)
    return tr.snapshot()[0]["state"]


def test_fan_still_is_off():
    base = _bg()
    assert _run_tracker("fan", lambda i: base.copy()) == "OFF"


def test_fan_moving_is_on():
    def frame(i):   # blades: a bar that keeps rotating, so the picture differs from 0.4 s ago
        f = _bg()
        angle = i * 0.5
        for k in range(-60, 61, 2):
            f[int(120 + k * np.sin(angle)) % 240, int(160 + k * np.cos(angle)) % 320] = 230
        f[100:140, 140:180] = 40 + (i % 5) * 40          # hub flicker
        return f
    assert _run_tracker("fan", frame) == "ON"


def test_slow_fan_is_still_on():
    def frame(i):   # blade barely moves between consecutive frames (3 px) but clearly over 0.4 s
        f = _bg()
        x = 105 + i * 3
        f[70:170, x:x + 30] = 235
        return f
    assert _run_tracker("fan", frame, n=30) == "ON"


def test_fan_ignores_global_brightness_change():
    base = _bg()
    assert _run_tracker("fan", lambda i: np.clip(base.astype(int) + (40 if i % 2 else 0), 0, 255).astype(np.uint8)) == "OFF"


def test_light_glowing_is_on_and_dark_is_off():
    def lit(i):
        f = _bg(); f[60:180, 100:220] = 255
        return f
    assert _run_tracker("light", lit) == "ON"
    assert _run_tracker("light", lambda i: _bg()) == "OFF"


def test_light_inside_fan_is_not_a_separate_light():
    from core.appliances import ApplianceTracker
    tr = ApplianceTracker({})
    tr.update_detections([("fan", 0.8, (100, 60, 270, 150)), ("light", 0.3, (160, 70, 215, 130))], 0.0)
    assert [t["cls"] for t in tr.snapshot()] == ["fan"]
    tr.update_detections([("light", 0.3, (160, 70, 215, 130))], 1.0)       # later passes too
    assert [t["cls"] for t in tr.snapshot()] == ["fan"]
    tr.update_detections([("light", 0.5, (10, 10, 60, 40))], 2.0)           # a real separate light is kept
    assert sorted(t["cls"] for t in tr.snapshot()) == ["fan", "light"]


def test_visual_alert_only_when_room_empty(tmp_path):
    e, v, l = make_engine(tmp_path, automation=False)
    v.snapshot = lambda: VisionResult(ok=True, person_count=v.count,
                                      appliances=[{"cls": "fan", "state": "ON"}, {"cls": "light", "state": "OFF"}])
    e.step(0)
    assert "VISUAL_ENERGY_WASTE" not in e.alerts.active          # someone is in the room -> no alert
    v.count = 0; e.step(1)
    e.step(5)
    assert "VISUAL_ENERGY_WASTE" not in e.alerts.active          # empty for only 4 s (delay is 8 s)
    e.step(10)
    assert "VISUAL_ENERGY_WASTE" in e.alerts.active and "TURN OFF: Fan" in e.alerts.active["VISUAL_ENERGY_WASTE"].message
    v.count = 1; e.step(11); e.step(13)
    assert "VISUAL_ENERGY_WASTE" not in e.alerts.active          # person returned (seen 1.5 s+) -> cleared


def test_camera_fault_does_not_flap(tmp_path):
    e, v, l = make_engine(tmp_path, automation=False)
    bad = lambda: VisionResult(ok=False, error="Detection stalled")
    good = lambda: VisionResult(ok=True, person_count=1)
    sent = []
    e.alerts.notifiers.append(sent.append)
    e.alerts.cooldown_s = 300
    v.snapshot = bad; e.step(20)
    assert "CAMERA_FAULT" in e.alerts.active
    for t in range(21, 40, 2):                # camera flickers good/bad every 2 s
        v.snapshot = good if t % 4 == 1 else bad
        e.step(t)
    assert "CAMERA_FAULT" in e.alerts.active
    assert sum(1 for a in e.alerts.history if a.code == "CAMERA_FAULT") == 1   # ONE log entry, not one per flicker
    assert not [a for a in sent if a.code == "CAMERA_FAULT"]                  # and never a popup (not a turn-off alert)
    v.snapshot = good
    for t in range(40, 55): e.step(t)         # healthy for >10 s -> cleared
    assert "CAMERA_FAULT" not in e.alerts.active


def test_bright_window_is_not_a_lit_light():
    def glare(i):   # whole area around the box is just as bright: a window, not a lamp
        f = _bg(); f[:, :] = 250
        return f
    assert _run_tracker("light", glare) == "OFF"


def test_alert_says_what_to_turn_off_and_what_is_already_off(tmp_path):
    e, v, l = make_engine(tmp_path, automation=False)
    v.snapshot = lambda: VisionResult(ok=True, person_count=v.count, appliances=[
        {"cls": "fan", "state": "ON", "bbox": [300, 0, 450, 90]},
        {"cls": "light", "state": "ON", "bbox": [20, 10, 80, 60]},
        {"cls": "light", "state": "OFF", "bbox": [560, 10, 620, 60]}])
    v.count = 0
    e.step(0); e.step(10)
    a = e.alerts.active["VISUAL_ENERGY_WASTE"]
    assert a.title == "Turn OFF: Light (left), Fan - room is empty"
    assert "TURN OFF: Light (left), Fan" in a.message and "Already off: Light (right)" in a.message
    assert "Turn OFF: Light (left), Fan" in a.text()


def test_only_turn_off_alerts_notify():
    a = AlertManager(cooldown_s=0)
    sent = []
    a.notifiers.append(sent.append)
    for code in ("CAMERA_FAULT", "TEST_ALERT", "HIGH_TEMPERATURE", "ESP32_DISCONNECTED", "VACANCY_CONFIRMED"):
        a.raise_alert(code, "CRITICAL", "x", now=0)
    import time; time.sleep(0.1)
    assert sent == []                                   # faults / test / info: silent
    a.raise_alert("VISUAL_ENERGY_WASTE", "WARNING", "TURN OFF: Fan", now=1, title="Turn OFF: Fan")
    time.sleep(0.2)
    assert [x.code for x in sent] == ["VISUAL_ENERGY_WASTE"]
    assert len(a.history) == 6                          # but everything is still logged


def test_log_viewer_queries(tmp_path):
    from core.storage import Storage
    st = Storage(str(tmp_path / "l.db"))
    st.log_event("CAMERA_FAULT", "CRITICAL", "Vision problem")
    st.log_event("STATE", "INFO", "OCCUPIED -> VACANT")
    st.log_event("VISUAL_ENERGY_WASTE", "WARNING", "TURN OFF: Fan")
    st.log_status(2, "OCCUPIED", 29.5, True, False, "Fan=ON", "IDLE")
    assert len(st.query_events()) == 3
    assert [r[1] for r in st.query_events(levels=["CRITICAL"])] == ["CAMERA_FAULT"]
    assert [r[1] for r in st.query_events(kinds=["STATE"])] == ["STATE"]
    assert [r[1] for r in st.query_events(text="fan")] == ["VISUAL_ENERGY_WASTE"]
    assert st.query_events(since="2999-01-01 00:00:00") == []
    assert st.event_counts() == {"CRITICAL": 1, "INFO": 1, "WARNING": 1}
    assert "STATE" in st.event_kinds()
    assert st.query_status()[0][1] == 2


def test_fan_quarter_turn_aliasing_is_still_on():
    import cv2
    def frame(i):   # 4 blades, turning exactly 90 degrees per 0.4 s: identical to the frame 0.4 s earlier
        f = _bg()
        ang = np.deg2rad(i * 22.5)
        for k in range(4):
            a = ang + k * np.pi / 2
            cv2.line(f, (160, 120), (int(160 + 55 * np.cos(a)), int(120 + 55 * np.sin(a))), 235, 9)
        return f
    assert _run_tracker("fan", frame, n=30) == "ON"


def test_one_spurious_person_does_not_restart_empty_timer(tmp_path):
    e, v, l = make_engine(tmp_path, automation=False)
    v.snapshot = lambda: VisionResult(ok=True, person_count=v.count,
                                      appliances=[{"id": 1, "cls": "fan", "state": "ON", "bbox": [0, 0, 9, 9]}])
    v.count = 1; e.step(0)
    v.count = 0; e.step(1)
    v.count = 1; e.step(4)            # one false detection
    v.count = 0; e.step(4.5); e.step(8); e.step(10)
    assert "VISUAL_ENERGY_WASTE" in e.alerts.active           # countdown was not reset by the blip
    v.count = 1; e.step(11); e.step(12); e.step(13)           # a real person stays
    assert "VISUAL_ENERGY_WASTE" not in e.alerts.active


def _blade_frame(i, spinning):
    """Stopped fan = crisp dark blades. Spinning fast = blades blurred away, only the hub stays."""
    import cv2
    f = np.full((240, 320), 190, dtype=np.uint8) + np.random.default_rng(i).integers(0, 4, (240, 320), dtype=np.uint8)
    cv2.circle(f, (160, 120), 14, 50, -1)                                   # motor housing
    if not spinning:
        for k in range(4):
            a = k * np.pi / 2 + 0.4
            cv2.line(f, (160, 120), (int(160 + 70 * np.cos(a)), int(120 + 70 * np.sin(a))), 40, 16)
    return f


def _zone_state(tr, spinning, n=20):
    for i in range(n):
        tr.update_frame(_blade_frame(0 if not spinning else 0, spinning), 100 + i * 0.1)
    return next(a["state"] for a in tr.snapshot() if a.get("zone"))


def test_fan_zone_blurred_blades_means_on(tmp_path):
    from core.appliances import ApplianceTracker
    tr = ApplianceTracker({"zones_path": str(tmp_path / "z.json")})
    zid = tr.add_zone("fan", (80, 40, 240, 200))
    for i in range(10): tr.update_frame(_blade_frame(0, False), i * 0.1)       # stopped: learn the OFF look
    assert tr.mark_zone(zid, "OFF")
    for i in range(10): tr.update_frame(_blade_frame(0, True), 1 + i * 0.1)    # spinning: learn the ON look
    assert tr.mark_zone(zid, "ON")
    z = tr.zone_info()[0]
    assert z["off"] > z["on"] + 0.1
    assert _zone_state(tr, spinning=True) == "ON"      # no pixel motion at all (static blur), but blades are gone
    assert _zone_state(tr, spinning=False) == "OFF"


def test_zones_persist_and_own_their_area(tmp_path):
    from core.appliances import ApplianceTracker
    path = str(tmp_path / "z.json")
    a = ApplianceTracker({"zones_path": path})
    a.add_zone("fan", (80, 40, 240, 200))
    b = ApplianceTracker({"zones_path": path})                                  # restart: zone still there
    assert [z["cls"] for z in b.zone_info()] == ["fan"]
    b.update_detections([("light", 0.4, (140, 90, 190, 150)), ("light", 0.5, (10, 10, 40, 30))], 0.0)
    kinds = [(x["cls"], x.get("zone", False)) for x in b.snapshot()]
    assert ("light", False) in kinds and kinds.count(("light", False)) == 1     # housing "light" inside the fan zone ignored
    b.clear_zones()
    assert ApplianceTracker({"zones_path": path}).zone_info() == []


def test_person_inside_ignore_zone_is_dropped():
    from core.vision import center_in_zones
    zones = [[345, 262, 500, 385]]
    assert center_in_zones((360, 280, 480, 370), zones)            # the framed picture
    assert not center_in_zones((200, 250, 330, 470), zones)        # a real person elsewhere
    assert not center_in_zones((360, 280, 480, 370), [])


def test_esp32_port_classification():
    from core.esp32 import classify_port
    bt = classify_port("COM5", "Standard Serial over Bluetooth link (COM5)", "BTHENUM\\{0000}", None)
    assert bt["bluetooth"] and not bt["likely_esp32"]                         # Bluetooth is never the ESP32
    ch340 = classify_port("COM7", "USB-SERIAL CH340 (COM7)", "USB VID:PID=1A86:7523", 0x1A86)
    assert ch340["likely_esp32"] and not ch340["bluetooth"]
    cp210 = classify_port("COM8", "Silicon Labs CP210x USB to UART Bridge (COM8)", "USB VID:PID=10C4:EA60", 0x10C4)
    assert cp210["likely_esp32"]
    assert not classify_port("COM9", "Some Modem", "USB VID:PID=1234:5678", 0x1234)["likely_esp32"]


def test_auto_port_falls_back_to_simulator_when_unplugged(monkeypatch):
    import core.engine as eng_mod
    monkeypatch.setattr(eng_mod, "find_esp32_port", lambda: None)
    link = eng_mod.Engine._make_link({"esp32": {"port": "auto", "telemetry_timeout_s": 10}})
    assert link.simulated
    link.close()
    monkeypatch.setattr(eng_mod, "find_esp32_port", lambda: "COM7")
    made = []
    monkeypatch.setattr(eng_mod, "SerialLink", lambda port, baud, t: made.append(port) or object())
    eng_mod.Engine._make_link({"esp32": {"port": "auto", "telemetry_timeout_s": 10}})
    assert made == ["COM7"]


class FakeFirmware:
    """Speaks exactly the protocol of firmware/esp32_controller.ino over a local TCP socket."""

    def __init__(self):
        import socket
        self.srv = socket.socket(); self.srv.bind(("127.0.0.1", 0)); self.srv.listen(1)
        self.port = self.srv.getsockname()[1]
        self.relays, self.temp, self.sensor_ok, self.received = [0, 0, 0, 0], 28.0, True, []
        import threading
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        import json, time
        conn, _ = self.srv.accept(); conn.settimeout(0.05)
        conn.sendall(b'{"type":"boot","msg":"esp32 ready, relays OFF"}\n')
        buf, last = b"", 0.0
        while True:
            try:
                data = conn.recv(256)
                if not data:
                    return
                buf += data
            except OSError:
                pass
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                self.received.append(line.decode())
                msg = json.loads(line)
                if msg["type"] == "relay":
                    self.relays[msg["channel"] - 1] = msg["state"]
                    conn.sendall(json.dumps({"type": "ack", "channel": msg["channel"], "state": msg["state"]}).encode() + b"\n")
                    last = 0.0                                   # fresh telemetry right away, like the firmware
                elif msg["type"] == "all_off":
                    self.relays = [0, 0, 0, 0]
            if time.time() - last > 0.3:
                last = time.time()
                t = f'{self.temp:.1f}' if self.sensor_ok else "null"
                conn.sendall((f'{{"type":"telemetry","temp_c":{t},"sensor_ok":{str(self.sensor_ok).lower()},'
                              f'"relay1":{self.relays[0]},"relay2":{self.relays[1]},"relay3":{self.relays[2]},"relay4":{self.relays[3]}}}\n').encode())


def _wait(cond, timeout=8.0):
    import time
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def test_serial_link_end_to_end_with_fake_esp32():
    from core.esp32 import SerialLink
    fw = FakeFirmware()
    link = SerialLink(f"socket://127.0.0.1:{fw.port}", 115200, timeout_s=5)
    try:
        assert _wait(lambda: link.connected), link.error                       # telemetry arrives
        assert link.latest["temp_c"] == 28.0 and link.latest["sensor_ok"] is True
        link.set_relay(3, 1)                                                   # turn the Fan relay ON
        assert _wait(lambda: link.latest.get("relay3") == 1)                   # the ESP32 reports it back
        assert fw.relays == [0, 0, 1, 0] and any('"channel": 3' in r for r in fw.received)
        link.set_relay(3, 0)
        assert _wait(lambda: link.latest.get("relay3") == 0)
        fw.sensor_ok = False                                                   # unplug the DHT11
        assert _wait(lambda: link.latest.get("sensor_ok") is False)
        assert link.latest["temp_c"] is None                                   # never a fake temperature
    finally:
        link.close()


def test_engine_with_real_serial_link_switches_relays(tmp_path):
    """Whole chain: vision count -> state machine -> relay command -> (fake) ESP32 -> reported state."""
    from core.esp32 import SerialLink
    fw = FakeFirmware()
    link = SerialLink(f"socket://127.0.0.1:{fw.port}", 115200, timeout_s=5)
    e, v, _ = make_engine(tmp_path, automation=True)
    e.link = link
    try:
        assert _wait(lambda: link.connected), link.error
        for ch in (1, 2, 3):
            link.set_relay(ch, 1)
        assert _wait(lambda: link.latest.get("relay1") == 1 and link.latest.get("relay3") == 1)
        import time
        t0 = time.time() + 1000                                                # virtual clock for the state machine
        v.count = 3; e.step(t0)
        v.count = 0; e.step(t0 + 1); e.step(t0 + 62)                           # empty for > 60 s -> VACANT -> auto shutdown
        assert _wait(lambda: fw.relays[:3] == [0, 0, 0]), fw.relays             # the fake ESP32 really got the OFF commands
    finally:
        link.close()


# ------------------------------------------------------ continuous AC monitor
def _feed(mon, temp_of_t, t0, t1, step=2.0, seed=3, noise=0.6):
    """DHT11-like readings every 2 s: whole degrees with noise. Returns the reported changes."""
    import random
    rng, changes, t = random.Random(seed), [], t0
    while t <= t1:
        mon.add(float(round(temp_of_t(t) + rng.uniform(-noise, noise))), t)
        c = mon.update(t)
        if c:
            changes.append((round(t), c))
        t += step
    return changes


def test_ac_monitor_flat_room_is_off_and_noise_never_triggers():
    from core.logic import ACMonitor
    m = ACMonitor()
    assert _feed(m, lambda t: 29.0, 0, 900) == []
    assert m.state.value == "OFF"


def test_ac_monitor_detects_on_then_off():
    from core.logic import ACMonitor
    m = ACMonitor()
    # flat 29 C for 5 min, AC on: cools 0.6 C/min for 5 min, AC off: warms 0.4 C/min
    def room(t):
        if t < 300: return 29.0
        if t < 600: return 29.0 - 0.6 * (t - 300) / 60
        return 26.0 + 0.4 * (t - 600) / 60
    changes = _feed(m, room, 0, 1200)
    assert [c for _, c in changes] == ["ON", "OFF"]
    on_t, off_t = changes[0][0], changes[1][0]
    assert 300 < on_t < 600 and off_t > 600          # detected while cooling / after warming started


def test_ac_monitor_already_on_at_startup_and_stale_sensor():
    from core.logic import ACMonitor
    m = ACMonitor()
    changes = _feed(m, lambda t: 29.0 - 0.6 * t / 60, 0, 300)
    assert changes and changes[0][1] == "ON"           # an AC that was already running is still reported
    assert m.update(400) is None and m.state.value == "UNKNOWN"   # no readings for 100 s -> don't guess


def test_engine_ac_on_off_and_left_on_alerts(tmp_path):
    e, v, l = make_engine(tmp_path, automation=False)
    v.count = 2                                         # someone in the room
    t = 0.0
    while t < 300:                                      # flat room
        l.temp = 29.0; e.step(t); t += 2
    while t < 600:                                      # AC switched on: room cools
        l.temp = float(round(29.0 - 0.6 * (t - 300) / 60)); e.step(t); t += 2
    assert "AC_ON" in e.alerts.active and e.alerts.active["AC_ON"].title == "AC switched ON"
    assert "AC_LEFT_ON" not in e.alerts.active          # people are there: just informational
    v.count = 0                                         # everyone leaves, AC still running
    for _ in range(10):
        l.temp = float(round(29.0 - 0.6 * (t - 300) / 60)); e.step(t); t += 2
    assert "AC_LEFT_ON" in e.alerts.active and "TURN OFF: the AC" in e.alerts.active["AC_LEFT_ON"].message
    while t < 1300:                                     # AC switched off: room warms
        l.temp = float(round(24.0 + 0.5 * (t - 620) / 60)); e.step(t); t += 2
    assert "AC_LEFT_ON" not in e.alerts.active and "AC_ON" not in e.alerts.active
    assert any(a.code == "AC_OFF" for a in e.alerts.history)


# ------------------------------------------------------------------ analysis
def test_analysis_usage_waste_and_ac(tmp_path):
    from core import analysis as an
    from core.storage import Storage
    import sqlite3
    st = Storage(str(tmp_path / "a.db"))
    rows = []                       # 10 minutes: 5 occupied (all ON), 5 empty with the fan left ON and the AC running
    for i in range(120):
        occ = i < 60
        rows.append((f"2026-10-24 10:{i * 5 // 60:02d}:{i * 5 % 60:02d}", 2 if occ else 0,
                     "OCCUPIED" if occ else "VACANT", 29.0 - (0 if occ else 2), 1, 0,
                     "Light 1=ON, Light 2=ON, Fan=ON, AC (demo LED)=OFF" if occ else
                     "Light 1=OFF, Light 2=OFF, Fan=ON, AC (demo LED)=OFF",
                     "IDLE", 55, "Fan z1=ON", "OFF" if occ else "ON", 29.0, 29.5, None, None))
    con = sqlite3.connect(str(tmp_path / "a.db"))
    con.executemany("INSERT INTO log (timestamp, occupancy_count, state, temp_c, sensor_ok, simulated, appliance_status,"
                    " ac_inferred_state, humidity, visual_status, ac_monitor, region_temp_1, region_temp_2,"
                    " region_temp_3, region_temp_4) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    con.commit()
    df = an.status_frame(st.query_status_full())
    s = an.summary(df)
    assert abs(s["occupied_pct"] - 50) < 2 and abs(s["vacant_pct"] - 50) < 2
    assert s["temp_min"] == 27.0 and s["temp_max"] == 29.0 and s["humidity_avg"] == 55
    assert abs(s["ac_on_vacant_h"] * 60 - 5) < 0.2                      # AC ran 5 min in the empty room
    relays = {1: {"name": "Light 1", "kind": "light"}, 3: {"name": "Fan", "kind": "fan", "watts": 75}}
    use = an.appliance_usage(df, relays).set_index("appliance")
    assert abs(use.loc["Fan", "on_h"] * 60 - 10) < 0.2 and abs(use.loc["Fan", "on_empty_h"] * 60 - 5) < 0.2
    assert abs(use.loc["Fan", "wasted_kwh"] - 75 * 5 / 60 / 1000) < 1e-3
    assert use.loc["Light 1", "on_empty_h"] == 0 and use.loc["Light 1", "watts"] == 40   # default watts
    cam = an.camera_waste(df).set_index("item")
    assert abs(cam.loc["Fan z1", "seen_on_empty_h"] * 60 - 5) < 0.2
    assert len(an.hourly_profile(df)) == 1


def test_analysis_handles_gaps_and_sessions():
    from core import analysis as an
    rows = [("2026-10-24 10:00:00", 1, "OCCUPIED", 29, 1, 0, "", "IDLE", None, "", "OFF", None, None, None, None),
            ("2026-10-24 10:00:05", 1, "OCCUPIED", 29, 1, 0, "", "IDLE", None, "", "OFF", None, None, None, None),
            ("2026-10-24 13:00:00", 1, "OCCUPIED", 29, 1, 0, "", "IDLE", None, "", "OFF", None, None, None, None)]   # app was stopped 3 h
    df = an.status_frame(rows)
    assert df["dur_s"].sum() <= 35                                       # the 3 h gap counts as 30 s at most
    ev = [("2026-10-24 10:00:00", "AC", "INFO", "AC detected ON: ..."), ("2026-10-24 10:12:00", "AC", "INFO", "AC detected OFF: ...")]
    sess = an.ac_sessions(ev)
    assert len(sess) == 1 and sess.loc[0, "minutes"] == 12.0
    assert an.summary(an.status_frame([]))["monitored_h"] == 0.0


def test_fan_zone_black_frame_is_unknown_not_on(tmp_path):
    from core.appliances import ApplianceTracker
    tr = ApplianceTracker({"zones_path": str(tmp_path / "z.json")})
    zid = tr.add_zone("fan", (80, 40, 240, 200))
    for i in range(10): tr.update_frame(_blade_frame(0, False), i * 0.1)
    tr.mark_zone(zid, "OFF")
    for i in range(10): tr.update_frame(_blade_frame(0, True), 1 + i * 0.1)
    tr.mark_zone(zid, "ON")
    black = np.zeros((240, 320), dtype=np.uint8)                      # camera covered / privacy shutter
    for i in range(20): tr.update_frame(black, 5 + i * 0.1)
    assert next(a["state"] for a in tr.snapshot() if a.get("zone")) == "UNKNOWN"


def test_analysis_old_rows_without_new_columns():
    from core import analysis as an
    rows = [("2026-10-24 10:00:00", 1, "OCCUPIED", 29, 1, 0, "Fan=ON", "IDLE", None, None, None, None, None, None, None),
            ("2026-10-24 10:00:05", 0, "VACANT", None, 0, 0, None, "IDLE", None, None, None, None, None, None, None)]
    df = an.status_frame(rows)
    assert an.camera_waste(df).empty and an.parse_states(float("nan")) == {}
    assert an.summary(df)["temp_min"] == 29
    assert an.appliance_usage(df, {3: {"name": "Fan", "kind": "fan"}}).loc[0, "on_h"] > 0


# --------------------------------------------- auto fan housing + glow lights
def _ceiling(seed=0):
    return (np.full((480, 640), 165, dtype=np.uint8)
            + np.random.default_rng(seed).integers(0, 4, (480, 640), dtype=np.uint8))


def _hanging_fan(img, cx=135, top=118, blades=True):
    import cv2
    cv2.rectangle(img, (cx - 3, top), (cx + 3, top + 35), 120, -1)            # down-rod
    cv2.ellipse(img, (cx, top + 48), (15, 10), 0, 0, 360, 110, -1)             # motor housing
    if blades:                                                                  # stopped: blades visible
        for ang in (10, 100, 190, 280):
            a = np.deg2rad(ang)
            cv2.line(img, (cx, top + 50), (int(cx + 60 * np.cos(a)), int(top + 50 + 26 * np.sin(a))), 90, 7)
    return img


def _run(tr, frame, passes=2, dets=lambda: []):
    t = 0.0
    for _ in range(passes):
        for _ in range(20):
            tr.update_frame(frame, t); t += 0.1
        tr.update_detections(dets(), t)
    for _ in range(20):
        tr.update_frame(frame, t); t += 0.1
    return {(a["cls"], a["state"]) for a in tr.snapshot()}


def test_unlit_lamp_is_a_fan_housing_on_when_blades_blurred_off_when_visible():
    from core.appliances import ApplianceTracker
    lamp = lambda: [("light", 0.45, (119, 115, 152, 172))]                       # what the model reports
    assert _run(ApplianceTracker({}), _hanging_fan(_ceiling(), blades=False), dets=lamp) == {("fan", "ON")}
    assert _run(ApplianceTracker({}), _hanging_fan(_ceiling(), blades=True), dets=lamp) == {("fan", "OFF")}


def test_glowing_lamp_stays_a_light():
    import cv2
    from core.appliances import ApplianceTracker
    img = _ceiling()
    cv2.circle(img, (345, 65), 18, 245, -1)
    assert _run(ApplianceTracker({}), img, dets=lambda: [("light", 0.3, (320, 40, 370, 92))]) == {("light", "ON")}


def test_glow_finder_lights_vs_reflections_windows_and_people():
    import cv2
    from core.vision import find_glow_spots
    img = _ceiling()
    cv2.circle(img, (345, 65), 16, 244, -1)                                     # lit light 1
    cv2.circle(img, (350, 145), 14, 244, -1)                                    # lit light 2
    cv2.ellipse(img, (356, 200), (14, 30), 0, 0, 360, 233, -1)                  # its dimmer reflection on the wall
    cv2.rectangle(img, (0, 160), (90, 280), 250, -1)                            # big bright window (too large)
    cv2.circle(img, (500, 400), 15, 250, -1)                                    # white shirt low in the picture
    boxes = find_glow_spots(img, {})
    centers = sorted(((b[0] + b[2]) // 2, (b[1] + b[3]) // 2) for b in boxes)
    assert len(boxes) == 2, boxes
    assert abs(centers[0][1] - 65) < 15 and abs(centers[1][1] - 145) < 20


def test_glow_and_model_light_on_same_spot_are_one_light():
    import cv2
    from core.appliances import ApplianceTracker
    from core.vision import find_glow_spots
    img = _ceiling()
    cv2.circle(img, (345, 65), 16, 244, -1)
    dets = lambda: [("light", 0.16, (326, 0, 363, 94))] + [("light", 0.9, b, "glow") for b in find_glow_spots(img, {})]
    assert sorted(_run(ApplianceTracker({}), img, dets=dets)) == [("light", "ON")]


def test_glow_light_switched_off_reads_off_and_stays_listed():
    import cv2
    from core.appliances import ApplianceTracker
    from core.vision import find_glow_spots
    lit = _ceiling(); cv2.circle(lit, (345, 65), 16, 244, -1)
    off = _ceiling(); cv2.circle(off, (345, 65), 16, 175, -1)                    # same fixture, not glowing
    tr = ApplianceTracker({"confirm_hits": 2})
    assert _run(tr, lit, dets=lambda: [("light", 0.9, b, "glow") for b in find_glow_spots(lit, {})]) == {("light", "ON")}
    for i in range(40):
        tr.update_frame(off, 100 + i * 0.1)
    tr.update_detections([("light", 0.9, b, "glow") for b in find_glow_spots(off, {})], 104.0)   # nothing glows now
    assert {(a["cls"], a["state"]) for a in tr.snapshot()} == {("light", "OFF")}


def test_unlit_model_lamp_that_is_not_a_hanging_fixture_is_dropped():
    from core.appliances import ApplianceTracker
    img = _ceiling()                                                           # nothing glows, nothing hangs
    assert _run(ApplianceTracker({}), img, dets=lambda: [("light", 0.3, (30, 140, 90, 190))]) == set()


def test_glow_light_is_not_duplicated_when_its_box_changes_size():
    import cv2
    from core.appliances import ApplianceTracker
    img = _ceiling(); cv2.circle(img, (345, 65), 16, 244, -1)
    sizes = iter([(325, 45, 365, 85), (315, 30, 378, 100), (330, 50, 360, 80), (318, 35, 372, 96)])
    tr = ApplianceTracker({"confirm_hits": 2})
    t = 0.0
    for box in sizes:
        for _ in range(10):
            tr.update_frame(img, t); t += 0.1
        tr.update_detections([("light", 0.9, box, "glow")], t)
    assert [a["cls"] for a in tr.snapshot()] == ["light"]


def test_white_shirt_on_a_person_is_not_a_light():
    import cv2
    from core.vision import find_glow_spots
    img = _ceiling(); cv2.circle(img, (500, 260), 12, 246, -1)                 # bright shirt, inside the top 60 %
    assert find_glow_spots(img, {}) and not find_glow_spots(img, {}, exclude=[(470, 220, 600, 420)])


def test_overlapping_light_tracks_merge_into_one():
    import cv2
    from core.appliances import ApplianceTracker
    img = _ceiling(); cv2.circle(img, (400, 60), 16, 244, -1)
    tr = ApplianceTracker({"confirm_hits": 1})
    t = 0.0
    for box in [(380, 40, 420, 80), (385, 10, 415, 140), (370, 30, 430, 100)]:   # glow splits/merges differently
        for _ in range(10):
            tr.update_frame(img, t); t += 0.1
        tr.update_detections([("light", 0.9, box, "glow")], t)
    assert [(a["cls"], a["state"]) for a in tr.snapshot()] == [("light", "ON")]


def test_overexposed_light_is_lit_not_a_fan():
    import cv2
    from core.appliances import ApplianceTracker
    img = np.full((480, 640), 178, dtype=np.uint8)                              # bright, over-exposed ceiling
    cv2.rectangle(img, (360, 0), (392, 110), 222, -1)                           # light merged into a bright streak
    cv2.rectangle(img, (370, 20), (380, 60), 236, -1)
    assert _run(ApplianceTracker({}), img, dets=lambda: [("light", 0.2, (357, 0, 397, 153))]) == {("light", "ON")}



# ------------------------------------------------ report section 3.4: 4-region delta T
def test_report_formula_four_regions():
    """dT_i = |T_i,final - T_i,init|, mean over the 4 regions, AC active if mean > threshold."""
    ac = ACInference(observation_s=120, cooling_threshold_c=1.0)              # default mode = report ("absolute")
    ac.start([29.0, 29.5, 28.5, 30.0], True, 0)
    for t in range(5, 121, 5):                                                # a reading every 5 s
        k = t / 120
        ac.update([29.0 - 3.0 * k, 29.5 - 2.0 * k, 28.5 - 1.0 * k, 30.0 - 0.4 * k], True, float(t))
    assert ac.status == ACStatus.AC_ACTIVE
    deltas = {i: r["delta"] for i, r in ac.regions.items()}
    assert set(deltas) == {1, 2, 3, 4} and deltas[1] > deltas[2] > deltas[3] > deltas[4] > 0
    assert abs(ac.delta_t - sum(deltas.values()) / 4) < 0.01                  # mean of the regional deltas
    assert ac.delta_signed < 0                                                # the room really cooled


def test_report_formula_small_change_is_inactive_and_missing_regions_are_skipped():
    ac = ACInference(observation_s=120, cooling_threshold_c=1.0)
    ac.start([29.0, None, None, None], True, 0)                               # only one DHT11 connected
    for t in range(5, 121, 5):
        ac.update([29.0 - 0.5 * t / 120, None, None, None], True, float(t))
    assert ac.status == ACStatus.AC_INACTIVE and list(ac.regions) == [1]      # mean over the regions that exist
    assert ac.delta_t <= 1.0


def test_parse_four_region_telemetry():
    m = parse_line('{"type":"telemetry","temps":[29.0,28.0,null,200],"hums":[55,54,null,null],'
                   '"temp_c":28.5,"humidity":55,"sensor_ok":true,"relay1":0}')
    assert m["temps"] == [29.0, 28.0, None, None]                             # 200 C is impossible -> missing
    assert m["temp_c"] == 28.5 and m["sensor_ok"] is True
    old = parse_line('{"type":"telemetry","temp_c":27.0,"sensor_ok":true}')   # older firmware still works
    assert old["temps"] == [27.0]
    dead = parse_line('{"type":"telemetry","temps":[null,null,null,null],"sensor_ok":false}')
    assert dead["sensor_ok"] is False and dead["temp_c"] is None



def test_vacancy_needs_consecutive_empty_frames():
    m = OccupancyMachine(vacancy_confirm_s=0, occupied_confirm_frames=1, fail_timeout_s=30, vacancy_confirm_frames=5)
    m.update(2, 0.0)
    for i in range(1, 5):
        assert m.update(0, i * 0.5) == State.VACANCY_PENDING                  # 4 empty frames: not yet
    assert m.update(0, 2.5) == State.VACANT                                    # 5th consecutive empty frame
    m2 = OccupancyMachine(vacancy_confirm_s=0, occupied_confirm_frames=1, fail_timeout_s=30, vacancy_confirm_frames=5)
    m2.update(1, 0.0)
    for i in range(1, 4):
        m2.update(0, i * 0.5)
    m2.update(1, 2.0)                                                          # a person breaks the streak
    for i in range(5, 9):
        assert m2.update(0, i * 0.5) == State.VACANCY_PENDING                 # counting starts again
    assert m2.update(0, 4.5) == State.VACANT


def test_same_frame_seen_twice_is_not_consecutive():
    m = OccupancyMachine(vacancy_confirm_s=0, occupied_confirm_frames=1, fail_timeout_s=30, vacancy_confirm_frames=3)
    m.update(1, 0.0)
    m.update(0, 0.5)
    for t in (1.0, 1.5, 2.0, 2.5):
        assert m.update(0, t, fresh=False) == State.VACANCY_PENDING            # no new frame from the camera
    m.update(0, 3.0); assert m.update(0, 3.5) == State.VACANT


def test_automation_waits_before_switching_off(tmp_path):
    e, v, l = make_engine(tmp_path, automation=True)
    e.cfg["automation"]["off_after_vacant_s"] = 30
    e.step(0); v.count = 0; e.step(1); e.step(62)                              # VACANT at 62 s
    assert l.relays[1] == 1                                                    # not yet: vacant for 0 s
    e.step(80); assert l.relays[1] == 1
    e.step(93); assert l.relays[1] == 0                                        # vacant 31 s -> switched off



def test_region_temps_are_stored_like_report_table_4_2(tmp_path):
    from core import analysis as an
    from core.storage import Storage
    st = Storage(str(tmp_path / "r.db"))
    st.log_status(0, "VACANT", 28.6, True, False, "", "OBSERVING", region_temps=[28.0, 29.0, None, 28.8])
    st.log_status(0, "VACANT", 28.4, True, False, "", "OBSERVING", region_temps=[27.6])   # 1 sensor: rest empty
    rows = st.query_status_full()
    assert rows[1][-4:] == (28.0, 29.0, None, 28.8) and rows[0][-4:] == (27.6, None, None, None)
    s = an.summary(an.status_frame(rows))
    assert set(s["region_avg"]) == {"r1", "r2", "r4"} and abs(s["region_avg"]["r1"] - 27.8) < 1e-6


def test_report_flow_end_to_end_four_regions(tmp_path):
    """Report 3.4: consecutive empty frames -> vacant -> T_init (4 regions) -> 2 min -> T_final -> mean dT -> alert."""
    e, v, l = make_engine(tmp_path, automation=False)
    e.occ.vacancy_confirm_s, e.occ.vacancy_confirm_frames = 0, 10
    e.ac = ACInference(observation_s=120, cooling_threshold_c=1.0)            # report formula
    base = [29.0, 29.5, 28.5, 30.0]
    l.region = list(base)
    l_latest = type(l).latest
    type(l).latest = property(lambda self: {**l_latest.fget(self), "temps": list(self.region)})
    try:
        v.count = 3; e.step(0)
        v.count = 0
        t = 0.5
        while e.occ.state.value != "VACANT":
            e.step(t); t += 0.5
        assert 4.5 <= t <= 6.0                                                  # 10 consecutive empty frames
        t0 = t
        while t < t0 + 125:                                                     # AC running: every region cools
            k = (t - t0) / 120
            l.region = [round(b - d * k, 1) for b, d in zip(base, (3.0, 2.0, 1.5, 1.0))]
            e.step(t); t += 5
        assert e.ac.status == ACStatus.AC_ACTIVE and set(e.ac.regions) == {1, 2, 3, 4}
        assert e.ac.delta_t > 1.0 and "AC_LEFT_ON" in e.alerts.active
        assert "Turn OFF the AC" in e.alerts.active["AC_LEFT_ON"].title
    finally:
        type(l).latest = l_latest


# ------------------------------------------------------------------ evaluation metrics
def test_eval_perfect_predictions():
    from core.evaluation import CLASSES, evaluate
    ims = [{"name": "a", "gts": [(0, [10, 10, 50, 90]), (3, [200, 5, 240, 40])],
            "preds": [(0, 0.9, [10, 10, 50, 90]), (3, 0.8, [200, 5, 240, 40])]}]
    r = evaluate(ims, CLASSES)
    assert r["summary"]["map50"] == 1.0 and r["summary"]["map50_95"] == 1.0
    assert r["per_class"]["person"]["precision"] == 1.0 and r["per_class"]["light_on"]["recall"] == 1.0
    m = r["confusion"]["matrix"]
    assert m[0][0] == 1 and m[3][3] == 1 and sum(map(sum, m)) == 2


def test_eval_ap_hand_computed():
    """2 true boxes; predictions TP(0.9), FP(0.8), TP(0.7): COCO 101-point AP = (51*1 + 50*2/3) / 101."""
    from core.evaluation import evaluate
    ims = [{"name": "a", "gts": [(0, [0, 0, 10, 10]), (0, [100, 100, 110, 110])],
            "preds": [(0, 0.9, [0, 0, 10, 10]), (0, 0.8, [50, 50, 60, 60]), (0, 0.7, [100, 100, 110, 110])]}]
    r = evaluate(ims, ["person"])["per_class"]["person"]
    assert abs(r["ap50"] - (51 + 50 * 2 / 3) / 101) < 1e-6
    assert r["tp"] == 2 and r["fp"] == 1 and r["fn"] == 0 and abs(r["precision"] - 2 / 3) < 1e-9


def test_eval_confusion_misses_false_alarms_and_wrong_state():
    from core.evaluation import CLASSES, confusion_matrix, merge_states
    ims = [{"name": "a",
            "gts": [(0, [0, 0, 50, 100]), (2, [200, 0, 260, 60]), (4, [400, 0, 440, 40])],      # person, fan_off, light_off
            "preds": [(1, 0.9, [202, 2, 258, 58]),                                               # fan_on on the fan_off
                      (3, 0.9, [500, 300, 540, 340])]}]                                          # light_on on nothing
    m = confusion_matrix(ims, CLASSES)["matrix"]
    bg = len(CLASSES)
    assert m[1][2] == 1                 # predicted fan_on, truly fan_off  (wrong state)
    assert m[bg][0] == 1 and m[bg][4] == 1   # person and light_off missed
    assert m[3][bg] == 1                # light_on false alarm
    merged, names, state = merge_states(ims)
    assert names == ["person", "fan", "light"] and state["matched"] == 1 and state["accuracy"] == 0.0


def test_eval_iou_strictness_and_yolo_label_round_trip(tmp_path):
    from core.evaluation import evaluate, read_yolo_labels, write_yolo_labels
    ims = [{"name": "a", "gts": [(0, [0, 0, 100, 100])], "preds": [(0, 0.9, [0, 0, 100, 140])]}]   # IoU = 0.714
    r = evaluate(ims, ["person"])["per_class"]["person"]
    assert r["ap50"] == 1.0 and 0 < r["ap50_95"] < 1.0       # counts at 0.5, fails the strict thresholds
    write_yolo_labels(tmp_path / "x.txt", [(2, [10, 20, 110, 70])], 640, 480)
    assert read_yolo_labels(tmp_path / "x.txt", 640, 480)[0][0] == 2
    assert all(abs(a - b) < 0.01 for a, b in zip(read_yolo_labels(tmp_path / "x.txt", 640, 480)[0][1], [10, 20, 110, 70]))


def test_camera_movement_clears_old_fans_and_lights_but_exposure_does_not():
    import cv2
    from core.appliances import ApplianceTracker
    img = _ceiling(); cv2.circle(img, (345, 65), 16, 244, -1)
    for x in range(0, 640, 80):                                                  # room-like big shapes
        for y in range(160, 480, 80):
            if (x // 80 + y // 80) % 2:
                img[y:y + 80, x:x + 80] = 95
    tr = ApplianceTracker({"confirm_hits": 2})
    tr.update_frame(img, 0.0)
    tr.update_detections([("light", 0.9, (325, 45, 365, 85), "glow")], 0.1)
    assert [a["state"] for a in tr.snapshot()] in (["UNKNOWN"], ["ON"])          # glow light listed at once
    brighter = np.clip(img.astype(int) + 30, 0, 255).astype(np.uint8)
    for i in range(1, 6):
        tr.update_frame(brighter, i * 0.3)                                       # auto-exposure change: keep
    assert len(tr.snapshot()) == 1
    panned = np.roll(img, 60, axis=1)
    for i in range(6, 10):
        tr.update_frame(panned, i * 0.3)                                         # camera turned: start again
    assert tr.snapshot() == [] and tr.camera_moves == 1



def test_exposure_controller_steps_toward_target():
    from core.vision import next_exposure
    assert next_exposure(-5, 200, (70, 150), -8, -1) == -6        # too bright (face close): darker
    assert next_exposure(-5, 40, (70, 150), -8, -1) == -4         # too dark (lights off): brighter
    assert next_exposure(-5, 110, (70, 150), -8, -1) == -5        # in range: keep
    assert next_exposure(-8, 220, (70, 150), -8, -1) == -8        # limits respected
    assert next_exposure(-1, 20, (70, 150), -8, -1) == -1


def test_eval_micro_average_pools_boxes():
    from core.evaluation import evaluate
    ims = [{"name": "a", "gts": [(0, [i * 20, 0, i * 20 + 10, 10]) for i in range(9)] + [(1, [300, 0, 310, 10])],
            "preds": [(0, 0.9, [i * 20, 0, i * 20 + 10, 10]) for i in range(9)] + [(1, 0.9, [400, 0, 410, 10])]}]
    s = evaluate(ims, ["a", "b"])["summary"]
    assert s["precision"] == 0.5                      # macro: class a 100 %, class b 0 %
    assert abs(s["micro_precision"] - 0.9) < 1e-9     # micro: 9 of 10 boxes right
