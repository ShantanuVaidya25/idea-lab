"""Live dashboard: camera, room status, temperature and AC, appliance control."""
from __future__ import annotations

from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

from core.config import load_config, save_config
from core.ui import (STATE_TEXT, STATE_TONE, banner, card, esc, get_engine, kpi, kpi_row, page_head, pill,
                     region_grid, section)

eng = get_engine()
cfg = eng.cfg
AC_RESULT = {"AC_ACTIVE": "AC active", "AC_INACTIVE": "AC not active", "COOLING_OBSERVED": "Cooling observed", "NO_SIGNIFICANT_COOLING_OBSERVED": "No cooling",
             "INCONCLUSIVE": "Inconclusive", "SENSOR_ERROR": "Sensor error", "OBSERVING": "Observing", "IDLE": "Idle"}


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### Operation")
    auto = st.toggle("Automation", value=eng.automation_enabled,
                     help="Off by default. When on, lights and fan switch off after the room is confirmed empty.")
    if auto != eng.automation_enabled:
        eng.set_automation(auto)
    ovr = st.toggle("Manual override", value=eng.manual_override,
                    help="While on, automation never changes relays. Using a relay button turns this on.")
    if ovr != eng.manual_override:
        eng.set_manual_override(ovr)

    st.markdown("### Camera")
    v = cfg["vision"]
    mode = st.segmented_control("Source", ["Webcam", "Simulated"], default="Simulated" if v.get("simulate") else "Webcam")
    cur_src = int(v["source"]) if str(v.get("source", 0)).isdigit() else 0
    cam_opts = [0, 1, 2]
    cam_idx = cam_opts.index(cur_src) if cur_src in cam_opts else 0
    cam = st.selectbox("Camera index", cam_opts, index=cam_idx,
                       help="0 = built-in webcam, 1 = external USB camera")
    mpath = st.text_input("Person model", v["model_path"])
    conf = st.slider("Detection confidence", 0.1, 0.9, float(v["confidence"]), 0.05)
    if st.button("Apply camera settings", use_container_width=True):
        eng.set_vision(mode == "Simulated", int(cam), mpath.strip(), conf)
        st.rerun()

    st.markdown("### Controller (ESP32)")
    try:
        from core.esp32 import list_serial_ports
        found = [p for p in list_serial_ports() if not p["bluetooth"]]
    except Exception:
        found = []
    labels = {f"{p['device']}  {p['description']}" + ("  (ESP32)" if p["likely_esp32"] else ""): p["device"] for p in found}
    options = ["Simulator", "Auto-detect"] + list(labels)
    cur = cfg["esp32"].get("port") or ""
    idx = 1 if str(cur).lower() == "auto" else next((i + 2 for i, d in enumerate(labels.values()) if d == cur), 0)
    choice = st.selectbox("Port", options, index=idx)
    if not any(p["likely_esp32"] for p in found):
        st.caption("No ESP32 detected. Connect it with a data USB cable and install the CH340 / CP210x driver.")
    c1, c2 = st.columns(2)
    if c1.button("Connect", use_container_width=True, type="primary"):
        eng.set_esp32_port("" if choice == "Simulator" else "auto" if choice == "Auto-detect" else labels[choice])
        st.rerun()
    if c2.button("Rescan", use_container_width=True):
        st.rerun()

    if eng.link.simulated or cfg["vision"].get("simulate"):
        st.markdown("### Simulation")
    if cfg["vision"].get("simulate"):
        eng.vision.person_count = st.number_input("People in room", 0, 50, eng.vision.person_count)
    if eng.link.simulated:
        eng.link.ac_running = st.checkbox("AC running", value=eng.link.ac_running)
        eng.link.sensor_fault = st.checkbox("DHT11 fault", value=eng.link.sensor_fault)

    st.markdown("### Notifications")
    tg = cfg["alerts"]["telegram"]
    st.caption(f"Desktop: {'on' if cfg['alerts'].get('desktop_popup') else 'off'}  ·  "
               f"Telegram: {'on' if tg.get('enabled') and tg.get('bot_token') else 'off'}")
    st.caption("Sent only for: turn-off reminders, AC switched on, AC switched off.")


# ------------------------------------------------------------------ header + live status
@st.fragment(run_every=1)
def header_and_status() -> None:
    s = eng.snapshot()
    vis = eng.vision.snapshot()
    sim_cam = cfg["vision"].get("simulate")
    cam_pill = (pill("Camera simulated", "gray") if sim_cam else pill("Camera live", "green") if vis.ok
                else pill("Camera starting" if getattr(eng.vision, "loading", False) else "Camera error",
                          "amber" if getattr(eng.vision, "loading", False) else "red"))
    esp_pill = (pill("Controller simulated", "gray") if s["simulated_esp"] else
                pill(f"Controller {s['link_port']}", "green") if s["connected"] else pill("Controller offline", "red"))
    auto_pill = (pill("Manual override", "blue") if s["manual_override"] else
                 pill("Automation on", "green") if s["automation"] else pill("Automation off", "gray"))
    page_head("Smart Classroom", "Occupancy, energy and climate monitoring", cam_pill + esp_pill + auto_pill)

    for a in sorted((x for x in s["active_alerts"] if x.code in eng.alerts.notify_codes),
                    key=lambda x: x.level != "CRITICAL"):
        banner(a.title, a.message, a.time_str, a.level)
    if s["occ_state"] == "VACANCY_PENDING":
        got, need = s["empty_frames"]
        banner("Nobody detected", f"Confirming vacancy: {min(got, need)} of {need} consecutive empty frames. "
               f"A person in any frame cancels it.", level="INFO")
    if not s["connected"] and not s["simulated_esp"] and s["link_error"]:
        banner("Controller connection problem", s["link_error"], level="WARNING")

    acm = s["acmon"]
    ac_state = acm["state"]
    ac_val = {"ON": "On", "OFF": "Off"}.get(ac_state, "Unknown")
    ac_sub = (f"Trend {acm['slope_c_min']:+.2f} °C/min" if ac_state != "UNKNOWN" else
              f"Collecting readings ({acm['samples']})")
    temp_sub = (f"Humidity {s['humidity']:.0f} %" if s.get("humidity") is not None else
                "Simulated sensor" if s["simulated_esp"] else "DHT11")
    on_names = [a["cls"].capitalize() for a in vis.appliances or [] if a.get("state") == "ON"]
    room = "INITIALIZING" if s["state"] == "ERROR" and getattr(eng.vision, "loading", False) else s["state"]
    kpi_row([
        kpi("Room", STATE_TEXT.get(room, room), "", STATE_TONE.get(room, "gray")),
        kpi("People", vis.person_count if vis.ok else "–", "in camera view", "green" if vis.ok and vis.person_count else "gray"),
        kpi("Temperature", f"{s['temp']:.1f} °C" if s["temp"] is not None else "–", temp_sub,
            "blue" if s["temp"] is not None else "red"),
        kpi("AC", ac_val, ac_sub, "blue" if ac_state == "ON" else "gray"),
        kpi("Seen on", ", ".join(on_names) if on_names else "Nothing", "fans / lights in view", "amber" if on_names else "gray"),
    ])


header_and_status()


# ------------------------------------------------------------------ camera + detected appliances
@st.fragment(run_every=0.3)
def video_fallback() -> None:
    vis = eng.vision.snapshot()
    if vis.frame is not None:
        st.image(vis.frame, use_container_width=True,
                 caption="Simulated camera" if cfg["vision"].get("simulate") else None)
    else:
        st.info(vis.error or "Waiting for camera")


@st.fragment(run_every=1)
def detections() -> None:
    vis = eng.vision.snapshot()
    rows = []
    for a in vis.appliances or []:
        tone = {"ON": "green", "OFF": "gray"}.get(a["state"], "amber")
        txt = {"ON": "On", "OFF": "Off"}.get(a["state"], "Measuring")
        rows.append((f"{a['cls'].capitalize()} {a.get('id', '')}", pill(txt, tone)))
    note = ("Fan: on when the blades move or blur. Light: on when it glows. Visual estimate; relay state comes "
            "from the controller.")
    st.markdown(card("Detected fans and lights", rows or [("None in view", "")], note), unsafe_allow_html=True)
    sn = eng.snapshot()
    st.markdown('<div class="card"><h4>Regional temperature</h4>' + region_grid(sn["region_temps"], sn["ac"]["regions"])
                + '<div class="muted" style="margin-top:8px">One DHT11 per region. ΔT appears during the '
                  'post-vacancy check.</div></div>', unsafe_allow_html=True)
    if not cfg["vision"].get("simulate"):
        st.markdown(card("Camera", [("Processor", esc(eng.vision.device)), ("Detector rate", f"{vis.fps} fps"),
                                    ("Model", esc(cfg['vision']['model_path']))]), unsafe_allow_html=True)
    for w in eng.vision.warnings:
        st.caption(w)


section("Live view")
left, right = st.columns([2.2, 1])
with left:
    vis = eng.vision.snapshot()
    if not vis.ok and getattr(eng.vision, "loading", False):
        st.info("⌛ Loading YOLO model and camera...")
    elif not vis.ok and vis.error:
        st.error(f"⚠️ {vis.error}")
        st.caption("Tip: Select **Camera index: 0** in the sidebar, then click **Apply camera settings**.")
    elif getattr(eng.vision, "stream_url", None):
        url = eng.vision.stream_url
        st.markdown(
            f'<img class="video" src="{url}" '
            f'onerror="this.onerror=null; setTimeout(() => {{ this.src = \'{url}?t=\' + Date.now(); }}, 1000);" />',
            unsafe_allow_html=True,
        )
    else:
        video_fallback()
with right:
    detections()


# ------------------------------------------------------------------ tabs
tab_temp, tab_ctrl, tab_cal, tab_set = st.tabs(["Temperature and AC", "Appliances", "Calibration", "Settings"])


@st.fragment(run_every=2)
def temperature_tab() -> None:
    s = eng.snapshot()
    acm, ac = s["acmon"], s["ac"]
    c1, c2 = st.columns([2.2, 1])
    with c1:
        rows = [(t, f"Region {i + 1}", v) for t, temps in s["region_history"] for i, v in enumerate(temps) if v is not None]
        if not rows:
            rows = [(t, "Room", v) for t, v in s["history"]]
        if rows:
            df = pd.DataFrame(rows, columns=["time", "region", "temp_c"])
            df["time"] = pd.to_datetime(df["time"], unit="s")
            lo, hi = df["temp_c"].min(), df["temp_c"].max()
            chart = (alt.Chart(df).mark_line(strokeWidth=2, interpolate="step-after")
                     .encode(x=alt.X("time:T", title=None, axis=alt.Axis(format="%H:%M:%S", grid=False)),
                             y=alt.Y("temp_c:Q", title="°C", scale=alt.Scale(domain=[lo - 1, hi + 1])),
                             color=alt.Color("region:N", title=None, scale=alt.Scale(
                                 domain=["Region 1", "Region 2", "Region 3", "Region 4", "Room"],
                                 range=["#1D4ED8", "#12B76A", "#F79009", "#7A5AF8", "#1D4ED8"])),
                             tooltip=["region", alt.Tooltip("time:T", format="%H:%M:%S"), alt.Tooltip("temp_c:Q", title="°C")])
                     .properties(height=280))
            st.altair_chart(chart.configure_view(strokeWidth=0).configure_axis(labelColor="#667085",
                            titleColor="#667085", domainColor="#E4E7EC", tickColor="#E4E7EC", gridColor="#F2F4F7")
                            .configure_legend(orient="top", labelColor="#475467"), use_container_width=True)
        else:
            st.info("No temperature readings yet.")

        # report section 3.4: regional delta T after vacancy
        regs = ac["regions"]
        formula = "|T final − T init|" if ac["mode"] == "absolute" else "T final − T init"
        st.markdown('<div class="section">Post-vacancy check, report section 3.4</div>', unsafe_allow_html=True)
        table = pd.DataFrame([{"Region": f"Region {i}", "T init (°C)": r["init"], "T final (°C)": r["final"],
                               f"ΔT = {formula} (°C)": r["delta"]} for i, r in sorted(regs.items())])
        if not table.empty:
            st.dataframe(table, hide_index=True, use_container_width=True)
        else:
            st.caption("Starts when the room is confirmed vacant. Every region's temperature is recorded at t = 0 "
                       f"and again after {int(eng.ac.observation_s)} s.")
    with c2:
        mean = (ac["delta_t"] if ac["delta_t"] is not None else ac["delta_live"])
        tone = "red" if ac["status"] in ("AC_ACTIVE", "COOLING_OBSERVED") else "gray"
        st.markdown(card("Post-vacancy AC check (ΔT)", [
            ("Result", pill(AC_RESULT.get(ac["status"], ac["status"]), tone)),
            ("Mean ΔT over regions", (f"{mean:.2f} °C" + ("" if ac["delta_t"] is not None else " (live)"))
             if mean is not None else "–"),
            ("Threshold δ", f"{ac['threshold']} °C"),
            ("Regions measured", str(len(regs)) if regs else "–"),
            ("Time left", f"{int(ac['seconds_left'])} s" if ac["seconds_left"] is not None else "–"),
        ], f"AC is judged active when the mean ΔT is above δ while the room is vacant "
           f"(formula: ΔT = {formula}, averaged over the regions)."), unsafe_allow_html=True)
        tone = {"ON": "blue", "OFF": "gray"}.get(acm["state"], "amber")
        st.markdown(card("Continuous AC trend", [
            ("State", pill({"ON": "On", "OFF": "Off"}.get(acm["state"], "Unknown"), tone)),
            ("Trend", f"{acm['slope_c_min']:+.2f} °C/min"),
            ("Change in window", f"{-acm['drop_c']:+.1f} °C"),
            ("Window", f"{int(eng.acmon.window_s)} s, {acm['samples']} readings"),
        ], "Runs all the time (also while the room is occupied) and gives the AC on / off alerts."),
            unsafe_allow_html=True)


@st.fragment(run_every=1)
def appliances_tab() -> None:
    s = eng.snapshot()
    cols = st.columns(len(eng.relay_cfg))
    for col, (ch, rc) in zip(cols, eng.relay_cfg.items()):
        with col:
            on = s["reported"][ch] == 1
            st.markdown(card(rc["name"], [
                ("Relay", str(ch)),
                ("State", pill("On" if on else "Off", "green" if on else "gray")),
                ("Mode", "Automatic" if rc.get("auto") else "Manual only"),
                ("Commanded", "On" if s["commanded"][ch] else "Off"),
            ]), unsafe_allow_html=True)
            if st.button("Turn off" if on else "Turn on", key=f"r{ch}", disabled=not s["connected"],
                         use_container_width=True):
                if not eng.manual_set(ch, 0 if on else 1):
                    st.error("Command failed")
                st.rerun(scope="fragment")
    st.caption("Using a relay button switches on manual override, so automation does not undo it.")


def calibration_tab() -> None:
    tr = getattr(eng.vision, "tracker", None)
    if tr is None:
        st.info("Calibration is available when the webcam is the camera source.")
        return
    st.markdown("A fast fan blurs its blades away, so it looks different when it is on. Calibrate each fan once "
                "with the camera in its final position: save the zones with the fan stopped, then record how it "
                "looks off and on.")
    if st.button("Save detected fans and lights as zones"):
        n = tr.zones_from_tracks()
        st.success(f"Saved {n} zone(s)." if n else "Nothing new to save.")
    zones = tr.zone_info()
    fmt = lambda x: "–" if x is None else f"{x:.2f}"
    for z in zones:
        c1, c2, c3, c4 = st.columns([2, 2, 2, 3])
        c1.markdown(f"**{z['cls'].capitalize()}**  `{z['id']}`")
        if z["cls"] == "fan":
            c4.caption(f"Blade visibility: off {fmt(z.get('off'))}, on {fmt(z.get('on'))}, now {fmt(z.get('dark_now'))}")
            if c2.button("Fan is off now", key=f"off_{z['id']}"):
                st.success("Saved.") if tr.mark_zone(z["id"], "OFF") else st.error("No video yet.")
            if c3.button("Fan is on now", key=f"on_{z['id']}"):
                st.success("Saved.") if tr.mark_zone(z["id"], "ON") else st.error("No video yet.")
        else:
            c4.caption("Judged by glow, no calibration needed.")
    with st.form("add_zone"):
        st.caption("Add a zone by hand (picture coordinates, 640 x 480)")
        f1, f2, f3, f4, f5 = st.columns(5)
        cls = f1.selectbox("Type", ["fan", "light"])
        x1 = f2.number_input("x1", 0, 640, 320)
        y1 = f3.number_input("y1", 0, 480, 6)
        x2 = f4.number_input("x2", 0, 640, 480)
        y2 = f5.number_input("y2", 0, 480, 112)
        if st.form_submit_button("Add zone"):
            tr.add_zone(cls, (int(x1), int(y1), int(x2), int(y2)))
            st.rerun()
    if zones and st.button("Delete all zones"):
        tr.clear_zones()
        st.rerun()


def settings_tab() -> None:
    st.caption("Saved to config.yaml. Restart the app to apply.")
    new = load_config()
    c1, c2 = st.columns(2)
    with c1:
        new["esp32"]["port"] = st.text_input("ESP32 port (empty = simulator, auto = detect)", new["esp32"]["port"])
        new["vision"]["source"] = st.text_input("Camera source (0, 1, file or URL)", str(new["vision"]["source"]))
        new["vision"]["simulate"] = st.checkbox("Simulated camera", new["vision"].get("simulate", False))
        new["occupancy"]["vacancy_confirm_frames"] = st.number_input(
            "Vacancy: consecutive empty frames", 1, 500, int(new["occupancy"].get("vacancy_confirm_frames", 20)))
        new["occupancy"]["vacancy_confirm_s"] = st.number_input("Vacancy: minimum time (s)", 0, 3600,
                                                                int(new["occupancy"]["vacancy_confirm_s"]))
        new["alerts"]["visual_waste_delay_s"] = st.number_input("Turn-off reminder after nobody seen (s)", 1, 600,
                                                                int(new["alerts"].get("visual_waste_delay_s", 8)))
    with c2:
        new["ac_inference"]["observation_s"] = st.number_input("Post-vacancy timer Tvac (s)", 10, 1800,
                                                               int(new["ac_inference"]["observation_s"]))
        new["ac_inference"]["delta_threshold_c"] = st.number_input(
            "Threshold δ (°C)", 0.1, 10.0, float(new["ac_inference"].get("delta_threshold_c", 1.0)), 0.1)
        modes = ["absolute", "signed"]
        new["ac_inference"]["delta_mode"] = st.selectbox(
            "ΔT formula", modes, index=modes.index(new["ac_inference"].get("delta_mode", "absolute")),
            format_func=lambda m: "|T final − T init| (report)" if m == "absolute" else "T final − T init (cooling only)")
        mon = new.setdefault("ac_monitor", {})
        mon["window_s"] = st.number_input("AC trend window (s)", 60, 1800, int(mon.get("window_s", 180)))
        mon["on_drop_c"] = st.number_input("AC on: minimum drop (°C)", 0.5, 5.0, float(mon.get("on_drop_c", 1.0)), 0.5)
        mon["off_rise_c"] = st.number_input("AC off: minimum rise (°C)", 0.5, 5.0, float(mon.get("off_rise_c", 1.0)), 0.5)
        new["alerts"]["telegram"]["enabled"] = st.checkbox("Telegram notifications", new["alerts"]["telegram"]["enabled"])
        new["alerts"]["telegram"]["bot_token"] = st.text_input("Telegram bot token", new["alerts"]["telegram"]["bot_token"],
                                                               type="password")
        new["alerts"]["telegram"]["chat_id"] = st.text_input("Telegram chat id", str(new["alerts"]["telegram"]["chat_id"]))
    if st.button("Save settings", type="primary"):
        src = new["vision"]["source"]
        new["vision"]["source"] = int(src) if str(src).isdigit() else src
        save_config(new)
        st.success(f"Saved at {datetime.now():%H:%M:%S}. Restart the app to apply.")


with tab_temp:
    temperature_tab()
with tab_ctrl:
    appliances_tab()
with tab_cal:
    calibration_tab()
with tab_set:
    settings_tab()
