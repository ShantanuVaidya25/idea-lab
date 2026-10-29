"""Analysis: how the room was used, where energy was wasted, and how temperature and the AC behaved."""
from __future__ import annotations

from datetime import datetime, timedelta

import altair as alt
import pandas as pd
import streamlit as st

from core import analysis as an
from core.config import load_config
from core.ui import esc, get_storage, kpi, kpi_row, page_head, section

cfg = load_config()
store = get_storage()
tariff = float(cfg.get("analysis", {}).get("tariff_per_kwh", 8.0))
currency = cfg.get("analysis", {}).get("currency", "INR")
AXIS = dict(labelColor="#667085", titleColor="#667085", domainColor="#E4E7EC", tickColor="#E4E7EC", gridColor="#F2F4F7")


def styled(chart: alt.Chart) -> alt.Chart:
    return chart.configure_view(strokeWidth=0).configure_axis(**AXIS).configure_legend(
        labelColor="#475467", titleColor="#667085", orient="top")


def fmt_h(hours: float) -> str:
    m = int(round(hours * 60))
    return f"{m} min" if m < 60 else f"{m // 60} h {m % 60:02d} min"


page_head("Analysis", "Usage, energy waste, temperature and AC behaviour from the stored history")
period = st.segmented_control("Period", ["Last hour", "Today", "Last 24 hours", "Last 7 days", "All time"],
                              default="Today", label_visibility="collapsed")
now = datetime.now()
since_dt = {"Last hour": now - timedelta(hours=1), "Today": now.replace(hour=0, minute=0, second=0, microsecond=0),
            "Last 24 hours": now - timedelta(hours=24), "Last 7 days": now - timedelta(days=7)}.get(period or "Today")
since = since_dt.strftime("%Y-%m-%d %H:%M:%S") if since_dt else None

df = an.status_frame(store.query_status_full(since))
events = store.query_events(since=since, limit=100000)
if df.empty:
    st.info("No data recorded in this period yet. Data is saved every few seconds while the dashboard runs.")
    st.stop()

s = an.summary(df)
usage = an.appliance_usage(df, cfg["relays"])
cam = an.camera_waste(df)
kwh, wasted = usage["kwh"].sum(), usage["wasted_kwh"].sum()
temp_range = f"{s['temp_min']:.0f}–{s['temp_max']:.0f} °C" if s["temp_min"] is not None else "no readings"

kpi_row([
    kpi("Monitored", fmt_h(s["monitored_h"]), f"{df['time'].min():%d %b %H:%M} to {df['time'].max():%H:%M}", "gray"),
    kpi("Occupied", f"{s['occupied_pct']:.0f} %", f"peak {s['peak_people']} people", "green"),
    kpi("Energy used", f"{kwh:.2f} kWh", f"{currency} {kwh * tariff:.2f} at {tariff:g}/kWh", "blue"),
    kpi("Energy wasted", f"{wasted:.2f} kWh", f"{currency} {wasted * tariff:.2f}, room empty", "red" if wasted > 0 else "gray"),
    kpi("AC runtime", fmt_h(s["ac_on_h"]), f"{fmt_h(s['ac_on_vacant_h'])} with the room empty",
        "amber" if s["ac_on_vacant_h"] > 0 else "gray"),
    kpi("Temperature", f"{s['temp_avg']:.1f} °C" if s["temp_avg"] is not None else "–", temp_range, "blue"),
])

# ---------------------------------------------------------------- insights
notes = []
top = usage.sort_values("wasted_kwh", ascending=False).head(1)
if not top.empty and top.iloc[0]["on_empty_h"] > 0:
    r = top.iloc[0]
    notes.append(f"{r['appliance']} was on for {fmt_h(r['on_empty_h'])} in an empty room "
                 f"({r['wasted_kwh']:.2f} kWh, {currency} {r['wasted_kwh'] * tariff:.2f}).")
for _, r in cam[cam["seen_on_empty_h"] > 0].iterrows():
    notes.append(f"The camera saw {r['item']} on for {fmt_h(r['seen_on_empty_h'])} with nobody in view.")
if s["ac_on_vacant_h"] > 0:
    notes.append(f"The AC ran for {fmt_h(s['ac_on_vacant_h'])} while the room was empty.")
hourly = an.hourly_profile(df)
if not hourly.empty and hourly["occupied_pct"].max() > 0:
    h = hourly.loc[hourly["occupied_pct"].idxmax()]
    notes.append(f"The room is busiest around {int(h['hour']):02d}:00 ({h['occupied_pct']:.0f} % of that hour occupied).")
if not notes:
    notes.append("No energy waste found in this period.")
section("Findings")
st.markdown('<div class="card">' + "".join(f'<div class="kv"><span class="k">{esc(n)}</span></div>' for n in notes)
            + "</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------- timelines
span_h = (df["time"].max() - df["time"].min()).total_seconds() / 3600
rule = "1min" if span_h > 2 else "10s"
ts = (df.set_index("time")[["people", "temp_c", "humidity"]].apply(pd.to_numeric, errors="coerce")
      .resample(rule).mean().dropna(subset=["people", "temp_c"], how="all").reset_index())
# a new line segment after every gap (app not running), so charts never draw across missing time
ts["seg"] = (ts["time"].diff().dt.total_seconds().fillna(0) > (180 if rule == "1min" else 60)).cumsum()
ac_on = df[df["ac_monitor"] == "ON"].copy()
segments = pd.DataFrame(columns=["start", "end"])
if not ac_on.empty:
    gap = ac_on["time"].diff().dt.total_seconds().fillna(0) > 60
    ac_on["seg"] = gap.cumsum()
    segments = ac_on.groupby("seg")["time"].agg(start="min", end="max").reset_index(drop=True)

section("Occupancy")
occ_chart = (alt.Chart(ts).mark_area(color="#12B76A", opacity=0.18, line={"color": "#12B76A", "strokeWidth": 1.5},
                                     interpolate="step-after")
             .encode(x=alt.X("time:T", title=None), y=alt.Y("people:Q", title="People", stack=None), detail="seg:N",
                     tooltip=[alt.Tooltip("time:T", format="%d %b %H:%M"), alt.Tooltip("people:Q", format=".1f")])
             .properties(height=200))
st.altair_chart(styled(occ_chart), use_container_width=True)

section("Temperature and AC")
temp_line = (alt.Chart(ts.dropna(subset=["temp_c"])).mark_line(color="#1D4ED8", strokeWidth=2)
             .encode(x=alt.X("time:T", title=None), y=alt.Y("temp_c:Q", title="°C", scale=alt.Scale(zero=False)),
                     detail="seg:N",
                     tooltip=[alt.Tooltip("time:T", format="%d %b %H:%M"), alt.Tooltip("temp_c:Q", format=".1f", title="°C")]))
layers = [temp_line]
if not segments.empty:
    layers.insert(0, alt.Chart(segments).mark_rect(color="#2E90FA", opacity=0.12).encode(x="start:T", x2="end:T"))
st.altair_chart(styled(alt.layer(*layers).properties(height=240)), use_container_width=True)
st.caption("Shaded: AC detected on from the temperature trend.")

# ---------------------------------------------------------------- usage + profile
c1, c2 = st.columns(2)
with c1:
    section("Appliance on-time")
    if not usage.empty:
        u = usage.melt(id_vars="appliance", value_vars=["on_h", "on_empty_h"], var_name="kind_", value_name="hours")
        u["kind_"] = u["kind_"].map({"on_h": "On", "on_empty_h": "On while empty"})
        bar = (alt.Chart(u).mark_bar(cornerRadiusEnd=3, height=14)
               .encode(y=alt.Y("appliance:N", title=None), x=alt.X("hours:Q", title="Hours"),
                       color=alt.Color("kind_:N", title=None,
                                       scale=alt.Scale(domain=["On", "On while empty"], range=["#84ADFF", "#F04438"])),
                       yOffset="kind_:N", tooltip=["appliance", "kind_", alt.Tooltip("hours:Q", format=".2f")])
               .properties(height=60 * max(len(usage), 1)))
        st.altair_chart(styled(bar), use_container_width=True)
with c2:
    section("Occupancy by hour of day")
    if not hourly.empty:
        hb = (alt.Chart(hourly).mark_bar(color="#12B76A", cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
              .encode(x=alt.X("hour:O", title="Hour"), y=alt.Y("occupied_pct:Q", title="% occupied", scale=alt.Scale(domain=[0, 100])),
                      tooltip=[alt.Tooltip("hour:O"), alt.Tooltip("occupied_pct:Q", format=".0f"), alt.Tooltip("avg_temp:Q", format=".1f")])
              .properties(height=220))
        st.altair_chart(styled(hb), use_container_width=True)

# ---------------------------------------------------------------- tables
section("Energy by appliance")
tbl = usage.assign(cost=usage["kwh"] * tariff, wasted_cost=usage["wasted_kwh"] * tariff)
st.dataframe(tbl[["appliance", "watts", "on_h", "on_empty_h", "kwh", "wasted_kwh", "cost", "wasted_cost"]],
             hide_index=True, use_container_width=True, column_config={
                 "appliance": "Appliance", "watts": st.column_config.NumberColumn("Rated W", format="%d"),
                 "on_h": st.column_config.NumberColumn("On (h)", format="%.2f"),
                 "on_empty_h": st.column_config.NumberColumn("On, empty (h)", format="%.2f"),
                 "kwh": st.column_config.NumberColumn("Energy (kWh)", format="%.3f"),
                 "wasted_kwh": st.column_config.NumberColumn("Wasted (kWh)", format="%.3f"),
                 "cost": st.column_config.NumberColumn(f"Cost ({currency})", format="%.2f"),
                 "wasted_cost": st.column_config.NumberColumn(f"Wasted ({currency})", format="%.2f")})
st.caption("Energy is estimated from relay on-time and the rated power in config.yaml (relays → watts).")

c3, c4 = st.columns(2)
with c3:
    section("AC sessions")
    sess = an.ac_sessions(events)
    if sess.empty:
        st.caption("No AC on/off detected in this period.")
    else:
        st.dataframe(sess.rename(columns={"on": "Switched on", "off": "Switched off", "minutes": "Minutes"}),
                     hide_index=True, use_container_width=True)
    section("Seen by the camera")
    if cam.empty:
        st.caption("No fans or lights recorded by the camera.")
    else:
        st.dataframe(cam.rename(columns={"item": "Item", "seen_on_h": "On (h)", "seen_on_empty_h": "On, nobody in view (h)"}),
                     hide_index=True, use_container_width=True)
with c4:
    section("Events and alerts")
    counts = an.alert_counts(events)
    st.dataframe(counts.rename(columns={"type": "Type", "level": "Level", "count": "Count"}),
                 hide_index=True, use_container_width=True, height=320)
