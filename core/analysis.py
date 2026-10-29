"""Analysis of the stored history: occupancy, energy use and waste, temperature and AC behaviour.

Works on the status snapshots (one row every few seconds) and the event log written by `Storage`.
Each snapshot stands for the time until the next one (capped, so gaps while the app was stopped
do not count as hours of use).
"""
from __future__ import annotations

import re

import pandas as pd

VACANT_STATES = {"VACANT", "AC_OBSERVATION"}
DEFAULT_WATTS = {"light": 40, "fan": 75, "ac": 1500}     # used when config.yaml has no "watts" for a relay

STATUS_COLUMNS = ["time", "people", "state", "temp_c", "sensor_ok", "simulated", "appliances", "ac_inference",
                  "humidity", "visual", "ac_monitor", "r1", "r2", "r3", "r4"]
REGIONS = ["r1", "r2", "r3", "r4"]


def status_frame(rows: list[tuple], max_gap_s: float = 30.0) -> pd.DataFrame:
    """Rows from Storage.query_status_full (newest first) -> time-ordered frame with a duration per row.

    Snapshots are saved every few seconds; a longer gap means the app was not running, so a row never
    stands for more than `max_gap_s`.
    """
    df = pd.DataFrame(rows, columns=STATUS_COLUMNS)
    if df.empty:
        df["dur_s"] = []
        return df
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    df["dur_s"] = df["time"].shift(-1).sub(df["time"]).dt.total_seconds().clip(upper=max_gap_s).fillna(0)
    df["vacant"] = df["state"].isin(VACANT_STATES)
    df["occupied"] = df["people"].fillna(0) > 0
    return df


def parse_states(text: str | None) -> dict[str, str]:
    """'Light 1=ON, Fan=OFF' -> {'Light 1': 'ON', 'Fan': 'OFF'}"""
    out = {}
    if not isinstance(text, str):          # NULL in the database (older rows) arrives as NaN
        return out
    for part in text.split(","):
        m = re.match(r"\s*(.+?)\s*=\s*(ON|OFF|UNKNOWN)\s*$", part)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def summary(df: pd.DataFrame) -> dict:
    """Headline numbers for the selected period."""
    if df.empty:
        return {"monitored_h": 0.0}
    total = df["dur_s"].sum()
    temps = df.loc[df["sensor_ok"] == 1, "temp_c"].dropna()
    hum = df["humidity"].dropna() if "humidity" in df else pd.Series(dtype=float)
    ac_on = df["ac_monitor"] == "ON"
    return {
        "monitored_h": total / 3600,
        "occupied_pct": 100 * df.loc[df["occupied"], "dur_s"].sum() / total if total else 0.0,
        "vacant_pct": 100 * df.loc[df["vacant"], "dur_s"].sum() / total if total else 0.0,
        "peak_people": int(df["people"].fillna(0).max()),
        "temp_min": float(temps.min()) if len(temps) else None,
        "temp_max": float(temps.max()) if len(temps) else None,
        "temp_avg": float(temps.mean()) if len(temps) else None,
        "humidity_avg": float(hum.mean()) if len(hum) else None,
        "region_avg": {r: float(df[r].dropna().mean()) for r in REGIONS if r in df and df[r].notna().any()},
        "ac_on_h": df.loc[ac_on, "dur_s"].sum() / 3600,
        "ac_on_vacant_h": df.loc[ac_on & df["vacant"], "dur_s"].sum() / 3600,
    }


def appliance_usage(df: pd.DataFrame, relay_cfg: dict) -> pd.DataFrame:
    """Per relay: hours ON, hours ON while the room was empty, and estimated energy (kWh)."""
    rows = []
    if df.empty:
        return pd.DataFrame(columns=["appliance", "kind", "watts", "on_h", "on_empty_h", "kwh", "wasted_kwh"])
    states = df["appliances"].map(parse_states)
    for ch, rc in relay_cfg.items():
        name, kind = rc["name"], rc.get("kind", "")
        watts = float(rc.get("watts", DEFAULT_WATTS.get(kind, 0)))
        on = states.map(lambda d: d.get(name) == "ON")
        on_s = df.loc[on, "dur_s"].sum()
        waste_s = df.loc[on & df["vacant"], "dur_s"].sum()
        rows.append({"appliance": name, "kind": kind, "watts": watts, "on_h": on_s / 3600,
                     "on_empty_h": waste_s / 3600, "kwh": watts * on_s / 3.6e6, "wasted_kwh": watts * waste_s / 3.6e6})
    return pd.DataFrame(rows)


def camera_waste(df: pd.DataFrame) -> pd.DataFrame:
    """Fans / lights the CAMERA saw ON while nobody was in view (works without relays)."""
    if df.empty or "visual" not in df:
        return pd.DataFrame(columns=["item", "seen_on_h", "seen_on_empty_h"])
    states = df["visual"].map(parse_states)
    empty = ~df["occupied"]
    names = sorted({k for d in states for k in d})
    rows = []
    for n in names:
        on = states.map(lambda d: d.get(n) == "ON")
        rows.append({"item": n, "seen_on_h": df.loc[on, "dur_s"].sum() / 3600,
                     "seen_on_empty_h": df.loc[on & empty, "dur_s"].sum() / 3600})
    return pd.DataFrame(rows)


def hourly_profile(df: pd.DataFrame) -> pd.DataFrame:
    """By hour of day: share of time occupied, average people and average temperature."""
    if df.empty:
        return pd.DataFrame(columns=["hour", "occupied_pct", "avg_people", "avg_temp"])
    d = df.assign(hour=df["time"].dt.hour)
    g = d.groupby("hour")
    out = pd.DataFrame({
        "occupied_pct": g.apply(lambda x: 100 * x.loc[x["occupied"], "dur_s"].sum() / max(x["dur_s"].sum(), 1e-9),
                                include_groups=False),
        "avg_people": g["people"].mean(),
        "avg_temp": d[d["sensor_ok"] == 1].groupby("hour")["temp_c"].mean(),
    }).reset_index()
    return out


def ac_sessions(events: list[tuple]) -> pd.DataFrame:
    """AC ON periods from the event log ('AC detected ON' ... 'AC detected OFF')."""
    rows, start = [], None
    for ts, kind, _level, msg in sorted(events, key=lambda r: r[0]):
        if kind != "AC":
            continue
        if msg.startswith("AC detected ON"):
            start = start or ts
        elif msg.startswith("AC detected OFF") and start:
            rows.append({"on": start, "off": ts})
            start = None
    if start:
        rows.append({"on": start, "off": None})
    df = pd.DataFrame(rows, columns=["on", "off"])
    if not df.empty:
        on, off = pd.to_datetime(df["on"]), pd.to_datetime(df["off"])
        df["minutes"] = ((off - on).dt.total_seconds() / 60).round(1)
    return df


def alert_counts(events: list[tuple]) -> pd.DataFrame:
    """How often each alert / event type happened."""
    df = pd.DataFrame(events, columns=["time", "type", "level", "message"])
    if df.empty:
        return pd.DataFrame(columns=["type", "level", "count"])
    return (df.groupby(["type", "level"]).size().reset_index(name="count")
              .sort_values("count", ascending=False).reset_index(drop=True))
