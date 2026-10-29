"""Logs: every event and status snapshot stored in the database, with filters and CSV download."""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from core.analysis import STATUS_COLUMNS
from core.ui import get_storage, kpi, kpi_row, page_head

store = get_storage()
RANGES = {"Last 10 minutes": timedelta(minutes=10), "Last hour": timedelta(hours=1),
          "Last 24 hours": timedelta(hours=24), "Last 7 days": timedelta(days=7), "All time": None}

page_head("Logs", "Every event and status snapshot the system has recorded")

with st.sidebar:
    st.markdown("### Filters")
    rng = st.selectbox("Time range", list(RANGES), index=2)
    levels = st.multiselect("Level", ["INFO", "WARNING", "CRITICAL"], default=["INFO", "WARNING", "CRITICAL"])
    kinds = st.multiselect("Type", store.event_kinds(), placeholder="All types")
    text = st.text_input("Search", placeholder="fan, camera, relay ...")
    limit = st.select_slider("Maximum rows", [100, 500, 1000, 5000, 20000], value=1000)
    live = st.toggle("Live refresh", value=True)

since = None if RANGES[rng] is None else (datetime.now() - RANGES[rng]).strftime("%Y-%m-%d %H:%M:%S")


@st.fragment(run_every=3 if live else None)
def show() -> None:
    counts = store.event_counts(since)
    kpi_row([kpi("Events", sum(counts.values()), rng.lower(), "gray"),
             kpi("Critical", counts.get("CRITICAL", 0), "", "red" if counts.get("CRITICAL") else "gray"),
             kpi("Warning", counts.get("WARNING", 0), "", "amber" if counts.get("WARNING") else "gray"),
             kpi("Info", counts.get("INFO", 0), "", "blue")])
    st.write("")
    tab_ev, tab_status = st.tabs(["Events and alerts", "Status history"])

    with tab_ev:
        rows = store.query_events(levels or None, kinds or None, text.strip(), since, limit)
        if rows:
            df = pd.DataFrame(rows, columns=["Time", "Type", "Level", "Message"])
            st.caption(f"{len(df)} rows, newest first" + ("  (row limit reached)" if len(df) >= limit else ""))
            st.dataframe(df, hide_index=True, use_container_width=True, height=560,
                         column_config={"Message": st.column_config.TextColumn(width="large")})
            st.download_button("Download events (CSV)", df.to_csv(index=False).encode("utf-8"),
                               file_name=f"events_{datetime.now():%Y%m%d_%H%M%S}.csv", mime="text/csv")
        else:
            st.info("No events match these filters.")

    with tab_status:
        rows = store.query_status_full(since, 5000)
        if rows:
            df = pd.DataFrame(rows, columns=STATUS_COLUMNS).rename(columns={
                "time": "Time", "people": "People", "state": "Room", "temp_c": "Temp (°C)", "humidity": "Humidity (%)",
                "ac_monitor": "AC (trend)", "ac_inference": "Empty-room check", "appliances": "Relays",
                "visual": "Camera sees", "sensor_ok": "Sensor OK", "simulated": "Simulated",
                "r1": "Region 1 (°C)", "r2": "Region 2 (°C)", "r3": "Region 3 (°C)", "r4": "Region 4 (°C)"})
            order = ["Time", "People", "Room", "Temp (°C)", "Region 1 (°C)", "Region 2 (°C)", "Region 3 (°C)",
                     "Region 4 (°C)", "Humidity (%)", "AC (trend)", "Empty-room check", "Relays", "Camera sees",
                     "Sensor OK", "Simulated"]
            st.dataframe(df[order], hide_index=True, use_container_width=True, height=520)
            st.download_button("Download status history (CSV)", df[order].to_csv(index=False).encode("utf-8"),
                               file_name=f"status_{datetime.now():%Y%m%d_%H%M%S}.csv", mime="text/csv")
        else:
            st.info("No status snapshots yet. One is saved every few seconds while the dashboard runs.")


show()
