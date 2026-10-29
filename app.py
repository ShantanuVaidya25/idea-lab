"""Smart Classroom dashboard.   Run:  streamlit run app.py"""
from __future__ import annotations

import logging

import streamlit as st

from core.ui import inject_css

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
st.set_page_config(page_title="Smart Classroom", page_icon=":material/bolt:", layout="wide")
inject_css()

nav = st.navigation([
    st.Page("views/dashboard.py", title="Dashboard", icon=":material/space_dashboard:", default=True),
    st.Page("views/analysis.py", title="Analysis", icon=":material/insights:"),
    st.Page("views/evaluation.py", title="Evaluation", icon=":material/fact_check:"),
    st.Page("views/logs.py", title="Logs", icon=":material/list_alt:"),
], position="top")
nav.run()
