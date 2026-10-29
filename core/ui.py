"""Shared UI pieces: the one Engine for all pages, the light theme CSS and small HTML helpers."""
from __future__ import annotations

import html

import streamlit as st

from core.config import load_config
from core.engine import Engine
from core.storage import Storage


@st.cache_resource
def get_engine() -> Engine:
    """One engine for the whole app (model loaded once, camera opened once), shared by every page."""
    return Engine(load_config())


@st.cache_resource
def get_storage() -> Storage:
    return Storage(load_config()["logging"]["db_path"])


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, [data-testid="stMarkdownContainer"], button, input, textarea, select {
  font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
}
.stApp { background: #F6F7F9; }
header[data-testid="stHeader"] { background: rgba(246,247,249,0.85); backdrop-filter: blur(6px); }
#MainMenu, footer, .stDeployButton, [data-testid="stAppDeployButton"] { display: none !important; }
.block-container { padding-top: 3.2rem; padding-bottom: 3rem; max-width: 1440px; }
section[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E4E7EC; }
section[data-testid="stSidebar"] h3 { font-size: .78rem; text-transform: uppercase; letter-spacing: .06em;
  color: #667085; font-weight: 600; margin: 1.1rem 0 .3rem 0; }
h1 { font-size: 1.55rem !important; font-weight: 650 !important; letter-spacing: -0.015em; color: #101828; }
h2, h3 { letter-spacing: -0.01em; color: #101828; }

.page-head { display: flex; justify-content: space-between; align-items: flex-end; gap: 16px;
  flex-wrap: wrap; margin: 0 0 18px 0; }
.page-head .title { font-size: 1.5rem; font-weight: 650; color: #101828; letter-spacing: -0.015em; }
.page-head .sub { color: #667085; font-size: .9rem; margin-top: 2px; }

.pill { display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 999px;
  font-size: .74rem; font-weight: 600; letter-spacing: .02em; border: 1px solid #E4E7EC; background: #F9FAFB;
  color: #344054; white-space: nowrap; margin-left: 6px; }
.pill .dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; display: inline-block; }
.pill-green  { background: #ECFDF3; color: #067647; border-color: #ABEFC6; }
.pill-amber  { background: #FFFAEB; color: #B54708; border-color: #FEDF89; }
.pill-red    { background: #FEF3F2; color: #B42318; border-color: #FECDCA; }
.pill-blue   { background: #EFF8FF; color: #175CD3; border-color: #B2DDFF; }
.pill-gray   { background: #F2F4F7; color: #475467; border-color: #E4E7EC; }

.kpi { background: #FFFFFF; border: 1px solid #E4E7EC; border-radius: 12px; padding: 14px 16px 13px 16px;
  box-shadow: 0 1px 2px rgba(16,24,40,.04); height: 100%; min-height: 104px; }
.kpi .lbl { font-size: .72rem; text-transform: uppercase; letter-spacing: .06em; color: #667085; font-weight: 600; }
.kpi .val { font-size: 1.45rem; font-weight: 650; color: #101828; margin-top: 6px; line-height: 1.2;
  letter-spacing: -0.01em; word-break: break-word; }
.kpi .sub { font-size: .8rem; color: #667085; margin-top: 4px; }
.kpi.tone-green { border-top: 3px solid #12B76A; }
.kpi.tone-amber { border-top: 3px solid #F79009; }
.kpi.tone-red   { border-top: 3px solid #F04438; }
.kpi.tone-blue  { border-top: 3px solid #2E90FA; }
.kpi.tone-gray  { border-top: 3px solid #D0D5DD; }

.card { background: #FFFFFF; border: 1px solid #E4E7EC; border-radius: 12px; padding: 16px 18px;
  box-shadow: 0 1px 2px rgba(16,24,40,.04); margin-bottom: 14px; }
.card h4 { margin: 0 0 10px 0; font-size: .95rem; font-weight: 600; color: #101828; }
.kv { display: flex; justify-content: space-between; align-items: center; padding: 7px 0;
  border-bottom: 1px solid #F2F4F7; font-size: .88rem; color: #344054; }
.kv:last-child { border-bottom: none; }
.kv .k { color: #667085; }
.kv .v { font-weight: 600; color: #101828; }
.muted { color: #667085; font-size: .83rem; }

.banner { border: 1px solid; border-left-width: 4px; border-radius: 10px; padding: 12px 16px; margin: 0 0 10px 0; }
.banner .t { font-weight: 650; color: #101828; font-size: .95rem; }
.banner .m { color: #475467; white-space: pre-line; font-size: .88rem; margin-top: 3px; }
.banner .time { color: #98A2B3; font-size: .76rem; margin-top: 4px; }
.banner-critical { background: #FEF3F2; border-color: #FECDCA; border-left-color: #D92D20; }
.banner-warning  { background: #FFFAEB; border-color: #FEDF89; border-left-color: #F79009; }
.banner-info     { background: #EFF8FF; border-color: #B2DDFF; border-left-color: #2E90FA; }

.rgrid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 4px; }
.rcell { border-radius: 10px; padding: 10px 12px; border: 1px solid rgba(16,24,40,.08); min-height: 74px; }
.rcell .rn { font-size: .7rem; text-transform: uppercase; letter-spacing: .06em; color: #475467; font-weight: 600; }
.rcell .rt { font-size: 1.35rem; font-weight: 650; color: #101828; margin-top: 2px; }
.rcell .rd { font-size: .76rem; color: #475467; margin-top: 1px; }
.rcell.none { background: #F9FAFB; }
.rcell.none .rt { color: #98A2B3; font-size: 1rem; }
.video { width: 100%; border-radius: 12px; border: 1px solid #E4E7EC; background: #0B0F14; display: block;
  min-height: 260px; object-fit: contain; }
.section { font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; color: #667085;
  font-weight: 600; margin: 22px 0 8px 0; }
[data-testid="stDataFrame"] { border: 1px solid #E4E7EC; border-radius: 10px; overflow: hidden; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid #E4E7EC; }
.stTabs [data-baseweb="tab"] { font-weight: 500; padding: 8px 14px; }
div[data-testid="stExpander"] details { background: #FFFFFF; border: 1px solid #E4E7EC; border-radius: 12px; }
.stButton > button { border-radius: 8px; font-weight: 500; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(text) -> str:
    return html.escape(str(text))


def pill(text: str, tone: str = "gray", dot: bool = True) -> str:
    return f'<span class="pill pill-{tone}">{"<span class=dot></span>" if dot else ""}{esc(text)}</span>'


def kpi(label: str, value, sub: str = "", tone: str = "gray") -> str:
    return (f'<div class="kpi tone-{tone}"><div class="lbl">{esc(label)}</div>'
            f'<div class="val">{esc(value)}</div><div class="sub">{esc(sub)}</div></div>')


def kpi_row(cards: list[str]) -> None:
    for col, card in zip(st.columns(len(cards)), cards):
        col.markdown(card, unsafe_allow_html=True)


def page_head(title: str, sub: str = "", right_html: str = "") -> None:
    st.markdown(f'<div class="page-head"><div><div class="title">{esc(title)}</div>'
                f'<div class="sub">{esc(sub)}</div></div><div>{right_html}</div></div>', unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(f'<div class="section">{esc(title)}</div>', unsafe_allow_html=True)


def banner(title: str, message: str, when: str = "", level: str = "warning") -> None:
    kind = {"CRITICAL": "critical", "WARNING": "warning"}.get(level, "info")
    st.markdown(f'<div class="banner banner-{kind}"><div class="t">{esc(title)}</div>'
                f'<div class="m">{esc(message)}</div><div class="time">{esc(when)}</div></div>', unsafe_allow_html=True)


def card(title: str, rows: list[tuple[str, str]], note: str = "") -> str:
    body = "".join(f'<div class="kv"><span class="k">{esc(k)}</span><span class="v">{v}</span></div>' for k, v in rows)
    return f'<div class="card"><h4>{esc(title)}</h4>{body}' + (f'<div class="muted">{esc(note)}</div>' if note else "") + "</div>"


STATE_TONE = {"OCCUPIED": "green", "VACANCY_PENDING": "amber", "VACANT": "gray", "AC_OBSERVATION": "blue",
              "MANUAL_OVERRIDE": "blue", "ERROR": "red", "INITIALIZING": "gray"}
STATE_TEXT = {"OCCUPIED": "Occupied", "VACANCY_PENDING": "Vacancy pending", "VACANT": "Vacant",
              "AC_OBSERVATION": "Vacant, checking AC", "MANUAL_OVERRIDE": "Manual override", "ERROR": "Camera error",
              "INITIALIZING": "Starting"}


def heat_color(t: float, lo: float = 22.0, hi: float = 34.0) -> str:
    """Light-theme heatmap colour: cool blue -> neutral -> warm red."""
    stops = [(0.0, (209, 233, 255)), (0.5, (254, 247, 214)), (1.0, (254, 214, 210))]
    x = min(1.0, max(0.0, (t - lo) / (hi - lo)))
    for (x0, c0), (x1, c1) in zip(stops, stops[1:]):
        if x <= x1:
            f = (x - x0) / (x1 - x0)
            return "rgb(%d,%d,%d)" % tuple(round(a + (b - a) * f) for a, b in zip(c0, c1))
    return "rgb(%d,%d,%d)" % stops[-1][1]


def region_grid(temps: list | None, deltas: dict | None = None) -> str:
    """2 x 2 grid, one cell per classroom region (report: 4-region heatmap)."""
    temps = (list(temps or []) + [None] * 4)[:4]
    cells = []
    for i, t in enumerate(temps, start=1):
        d = (deltas or {}).get(i)
        extra = f'<div class="rd">ΔT {d["delta"]:.1f} °C</div>' if d else '<div class="rd">&nbsp;</div>'
        if t is None:
            cells.append(f'<div class="rcell none"><div class="rn">Region {i}</div><div class="rt">No sensor</div>{extra}</div>')
        else:
            cells.append(f'<div class="rcell" style="background:{heat_color(t)}"><div class="rn">Region {i}</div>'
                         f'<div class="rt">{t:.1f} °C</div>{extra}</div>')
    return '<div class="rgrid">' + "".join(cells) + "</div>"
