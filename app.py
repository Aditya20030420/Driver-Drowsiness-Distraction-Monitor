"""Glassmorphism Streamlit dashboard. Run:  streamlit run app.py

Three fluid columns (no sidebar): live stream | telemetry KPIs | timeline analytics.
Streamlit only *renders*; capture, mesh and inference live in pipeline.py threads.
"""
import logging
import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from pipeline import EAR_CLOSED, MAR_YAWN, Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
perf = logging.getLogger("perf")

st.set_page_config("Driver Monitor", layout="wide", initial_sidebar_state="collapsed")

# state -> (label, accent colour, animation)
STATES = {
    0: ("ALERT", "#00e5ff", "glow"),
    1: ("CRITICAL DROWSINESS", "#ff1744", "flash"),
    2: ("DISTRACTED", "#ff9100", "pulse"),
}

BASE_CSS = """
<style>
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="stHeader"] {display:none}
.stApp {background: radial-gradient(circle at 20% 0%, #1b2433 0%, #0b0f17 60%); color:#fff}
.block-container {padding-top:1.2rem; max-width:1800px}
h1,h2,h3,p,label,span {color:#fff}
/* each of the 3 columns is a glass card; the neon border is a pseudo-element animated via opacity only
   (compositor-friendly: no repaint of the live video underneath) */
[data-testid="stColumn"] {
  position: relative; background: rgba(30,41,59,.55); border-radius: 18px; padding: 14px 16px;}
[data-testid="stColumn"]::after {
  content:""; position:absolute; inset:0; border-radius:18px; pointer-events:none;
  border: 1px solid var(--accent); box-shadow: 0 0 22px var(--accent), inset 0 0 12px var(--accent);
  animation: var(--anim) 1.2s ease-in-out infinite; will-change: opacity;}
.kpi {background: rgba(15,23,42,.55); backdrop-filter: blur(12px); border:1px solid rgba(255,255,255,.12);
  border-radius:14px; padding:12px 16px; margin-bottom:12px}
.kpi .l {font-size:.72rem; letter-spacing:.12em; text-transform:uppercase; opacity:.65}
.kpi .v {font-size:2.1rem; font-weight:700; line-height:1.2}
.kpi svg {width:100%; height:34px; display:block}
.kpi.state .v {color: var(--accent); font-size:1.4rem}
.perf {font: .72rem/1.5 monospace; opacity:.6}
@keyframes glow  {0%,100% {opacity:1} 50% {opacity:.65}}
@keyframes pulse {0%,100% {opacity:1} 50% {opacity:.3}}
@keyframes flash {0%,100% {opacity:1} 50% {opacity:.15}}
</style>
"""
st.markdown(BASE_CSS, unsafe_allow_html=True)
theme_slot = st.empty()  # re-rendered only when the safety state changes


@st.cache_resource
def _registry():
    return {}


def get_pipeline(idx):
    """One live pipeline at a time; survives Streamlit reruns."""
    reg = _registry()
    for k in [k for k in reg if k != idx or reg[k].dead]:
        reg.pop(k).stop()
    return reg.setdefault(idx, Pipeline(idx))


def spark(vals, lo, hi, color, thresh=None):
    """Inline SVG sparkline."""
    if len(vals) < 2:
        return ""
    v = np.clip((np.asarray(vals) - lo) / (hi - lo), 0, 1)
    pts = " ".join(f"{i / (len(v) - 1) * 100:.1f},{30 - y * 28:.1f}" for i, y in enumerate(v))
    t = ""
    if thresh is not None:
        y = 30 - np.clip((thresh - lo) / (hi - lo), 0, 1) * 28
        t = f'<line x1="0" x2="100" y1="{y:.1f}" y2="{y:.1f}" stroke="#fff" stroke-opacity=".25" stroke-dasharray="2"/>'
    return (f'<svg viewBox="0 0 100 32" preserveAspectRatio="none">{t}'
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.5" vector-effect="non-scaling-stroke"/></svg>')


def kpi_html(s, state, ui_fps):
    pd, ear, mar = s["probs"][1], s["metrics"][0], s["metrics"][1]
    hist = s["history"][-90:]
    name, color, _ = STATES[state]
    var30 = float(np.var([h[1] for h in hist[-30:]])) if hist else 0.0
    return f"""
<div class="kpi state"><div class="l">Driver state</div><div class="v">{name}</div></div>
<div class="kpi"><div class="l">Drowsiness probability index</div><div class="v">{pd * 100:4.1f}%</div>
  {spark([h[1] for h in hist], 0, 1, color)}</div>
<div class="kpi"><div class="l">Eye aspect ratio (EAR)</div><div class="v">{ear:.3f}</div>
  {spark([h[3] for h in hist], 0.0, 0.45, "#00e5ff", EAR_CLOSED)}</div>
<div class="kpi"><div class="l">Mouth aspect ratio (MAR)</div><div class="v">{mar:.3f}</div>
  {spark([h[4] for h in hist], 0.0, 1.0, "#b388ff", MAR_YAWN)}</div>
<div class="kpi"><div class="l">Distraction prob. · 30-frame variance (drowsy)</div>
  <div class="v">{s["probs"][2] * 100:4.1f}% <span style="font-size:.9rem;opacity:.6">σ² {var30:.4f}</span></div></div>
<div class="perf">cam {s["cap_fps"]:.0f} fps · mesh {s["perc_fps"]:.0f} fps · infer {s["inf_ms"]:.1f} ms · ui {ui_fps:.0f} fps<br>
classifier: {s["label"]}</div>"""


def timeline(s):
    h = s["history"][-300:]
    if not h:
        return go.Figure()
    t = [x[0] - h[-1][0] for x in h]
    fig = go.Figure()
    fig.add_scatter(x=t, y=[x[1] for x in h], name="Drowsy", line=dict(color="#ff1744", width=2), fill="tozeroy",
                    fillcolor="rgba(255,23,68,.12)")
    fig.add_scatter(x=t, y=[x[2] for x in h], name="Distracted", line=dict(color="#ff9100", width=2))
    fig.add_hline(y=0.5, line_dash="dot", line_color="rgba(255,255,255,.3)")
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=420, margin=dict(l=8, r=8, t=8, b=8), legend=dict(orientation="h", y=1.08),
                      xaxis=dict(title="seconds ago", gridcolor="rgba(255,255,255,.08)"),
                      yaxis=dict(range=[0, 1], gridcolor="rgba(255,255,255,.08)"), uirevision="keep")
    return fig


# ---- control strip (main container, not sidebar) ----
top = st.columns([6, 1, 1])
top[0].markdown("### 🚘 Real-Time Driver Drowsiness & Distraction Monitor")
cam = int(top[1].number_input("Camera", 0, 9, 0))
run = top[2].toggle("Live", True)

c1, c2, c3 = st.columns([5, 3, 4])
c1.markdown("##### Visual stream")
video = c1.empty()
c2.markdown("##### Telemetry")
kpis = c2.empty()
c3.markdown("##### Historical timeline")
chart = c3.empty()

if not run:
    for p in list(_registry().values()):
        p.stop()
    _registry().clear()
    theme_slot.markdown("<style>:root{--accent:#475569;--anim:none}</style>", unsafe_allow_html=True)
    st.stop()

pipe = get_pipeline(cam)
last_seq, last_state, n, ui_rate, t_log, t_send = -1, None, 0, 0.0, time.time(), None
while True:
    t0 = time.perf_counter()
    if pipe.error:
        st.error(pipe.error)
        st.stop()
    if pipe.seq == last_seq:  # cheap check: avoid copying history while nothing is new
        time.sleep(0.003)
        continue
    s = pipe.snapshot()
    if s["frame"] is not None:
        last_seq = s["seq"]
        state = int(np.argmax(s["probs"]))
        if state != last_state:
            _, color, anim = STATES[state]
            theme_slot.markdown(f"<style>:root{{--accent:{color};--anim:{anim}}}</style>", unsafe_allow_html=True)
            last_state = state
        video.image(s["frame"], width="stretch")  # pre-encoded JPEG bytes
        if n % 6 == 0:  # ~5 Hz DOM updates
            kpis.markdown(kpi_html(s, state, ui_rate), unsafe_allow_html=True)
        if n % 60 == 0:  # ~0.5 Hz: Plotly re-serialisation is the expensive part
            chart.plotly_chart(timeline(s), width="stretch", config={"displayModeBar": False})
        n += 1
        now = time.perf_counter()
        if t_send:
            ui_rate = 0.9 * ui_rate + 0.1 / max(now - t_send, 1e-3)
        t_send = now
    if time.time() - t_log > 5:
        perf.info("cam=%.1f mesh=%.1f infer=%.1fms ui=%.1f", s["cap_fps"], s["perc_fps"], s["inf_ms"], ui_rate)
        t_log = time.time()
    dt = time.perf_counter() - t0

