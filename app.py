"""Glassmorphism Streamlit dashboard (browser webcam via WebRTC, so it can be deployed). Run:  streamlit run app.py

Three fluid columns (no sidebar): live stream | telemetry KPIs | timeline analytics.
Streamlit only *renders*; capture, mesh and inference live in pipeline.py threads.
"""
import logging
import os
import time

import numpy as np
import plotly.graph_objects as go
import av
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

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


def rtc_config():
    """Google STUN by default; add a TURN relay via env/secrets (TURN_URL as a comma-separated list, TURN_USER, TURN_PASS) when the host blocks
    UDP (Hugging Face Spaces, Render)."""
    servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
    try:
        get = lambda k: os.environ.get(k) or st.secrets.get(k, "")
        if get("TURN_URL"):
            servers.append({"urls": [u.strip() for u in get("TURN_URL").split(",")], "username": get("TURN_USER"), "credential": get("TURN_PASS")})
    except Exception:  # no secrets file when running locally
        pass
    return {"iceServers": servers}


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


def kpi_html(s, state):
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
<div class="perf">in {s["cap_fps"]:.0f} fps · mesh {s["perc_fps"]:.0f} fps · infer {s["inf_ms"]:.1f} ms<br>
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


# ---- layout (main container, no sidebar) ----
st.markdown("### 🚘 Real-Time Driver Drowsiness & Distraction Monitor")
if "pipe" not in st.session_state or st.session_state.pipe.dead:
    st.session_state.pipe = Pipeline()
pipe = st.session_state.pipe


def on_frame(frame):
    # runs in the WebRTC thread: hand the frame to the pipeline, return it mirrored + overlaid
    return av.VideoFrame.from_ndarray(pipe.push(frame.to_ndarray(format="bgr24")), format="bgr24")


c1, c2, c3 = st.columns([5, 3, 4])
c1.markdown("##### Visual stream")
with c1:
    ctx = webrtc_streamer(
        key="monitor", mode=WebRtcMode.SENDRECV, rtc_configuration=rtc_config(),
        video_frame_callback=on_frame, async_processing=True,
        media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480},
                                            "frameRate": {"ideal": 30}}, "audio": False})
    st.caption("Press START and allow camera access. Video is processed on the server and not stored.")
c2.markdown("##### Telemetry")
kpis = c2.empty()
c3.markdown("##### Historical timeline")
chart = c3.empty()

last_state, n, t_log = None, 0, time.time()
while ctx.state.playing:
    s = pipe.snapshot()
    state = int(np.argmax(s["probs"]))
    if state != last_state:
        _, color, anim = STATES[state]
        theme_slot.markdown(f"<style>:root{{--accent:{color};--anim:{anim}}}</style>", unsafe_allow_html=True)
        last_state = state
    kpis.markdown(kpi_html(s, state), unsafe_allow_html=True)
    if n % 5 == 0:  # ~1 Hz: Plotly re-serialisation is the expensive part
        chart.plotly_chart(timeline(s), width="stretch", config={"displayModeBar": False})
    n += 1
    if time.time() - t_log > 5:
        perf.info("in=%.1f mesh=%.1f infer=%.1fms", s["cap_fps"], s["perc_fps"], s["inf_ms"])
        t_log = time.time()
    time.sleep(0.2)  # 5 Hz telemetry; the video plays client-side and never waits on this loop

pipe.stop()  # camera stopped: free the threads (a fresh Pipeline is created on the next run)
theme_slot.markdown("<style>:root{--accent:#475569;--anim:none}</style>", unsafe_allow_html=True)
