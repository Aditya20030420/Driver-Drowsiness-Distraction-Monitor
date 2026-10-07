# Real-Time Edge Driver Drowsiness & Distraction Monitor

OpenCV + MediaPipe Face Mesh -> CNN (spatial) -> GRU (30-frame temporal) -> Alert / Drowsy / Distracted,
shown in a glassmorphism Streamlit dashboard.

- `model.py` – `DriverMonitorNet` (CNN + GRU + softmax head)
- `pipeline.py` – threaded capture, landmark ROI extraction, EAR/MAR, inference
- `app.py` – Streamlit dashboard

```
pip install -r requirements.txt
streamlit run app.py   # press START in the page; the browser webcam is streamed via WebRTC
```

Without `driver_monitor.pt` the app uses a rule-based fallback (PERCLOS / yawn / head-turn); train `DriverMonitorNet`
and save its `state_dict` as `driver_monitor.pt` to switch to the neural model.

## Deploy
- **Streamlit Community Cloud** (easiest; STUN is enough): point it at `app.py`; `packages.txt` installs the system libs.
- **Docker** (Hugging Face Spaces / Render): `Dockerfile` included. These hosts usually block UDP, so set `TURN_URL`, `TURN_USER`, `TURN_PASS` (e.g. a free metered.ca Open Relay account).
