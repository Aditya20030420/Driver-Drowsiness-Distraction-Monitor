# Real-Time Edge Driver Drowsiness & Distraction Monitor

OpenCV + MediaPipe Face Mesh -> CNN (spatial) -> GRU (30-frame temporal) -> Alert / Drowsy / Distracted,
shown in a glassmorphism Streamlit dashboard.

- `model.py` – `DriverMonitorNet` (CNN + GRU + softmax head)
- `pipeline.py` – threaded capture, landmark ROI extraction, EAR/MAR, inference
- `app.py` – Streamlit dashboard

```
pip install streamlit plotly torch mediapipe opencv-python
streamlit run app.py
```

Without `driver_monitor.pt` the app uses a rule-based fallback (PERCLOS / yawn / head-turn); train `DriverMonitorNet`
and save its `state_dict` as `driver_monitor.pt` to switch to the neural model.
