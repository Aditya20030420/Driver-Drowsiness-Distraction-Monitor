"""Threaded capture -> MediaPipe ROI extraction -> inference, decoupled from the UI.

    Grabber thread    : camera.read() as fast as the sensor delivers; keeps only the NEWEST frame (stale frames dropped).
    Perception thread : face mesh, EAR/MAR/yaw, 64x64 patch mosaic, overlay drawing; feeds the 30-frame deques.
    Inference thread  : ~10 Hz GRU/heuristic classification of the rolling window (never blocks the video path).
    UI (Streamlit)    : only calls Pipeline.snapshot().

Model input: C=1, 64x64 per frame. Three ROIs share that one canvas as a mosaic:
    [ left eye 32x32 | right eye 32x32 ]
    [        mouth 64x32               ]
"""
import collections
import logging
import os
import threading
import time

import cv2
import mediapipe as mp
import numpy as np

log = logging.getLogger("pipeline")
WEIGHTS = os.path.join(os.path.dirname(__file__), "driver_monitor.pt")
SEQ_LEN = 30

# MediaPipe Face Mesh landmark indices
L_EYE = [362, 385, 387, 263, 373, 380]
R_EYE = [33, 160, 158, 133, 153, 144]
MOUTH_V = [(13, 14), (81, 178), (311, 402)]
MOUTH_H = (61, 291)
MOUTH_ALL = [61, 291, 13, 14, 81, 178, 311, 402, 0, 17]
NOSE, CHEEK_L, CHEEK_R = 1, 234, 454
EAR_CLOSED, MAR_YAWN, YAW_OFF = 0.21, 0.60, 0.18  # calibration knobs: tune per camera/driver

_CONTOURS = np.array(sorted(mp.solutions.face_mesh.FACEMESH_CONTOURS), dtype=np.int32)


def _d(a, b):
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(p, idx):
    a = p[idx]
    return (_d(a[1], a[5]) + _d(a[2], a[4])) / (2 * _d(a[0], a[3]) + 1e-6)


def mouth_aspect_ratio(p):
    return sum(_d(p[u], p[l]) for u, l in MOUTH_V) / (3 * _d(p[MOUTH_H[0]], p[MOUTH_H[1]]) + 1e-6)


def crop_patch(gray, pts, w_out, h_out, pad=0.4):
    """Crop a landmark-bounded ROI at a fixed aspect, resize (INTER_AREA) and min-max normalise -> uint8."""
    (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
    w, h = (x1 - x0) * (1 + pad), (y1 - y0) * (1 + pad)
    ar = w_out / h_out
    w, h = (h * ar, h) if w / h < ar else (w, w / ar)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    H, W = gray.shape
    bx0, by0 = max(int(cx - w / 2), 0), max(int(cy - h / 2), 0)
    bx1, by1 = min(int(cx + w / 2), W), min(int(cy + h / 2), H)
    roi = gray[by0:by1, bx0:bx1]
    if roi.size == 0:
        return np.zeros((h_out, w_out), np.uint8), (bx0, by0, bx1, by1)
    roi = cv2.resize(roi, (w_out, h_out), interpolation=cv2.INTER_AREA)
    return cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX), (bx0, by0, bx1, by1)


class Rate:
    """EMA frames-per-second meter."""
    def __init__(self):
        self.t, self.fps = None, 0.0

    def tick(self):
        now = time.perf_counter()
        if self.t:
            self.fps = 0.9 * self.fps + 0.1 / max(now - self.t, 1e-6)
        self.t = now


class Grabber(threading.Thread):
    def __init__(self, idx):
        super().__init__(daemon=True)
        self.idx, self.stop_evt, self.cond = idx, threading.Event(), threading.Condition()
        self.frame, self.seq, self.error, self.rate = None, 0, None, Rate()

    def run(self):
        backends = (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY) if os.name == "nt" else (cv2.CAP_ANY,)
        for be in backends:  # DirectShow is fastest but flaky; fall back to Media Foundation
            cap = cv2.VideoCapture(self.idx, be)
            if cap.isOpened():
                break
        for k, v in ((cv2.CAP_PROP_FRAME_WIDTH, 640), (cv2.CAP_PROP_FRAME_HEIGHT, 480),
                     (cv2.CAP_PROP_FPS, 30), (cv2.CAP_PROP_BUFFERSIZE, 1)):
            cap.set(k, v)
        if not cap.isOpened():
            self.error = f"Cannot open camera {self.idx}"
            self.stop_evt.set()
            with self.cond:
                self.cond.notify_all()
            return
        while not self.stop_evt.is_set():
            ok, f = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.cond:
                self.frame, self.seq = f, self.seq + 1
                self.cond.notify_all()
            self.rate.tick()
        cap.release()

    def wait_new(self, last, timeout=0.5):
        with self.cond:
            self.cond.wait_for(lambda: self.seq != last or self.stop_evt.is_set(), timeout)
            return self.frame, self.seq


def heuristic_probs(feats):
    """Fallback when no trained weights exist: PERCLOS / yawn / gaze-off fractions over the 30-frame window.
    feats: list of (ear, mar, yaw, face)."""
    f = np.array(feats, dtype=np.float32)
    face = f[:, 3] > 0
    closed = np.mean((f[:, 0] < EAR_CLOSED) & face)
    yawn = np.mean((f[:, 1] > MAR_YAWN) & face)
    off = np.mean((np.abs(f[:, 2]) > YAW_OFF) | ~face)
    drowsy, dist = min(1.0, 2 * closed + 2 * yawn), min(1.0, 2.5 * off)
    p = np.array([max(0.0, 1 - drowsy - dist), drowsy, dist]) + 1e-3
    return p / p.sum()


def load_classifier():
    """Returns (callable(patches, feats) -> probs[3], label). Uses driver_monitor.pt if present."""
    if not os.path.exists(WEIGHTS):
        return (lambda patches, feats: heuristic_probs(feats)), "heuristic (no driver_monitor.pt)"
    import torch
    from model import DriverMonitorNet
    torch.set_num_threads(2)
    net = DriverMonitorNet().eval()
    net.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))

    @torch.inference_mode()
    def run(patches, feats):
        x = torch.from_numpy(np.stack(patches)).float().div_(255).view(1, len(patches), 1, 64, 64)
        return net(x)[0].numpy()
    return run, "DriverMonitorNet (CNN+GRU)"


class Pipeline:
    def __init__(self, cam_idx=0):
        self.cam_idx = cam_idx
        self.lock = threading.Lock()
        self.patches = collections.deque(maxlen=SEQ_LEN)   # uint8 64x64 mosaics (~120 KB total)
        self.feats = collections.deque(maxlen=SEQ_LEN)     # (ear, mar, yaw, face)
        self.history = collections.deque(maxlen=300)       # (t, p_drowsy, p_distracted, ear, mar)
        self.geo = None
        self.probs, self.frame, self.metrics = np.array([1.0, 0.0, 0.0]), None, (0.0, 0.0, 0.0, False)
        self.seq, self.inf_ms, self.perc_rate, self.label = 0, 0.0, Rate(), ""
        self.grab = Grabber(cam_idx)
        self.dead = False
        self._new = threading.Event()
        self._threads = [self.grab,
                         threading.Thread(target=self._perceive, daemon=True),
                         threading.Thread(target=self._render, daemon=True),
                         threading.Thread(target=self._infer, daemon=True)]
        for t in self._threads:
            t.start()

    @property
    def error(self):
        return self.grab.error

    def stop(self):
        self.dead = True
        self.grab.stop_evt.set()

    def _perceive(self):
        mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=False,
                                               min_detection_confidence=0.5, min_tracking_confidence=0.5)
        last = 0
        while not self.dead:
            frame, seq = self.grab.wait_new(last)
            if frame is None or seq == last:
                continue
            last = seq
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(cv2.flip(cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA), 1), cv2.COLOR_BGR2RGB)  # mirrored; landmarks are normalised so they scale back
            rgb.flags.writeable = False
            res = mesh.process(rgb)
            mosaic, geo = np.zeros((64, 64), np.uint8), None
            ear = mar = yaw = 0.0
            face = bool(res.multi_face_landmarks)
            if face:
                lm = res.multi_face_landmarks[0].landmark
                p = np.array([(l.x * w, l.y * h) for l in lm], np.float32)
                ear = (eye_aspect_ratio(p, L_EYE) + eye_aspect_ratio(p, R_EYE)) / 2
                mar = mouth_aspect_ratio(p)
                yaw = (p[NOSE, 0] - p[CHEEK_L, 0]) / (p[CHEEK_R, 0] - p[CHEEK_L, 0] + 1e-6) - 0.5
                if self.metrics[3]:  # EMA: steadier numbers, no single-frame flicker
                    ear, mar, yaw = (0.5 * n + 0.5 * o for n, o in zip((ear, mar, yaw), self.metrics[:3]))
                gray = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1)
                le, lb = crop_patch(gray, p[L_EYE], 32, 32)
                re, rb = crop_patch(gray, p[R_EYE], 32, 32)
                mo, mb = crop_patch(gray, p[MOUTH_ALL], 64, 32)
                mosaic = np.vstack([np.hstack([le, re]), mo])
                eye_c = (0, 0, 255) if ear < EAR_CLOSED else (0, 255, 120)
                mouth_c = (0, 140, 255) if mar > MAR_YAWN else (0, 255, 120)
                geo = (p.astype(np.int32)[_CONTOURS], ((lb, eye_c), (rb, eye_c), (mb, mouth_c)))
            with self.lock:
                self.patches.append(mosaic)
                self.feats.append((ear, mar, yaw, face))
                self.metrics, self.geo = (ear, mar, yaw, face), geo
                self.perc_rate.tick()
                self.history.append((time.time(), *self.probs[1:], ear, mar))
            self._new.set()
        mesh.close()

    def _render(self, fps=30):
        """Streams the newest camera frame at a fixed 30 fps cadence with the latest mesh result drawn on it, so the video stays
        smooth even when the mesh stage (heavy on a 2-core CPU) only manages ~20 Hz."""
        last, t_prev = 0, time.perf_counter()
        while not self.dead:
            frame, seq = self.grab.wait_new(last)
            if frame is None or seq == last:
                continue
            last = seq
            out = cv2.flip(frame, 1)  # mirror: natural for the driver
            geo = self.geo
            if geo:
                cv2.polylines(out, geo[0], False, (255, 220, 0), 1, cv2.LINE_AA)
                for b, c in geo[1]:
                    cv2.rectangle(out, b[:2], b[2:], c, 2)
            jpeg = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()  # GIL-free, off the UI thread
            with self.lock:
                self.frame, self.seq = jpeg, self.seq + 1
            t_prev += 1 / fps  # fixed cadence: bursty camera delivery would otherwise show up as stutter
            time.sleep(max(0.0, t_prev - time.perf_counter()))
            t_prev = max(t_prev, time.perf_counter() - 0.1)

    def _infer(self):
        classify, self.label = load_classifier()
        smooth, interval = None, 0.1
        while not self.dead:
            self._new.wait(0.5)
            self._new.clear()
            with self.lock:
                patches, feats = list(self.patches), list(self.feats)
            if not patches:
                continue
            while len(patches) < SEQ_LEN:  # warm-up: left-pad with the oldest frame
                patches.insert(0, patches[0])
                feats.insert(0, feats[0])
            t0 = time.perf_counter()
            p = np.asarray(classify(patches, feats), dtype=np.float32)
            smooth = p if smooth is None else 0.6 * smooth + 0.4 * p  # EMA against flicker
            dt = time.perf_counter() - t0
            with self.lock:
                self.probs, self.inf_ms = smooth, dt * 1000
            time.sleep(max(0.0, interval - dt))

    def snapshot(self):
        with self.lock:
            return dict(frame=self.frame, seq=self.seq, probs=self.probs.copy(), metrics=self.metrics,
                        history=list(self.history), inf_ms=self.inf_ms, label=self.label,
                        cap_fps=self.grab.rate.fps, perc_fps=self.perc_rate.fps)


if __name__ == "__main__":
    # self-check: geometry helpers + heuristic behave sensibly
    assert crop_patch(np.random.randint(0, 255, (480, 640), np.uint8), np.array([[300, 200], [340, 220]], np.float32), 32, 32)[0].shape == (32, 32)
    awake = [(0.3, 0.1, 0.0, True)] * SEQ_LEN
    asleep = [(0.1, 0.1, 0.0, True)] * SEQ_LEN
    away = [(0.3, 0.1, 0.3, True)] * SEQ_LEN
    assert heuristic_probs(awake).argmax() == 0 and heuristic_probs(asleep).argmax() == 1 and heuristic_probs(away).argmax() == 2
    print("ok")
