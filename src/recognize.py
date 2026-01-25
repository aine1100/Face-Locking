# src/recognize.py
"""
Multi-face recognition (CPU-friendly) using your now-stable pipeline:
Haar (multi-face) -> FaceMesh 5pt (per-face ROI) -> align_face_5pt (112x112)
-> ArcFace ONNX embedding -> cosine distance to DB -> label each face.
Run:
python -m src.recognize
Keys:
q : quit
r : reload DB from disk (data/db/face_db.npz)
+/- : adjust threshold (distance) live
d : toggle debug overlay
Notes:
- We run FaceMesh on EACH Haar face ROI (not the full frame).
- DB is expected from enroll: data/db/face_db.npz (name -> embedding vector)
- Distance definition: cosine_distance = 1 - cosine_similarity.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

try:
    import mediapipe as mp
except Exception as e:
    mp = None
    _MP_IMPORT_ERROR = e

from .haar_5pt import align_face_5pt

# -------------------------
# Data
# -------------------------

@dataclass
class FaceDet:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    kps: np.ndarray  # (5,2) float32 in FULL-frame coords

@dataclass
class MatchResult:
    name: Optional[str]
    distance: float
    similarity: float
    accepted: bool

# -------------------------
# Math helpers
# -------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    return float(np.dot(a, b))

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - cosine_similarity(a, b)

def _clip_xyxy(
    x1: float, y1: float, x2: float, y2: float, W: int, H: int
) -> Tuple[int, int, int, int]:
    x1 = int(max(0, min(W - 1, round(x1))))
    y1 = int(max(0, min(H - 1, round(y1))))
    x2 = int(max(0, min(W - 1, round(x2))))
    y2 = int(max(0, min(H - 1, round(y2))))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    return x1, y1, x2, y2

def _bbox_from_5pt(
    kps: np.ndarray,
    pad_x: float = 0.55,
    pad_y_top: float = 0.85,
    pad_y_bot: float = 1.15,
) -> np.ndarray:
    k = kps.astype(np.float32)
    x_min, x_max = float(np.min(k[:, 0])), float(np.max(k[:, 0]))
    y_min, y_max = float(np.min(k[:, 1])), float(np.max(k[:, 1]))

    w = max(1.0, x_max - x_min)
    h = max(1.0, y_max - y_min)

    return np.array(
        [
            x_min - pad_x * w,
            y_min - pad_y_top * h,
            x_max + pad_x * w,
            y_max + pad_y_bot * h,
        ],
        dtype=np.float32,
    )

def _kps_span_ok(kps: np.ndarray, min_eye_dist: float) -> bool:
    le, re, no, lm, rm = kps.astype(np.float32)
    if np.linalg.norm(re - le) < min_eye_dist:
        return False
    if not (lm[1] > no[1] and rm[1] > no[1]):
        return False
    return True

# -------------------------
# DB helpers
# -------------------------

def load_db_npz(db_path: Path) -> Dict[str, np.ndarray]:
    if not db_path.exists():
        return {}

    data = np.load(str(db_path), allow_pickle=True)
    return {k: np.asarray(data[k], dtype=np.float32).reshape(-1) for k in data.files}

# -------------------------
# Embedder
# -------------------------

class ArcFaceEmbedderONNX:
    def __init__(
        self,
        model_path: str = "models/arcface.onnx",
        input_size: Tuple[int, int] = (112, 112),
        debug: bool = False,
    ):
        self.in_w, self.in_h = input_size
        self.debug = debug
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        if img.shape[:2] != (self.in_h, self.in_w):
            img = cv2.resize(img, (self.in_w, self.in_h))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 128.0
        # return np.transpose(rgb, (2, 0, 1))[None, ...]
        return rgb[None, ...]

    def embed(self, img: np.ndarray) -> np.ndarray:
        x = self._preprocess(img)
        y = self.sess.run([self.out_name], {self.in_name: x})[0]
        v = y.reshape(-1).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-12)

# -------------------------
# Haar + FaceMesh detector
# -------------------------

class HaarFaceMesh5pt:
    def __init__(
        self,
        min_size: Tuple[int, int] = (70, 70),
        debug: bool = False,
    ):
        self.debug = debug
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load Haar cascade")

        if mp is None:
            raise RuntimeError(f"mediapipe import failed: {_MP_IMPORT_ERROR}")

        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.IDXS = [33, 263, 1, 61, 291]

    def detect(self, frame: np.ndarray, max_faces: int = 5) -> List[FaceDet]:
        H, W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) == 0:
            return []

        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[:max_faces]
        out: List[FaceDet] = []

        for (x, y, w, h) in faces:
            mx, my = 0.25 * w, 0.35 * h
            rx1, ry1, rx2, ry2 = _clip_xyxy(
                x - mx, y - my, x + w + mx, y + h + my, W, H
            )
            roi = frame[ry1:ry2, rx1:rx2]
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            res = self.mesh.process(rgb)

            if not res.multi_face_landmarks:
                continue

            lm = res.multi_face_landmarks[0].landmark
            kps = np.array([[lm[i].x * (rx2 - rx1), lm[i].y * (ry2 - ry1)] for i in self.IDXS])
            kps[:, 0] += rx1
            kps[:, 1] += ry1

            if not _kps_span_ok(kps, min_eye_dist=max(10, 0.18 * w)):
                continue

            bb = _bbox_from_5pt(kps)
            x1, y1, x2, y2 = _clip_xyxy(*bb, W, H)

            out.append(FaceDet(x1, y1, x2, y2, 1.0, kps))

        return out

# -------------------------
# Matcher
# -------------------------

class FaceDBMatcher:
    def __init__(self, db: Dict[str, np.ndarray], dist_thresh: float):
        self.db = db
        self.dist_thresh = dist_thresh
        self.names = sorted(db.keys())
        self.mat = np.stack([db[n] for n in self.names]) if self.names else None

    def reload_from(self, path: Path):
        self.db = load_db_npz(path)
        self.names = sorted(self.db.keys())
        self.mat = np.stack([self.db[n] for n in self.names]) if self.names else None

    def match(self, emb: np.ndarray) -> MatchResult:
        if self.mat is None:
            return MatchResult(None, 1.0, 0.0, False)

        sims = self.mat @ emb.reshape(-1, 1)
        if sims.ndim > 1:
            sims = sims.flatten()
        i = int(np.argmax(sims))
        sim = float(sims[i])
        dist = 1.0 - sim
        ok = dist <= self.dist_thresh

        return MatchResult(self.names[i] if ok else None, dist, sim, ok)

# -------------------------
# Demo
# -------------------------

def main():
    db_path = Path("data/db/face_db.npz")

    det = HaarFaceMesh5pt()
    embedder = ArcFaceEmbedderONNX()
    matcher = FaceDBMatcher(load_db_npz(db_path), dist_thresh=0.34)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not available")

    print("Recognize: q quit | r reload | +/- thresh | d debug")

    show_debug = False
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        vis = frame.copy()
        faces = det.detect(frame)

        for f in faces:
            aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
            emb = embedder.embed(aligned)
            mr = matcher.match(emb)

            color = (0, 255, 0) if mr.accepted else (0, 0, 255)
            label = mr.name if mr.name else "Unknown"

            cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), color, 2)
            cv2.putText(vis, label, (f.x1, f.y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("recognize", vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            matcher.reload_from(db_path)
        elif key in (ord("+"), ord("=")):
            matcher.dist_thresh = min(1.2, matcher.dist_thresh + 0.01)
        elif key == ord("-"):
            matcher.dist_thresh = max(0.05, matcher.dist_thresh - 0.01)
        elif key == ord("d"):
            show_debug = not show_debug

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
