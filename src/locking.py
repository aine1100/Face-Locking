import cv2
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    import mediapipe as mp
except ImportError:
    mp = None

from .recognize import FaceDBMatcher, ArcFaceEmbedderONNX, HaarFaceMesh5pt, load_db_npz, align_face_5pt

@dataclass
class ActionRecord:
    timestamp: float
    action_type: str
    description: str

class FaceLockingSystem:
    def __init__(self, db_path: Path, model_path: str = "models/arcface.onnx"):
        self.db_path = db_path
        self.matcher = FaceDBMatcher(load_db_npz(db_path), dist_thresh=0.35)
        self.embedder = ArcFaceEmbedderONNX(model_path=model_path)
        self.detector = HaarFaceMesh5pt()
        
        # Locking state
        self.selected_name: Optional[str] = None
        self.is_locked = False
        self.last_seen_time = 0
        self.lock_timeout = 3.0 # seconds
        
        # Tracking Stability
        self.smooth_box: Optional[np.ndarray] = None # [x1, y1, x2, y2]
        self.prev_centroid: Optional[np.ndarray] = None
        self.ema_alpha = 0.5 # Smoothing for box
        
        # Action Detection State
        self.blink_frames = 0
        self.blink_req_frames = 2 
        self.is_blinking = False
        self.blink_threshold = 0.20 
        
        self.smile_mar_buffer: List[float] = []
        self.smile_buffer_size = 5
        self.is_smiling = False
        self.smile_threshold = 0.45 
        
        self.history: List[ActionRecord] = []
        self.history_file: Optional[Path] = None

        # Landmarks indices for EAR/MAR
        self.L_EYE = [33, 160, 158, 133, 153, 144]
        self.R_EYE = [362, 385, 387, 263, 373, 380]
        self.MOUTH = [61, 291, 0, 17]

        # Lighting Robustness (CLAHE)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def normalize_lighting(self, aligned_img: np.ndarray) -> np.ndarray:
        """Apply CLAHE to improve feature extraction in bad lighting."""
        # Convert to YUV to separate luminance from color
        yuv = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2YUV)
        # Apply CLAHE to the Y channel
        yuv[:, :, 0] = self.clahe.apply(yuv[:, :, 0])
        # Convert back to BGR
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    def set_selected_name(self, name: str):
        if name != self.selected_name:
            self.selected_name = name
            self.is_locked = False
            self.history = []
            self.history_file = None
            self.smooth_box = None
            self.prev_centroid = None
            print(f"Selected identity: {name}")

    def start_history_logging(self):
        if not self.selected_name: return
        ts = time.strftime("%Y%m%d%H%M%S")
        filename = f"{self.selected_name.lower()}_history_{ts}.txt"
        self.history_file = Path("data/history") / filename
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self.record_action("LOCK_START", f"System locked onto {self.selected_name}")

    def record_action(self, action_type: str, description: str):
        now = time.time()
        record = ActionRecord(now, action_type, description)
        self.history.append(record)
        
        if self.history_file:
            time_str = time.strftime("%H:%M:%S", time.localtime(now))
            milli = int((now % 1) * 1000)
            with open(self.history_file, "a") as f:
                f.write(f"[{time_str}.{milli:03d}] {action_type}: {description}\n")

    def calculate_ear(self, landmarks, indices):
        p2, p3, p5, p6 = [landmarks[indices[i]] for i in [1, 2, 4, 5]]
        p1, p4 = [landmarks[indices[i]] for i in [0, 3]]
        d_v1 = np.linalg.norm(np.array([p2.x, p2.y]) - np.array([p6.x, p6.y]))
        d_v2 = np.linalg.norm(np.array([p3.x, p3.y]) - np.array([p5.x, p5.y]))
        d_h = np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p4.x, p4.y]))
        return (d_v1 + d_v2) / (2.0 * d_h)

    def calculate_mar(self, landmarks, indices):
        p1, p2, p3, p4 = [landmarks[indices[i]] for i in range(4)]
        d_v = np.linalg.norm(np.array([p3.x, p3.y]) - np.array([p4.x, p4.y]))
        d_h = np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))
        return d_v / d_h

    def process_frame(self, frame: np.ndarray):
        H, W = frame.shape[:2]
        faces = self.detector.detect(frame)
        
        locked_face_found = False
        target_f = None
        
        # 1. Identity Verification
        curr_thresh = self.matcher.dist_thresh
        if self.is_locked:
            curr_thresh *= 1.4 # Sticky lock
            
        for f in faces:
            aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
            
            # V5: Normalize lighting before embedding
            normalized = self.normalize_lighting(aligned)
            emb = self.embedder.embed(normalized)
            
            if self.matcher.mat is None: continue
            sims = self.matcher.mat @ emb.reshape(-1, 1)
            if sims.ndim > 1: sims = sims.flatten()
            i = int(np.argmax(sims))
            dist = 1.0 - float(sims[i])
            name = self.matcher.names[i]
            
            if dist <= curr_thresh and name == self.selected_name:
                locked_face_found = True
                target_f = f
                self.is_locked = True
                self.last_seen_time = time.time()
                if not self.history_file:
                    self.start_history_logging()
                break
        
        # 2. Stable Tracking
        if not locked_face_found and self.is_locked:
            now = time.time()
            if now - self.last_seen_time < self.lock_timeout:
                if self.prev_centroid is not None and faces:
                    best_dist = float('inf')
                    for f in faces:
                        curr_centroid = np.array([(f.x1 + f.x2)/2, (f.y1 + f.y2)/2])
                        dist = np.linalg.norm(curr_centroid - self.prev_centroid)
                        if dist < 150 and dist < best_dist: 
                            best_dist = dist
                            target_f = f
                            locked_face_found = True
                            self.last_seen_time = time.time() 
            else:
                self.is_locked = False
                self.record_action("LOCK_RELEASE", "Face lost for too long")
                self.history_file = None
                self.smooth_box = None

        # 3. Post-Process Locked Face
        if target_f:
            # Box Smoothing
            curr_box = np.array([target_f.x1, target_f.y1, target_f.x2, target_f.y2], dtype=np.float32)
            if self.smooth_box is None: self.smooth_box = curr_box
            else: self.smooth_box = self.ema_alpha * self.smooth_box + (1.0 - self.ema_alpha) * curr_box
            
            # Centroid & Movement
            curr_centroid = np.array([(self.smooth_box[0] + self.smooth_box[2])/2, 
                                      (self.smooth_box[1] + self.smooth_box[3])/2])
            if self.prev_centroid is not None:
                dx = curr_centroid[0] - self.prev_centroid[0]
                if dx > 20: self.record_action("MOVE", "Moved Right")
                elif dx < -20: self.record_action("MOVE", "Moved Left")
            self.prev_centroid = curr_centroid
            
            # Action Detection
            aligned_action, _ = align_face_5pt(frame, target_f.kps, out_size=(224, 224))
            rgb_aligned = cv2.cvtColor(aligned_action, cv2.COLOR_BGR2RGB)
            
            res = self.detector.mesh.process(rgb_aligned) 
            if res.multi_face_landmarks:
                lms = res.multi_face_landmarks[0].landmark
                ear_l = self.calculate_ear(lms, self.L_EYE)
                ear_r = self.calculate_ear(lms, self.R_EYE)
                ear = (ear_l + ear_r) / 2.0
                
                if ear < self.blink_threshold:
                    self.blink_frames += 1
                else:
                    if self.blink_frames >= self.blink_req_frames:
                        self.record_action("BLINK", "Eye blink detected")
                    self.blink_frames = 0
                self.is_blinking = (self.blink_frames > 0)
                
                mar = self.calculate_mar(lms, self.MOUTH)
                self.smile_mar_buffer.append(mar)
                if len(self.smile_mar_buffer) > self.smile_buffer_size:
                    self.smile_mar_buffer.pop(0)
                
                avg_mar = sum(self.smile_mar_buffer) / len(self.smile_mar_buffer)
                if avg_mar > self.smile_threshold:
                    if not self.is_smiling:
                        self.is_smiling = True
                        self.record_action("EXPRESSION", "Smile/Laugh detected")
                else:
                    self.is_smiling = False

        return target_f, faces

def main():
    db_path = Path("data/db/face_db.npz")
    system = FaceLockingSystem(db_path)
    
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Camera 1 not available. Trying camera 0...")
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: No camera available"); return

    names = sorted(system.matcher.db.keys())
    selected_idx = 0
    if names: system.set_selected_name(names[selected_idx])

    print("--- Face Locking System v4 (Visual Focus) ---")
    print("Keys: n/p: Cycle names | +/-: Sensitivity | q: Quit")
    
    cv2.namedWindow("Face Locking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face Locking", 1280, 720) # Set a larger default size
    
    while True:
        ok, frame = cap.read()
        if not ok: break
        H, W = frame.shape[:2]
        
        target_f, all_faces = system.process_frame(frame)
        
        # --- UI Construction ---
        # 1. Background Blur (if locked)
        vis = frame.copy()
        if system.is_locked and system.smooth_box is not None:
            # Create a blurred version of the frame
            blurred_frame = cv2.GaussianBlur(frame, (25, 25), 0)
            
            # Create a mask for the focused area (person)
            mask = np.zeros((H, W), dtype=np.uint8)
            bx = system.smooth_box.astype(int)
            # Expand box slightly for a better "focus" area
            pad = 100
            x1, y1 = max(0, bx[0]-pad), max(0, bx[1]-pad*2)
            x2, y2 = min(W, bx[2]+pad), min(H, bx[3]+pad)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            # Soften mask edges
            mask = cv2.GaussianBlur(mask, (51, 51), 0)
            mask_3c = cv2.merge([mask, mask, mask]) / 255.0
            
            # Blend blurred background with clear foreground
            vis = (mask_3c * frame + (1 - mask_3c) * blurred_frame).astype(np.uint8)

        if not names:
            cv2.putText(vis, "DATABASE EMPTY - ENROLL FIRST", (50, H//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            for f in all_faces:
                label, color = "Unknown", (0, 0, 255) # Red for Unknown
                
                is_target = target_f and f.x1 == target_f.x1 and f.y1 == target_f.y1
                
                if is_target:
                    color = (0, 255, 0) if system.is_locked else (0, 255, 255)
                    status = "LOCKED" if system.is_locked else "SEARCHING"
                    
                    # Distance metric for UI (use normalized embedding for consistency)
                    aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
                    normalized = system.normalize_lighting(aligned)
                    emb = system.embedder.embed(normalized)
                    
                    sims = system.matcher.mat @ emb.reshape(-1, 1)
                    dist = 1.0 - float(np.max(sims))
                    
                    bx = system.smooth_box.astype(int) if system.smooth_box is not None else [f.x1, f.y1, f.x2, f.y2]
                    # Draw thicker box for locked person
                    cv2.rectangle(vis, (bx[0], bx[1]), (bx[2], bx[3]), color, 4)
                    
                    # Readability helper: subtle background for text
                    txt = f"[{status}] {system.selected_name} ({dist:.2f})"
                    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(vis, (bx[0], bx[1]-th-15), (bx[0]+tw, bx[1]), (0,0,0), -1)
                    cv2.putText(vis, txt, (bx[0], bx[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                else:
                    # Recognition for other faces (V5 lighting robust)
                    aligned_other, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
                    norm_other = system.normalize_lighting(aligned_other)
                    emb_other = system.embedder.embed(norm_other)
                    mr_other = system.matcher.match(emb_other)
                    
                    if mr_other.accepted:
                        label = mr_other.name
                        color = (0, 165, 255) # Orange
                    
                    # Draw thinner box for others
                    cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), color, 2)
                    cv2.putText(vis, label, (f.x1, f.y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Overlay status
        total, target_name = len(names), system.selected_name if system.selected_name else "N/A"
        curr_idx_display = selected_idx + 1 if total > 0 else 0
        
        # Subtle status bar
        cv2.rectangle(vis, (0, 0), (W, 80), (0,0,0), -1)
        cv2.putText(vis, f"Target: {target_name} ({curr_idx_display}/{total})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, f"Thresh: {system.matcher.dist_thresh:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        if system.is_locked:
            y_off = 120
            if system.is_blinking:
                cv2.putText(vis, "BLINKING", (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 50, 50), 2); y_off += 25
            if system.is_smiling:
                cv2.putText(vis, "SMILING", (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 50), 2)

        cv2.imshow("Face Locking", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('n') and len(names) > 1:
            selected_idx = (selected_idx + 1) % len(names); system.set_selected_name(names[selected_idx])
        elif key == ord('p') and len(names) > 1:
            selected_idx = (selected_idx - 1) % len(names); system.set_selected_name(names[selected_idx])
        elif key in (ord('+'), ord('=')):
            system.matcher.dist_thresh = min(0.8, system.matcher.dist_thresh + 0.02)
        elif key == ord('-'):
            system.matcher.dist_thresh = max(0.1, system.matcher.dist_thresh - 0.02)

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
