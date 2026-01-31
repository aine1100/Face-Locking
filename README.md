# Face Locking & Recognition with ONNX

A real-time face recognition and **face locking** system using ONNX Runtime, OpenCV, and MediaPipe. This system can lock onto a specific identity, track their consistent movement, detect actions (blinks, smiles), and log history automatically.

## Features

- **Real-time face detection** using Haar Cascade + MediaPipe FaceMesh
- **5-point landmark extraction** (eyes, nose, mouth corners)
- **Face alignment** to 112x112 standard format
- **ArcFace embeddings** via ONNX Runtime (CPU-friendly)
- **Face enrollment** with template storage
- **Multi-face recognition** with configurable threshold
- **Face Locking** - Focus and lock onto a specific identity
- **Action Detection** - Detect blinks, smiles, and movements while locked
- **History Logs** - Automatic recording of actions with timestamps

---

## Installation

### 1. Clone & Setup Environment

```bash
cd face-recogn-onnx
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the ArcFace ONNX Model

Download the model to the `models/` folder:

**Windows (PowerShell):**
```powershell
# Create models directory if it doesn't exist
mkdir models -Force
cd models
Invoke-WebRequest -Uri "https://huggingface.co/garavv/arcface-onnx/resolve/main/arc.onnx?download=true" -OutFile "arcface.onnx"
cd ..
```

**Linux/Mac:**
```bash
mkdir -p models
cd models
curl -L "https://huggingface.co/garavv/arcface-onnx/resolve/main/arc.onnx?download=true" -o arcface.onnx
cd ..
```

---

## Usage

All modules are run as Python modules from the project root directory.

### Test Camera

Verify your camera is working:

```bash
python -m src.camera
```

Press `q` to quit.

### Face Detection (Haar)

Basic face detection with bounding boxes:

```bash
python -m src.detect
```

### Landmark Detection

Haar + MediaPipe FaceMesh 5-point landmarks:

```bash
python -m src.landmarks
```

### Face Alignment

Detect, extract landmarks, and align faces to 112x112:

```bash
python -m src.align
```

**Controls:**
- `q` - Quit
- `s` - Save aligned face to `data/debug_aligned/`

### Embedding Demo

View face embeddings as a heatmap visualization:

```bash
python -m src.embed
```

**Controls:**
- `q` - Quit
- `p` - Print embedding stats to terminal

### Enroll a Person

Capture face samples and create a template:

```bash
python -m src.enroll
```

You'll be prompted to enter a name. Then:

**Controls:**
- `SPACE` - Capture one sample
- `a` - Toggle auto-capture mode
- `s` - Save enrollment to database
- `r` - Reset new samples
- `q` - Quit

Enrolled data is saved to:
- `data/db/face_db.npz` - Embeddings
- `data/db/face_db.json` - Metadata
- `data/enroll/<name>/` - Aligned face crops

### Face Recognition

Real-time recognition against enrolled faces:

```bash
python -m src.recognize
```

**Controls:**
- `q` - Quit
- `r` - Reload database
- `+`/`-` - Adjust distance threshold
- `d` - Toggle debug overlay

### Face Locking & Action Detection

Lock onto a specific identity and track their actions (blinks, smiles, movements):

```bash
python -m src.locking
```

**Controls:**
- `q` - Quit
- `n` - Next enrolled identity
- `p` - Previous enrolled identity

**How it works:**
1. **Selection**: Use `n`/`p` to select the person to lock.
2. **Locking**: When the selected person is recognized, the system "locks" (green box).
3. **Persistence**: The lock stays even if recognition fails briefly (up to 3 seconds), as long as the face is tracked.
4. **Action Logging**: While locked, every blink, smile, or movement is saved to `data/history/<name>_history_<timestamp>.txt`.

---

## Pipeline Overview

```
Camera Frame
    │
    ▼
┌──────────────────┐
│  Haar Cascade    │  → Fast face detection
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ MediaPipe Mesh   │  → 5-point landmarks (eyes, nose, mouth)
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ Affine Alignment │  → Warp to 112x112 standard
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ ArcFace ONNX     │  → 512-dim L2-normalized embedding
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ Cosine Matching  │  → Compare to enrolled templates
└──────────────────┘
    │
    ▼
  Identity Label
```

---

## Project Structure

```
face-recogn-onnx/
├── models/
│   └── embedder_arcface.onnx    # ArcFace model (download required)
├── data/
│   ├── db/                      # Enrolled face database
│   └── enroll/                  # Saved face crops per person
├── src/
│   ├── camera.py                # Camera test
│   ├── detect.py                # Haar face detection
│   ├── landmarks.py             # 5-point landmark detection
│   ├── haar_5pt.py              # Combined Haar + FaceMesh detector
│   ├── align.py                 # Face alignment demo
│   ├── embed.py                 # Embedding visualization
│   ├── enroll.py                # Face enrollment tool
│   ├── recognize.py             # Real-time recognition
│   └── locking.py               # Face locking & Action detection
├── data/
│   ├── db/                      # Enrolled face database
│   ├── enroll/                  # Saved face crops per person
│   └── history/                 # Action history logs (.txt)
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.8+
- Webcam/camera
- ~500MB disk space for models

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Camera not opening | Try different index: `cv2.VideoCapture(1)` or `(2)` |
| MediaPipe import error | `pip install mediapipe==0.10.21` |
| Model not found | Ensure `models/embedder_arcface.onnx` exists |
| Recognition not working | Enroll at least one person first with `src.enroll` |

---

## License

MIT
