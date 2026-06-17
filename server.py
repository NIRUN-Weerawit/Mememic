"""
Mememic Server — FastAPI backend for hand gesture + face expression recognition.

Runs MediaPipe HandLandmarker + FaceLandmarker on each frame.
Returns gesture name, confidence, and face expression data.
"""

import json
import os
import sys
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import uvicorn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MEME_DIR = os.path.join(BASE_DIR, "memes")

app = FastAPI(title="Mememic")

# ── Models ──────────────────────────────────────────────────────────────
ID_TO_LABEL = None
HAND_MODEL = None
FACE_MODEL = None
COMBINED_MODEL = None
GRU_MODEL = None
MOTION_BUFFER = deque(maxlen=60)
hand_landmarker = None
face_landmarker = None

BLEND_SHAPE_NAMES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight", "cheekPuff", "cheekSquintLeft",
    "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft",
    "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft",
    "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward",
    "jawLeft", "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft",
    "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
    "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft",
    "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft",
    "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight", "tongueOut",
]


def load_models():
    global ID_TO_LABEL, HAND_MODEL, FACE_MODEL, COMBINED_MODEL, GRU_MODEL
    global hand_landmarker, face_landmarker

    label_path = os.path.join(MODEL_DIR, "labels.json")
    if not os.path.exists(label_path):
        print("  ⚠️  No labels.json found")
        return False

    with open(label_path) as f:
        ID_TO_LABEL = json.load(f)
        ID_TO_LABEL = {int(k): v for k, v in ID_TO_LABEL.items()}
    print(f"  Labels loaded: {len(ID_TO_LABEL)} classes")

    import torch
    import torch.nn as nn
    n_classes = len(ID_TO_LABEL)

    # Hand MLP
    path = os.path.join(MODEL_DIR, "hand_classifier.pt")
    if os.path.exists(path):
        HAND_MODEL = nn.Sequential(
            nn.Linear(63, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )
        HAND_MODEL.load_state_dict(torch.load(path, map_location="cpu"))
        HAND_MODEL.eval()
        print(f"  ✅ Hand MLP loaded")

    # Face MLP
    path = os.path.join(MODEL_DIR, "face_classifier.pt")
    if os.path.exists(path):
        FACE_MODEL = nn.Sequential(
            nn.Linear(53, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )
        FACE_MODEL.load_state_dict(torch.load(path, map_location="cpu"))
        FACE_MODEL.eval()
        print(f"  ✅ Face MLP loaded")

    # Combined fusion
    path = os.path.join(MODEL_DIR, "combined_classifier.pt")
    if os.path.exists(path):
        class FusionMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.hand_net = nn.Sequential(nn.Linear(63, 64), nn.ReLU(), nn.Dropout(0.2))
                self.face_net = nn.Sequential(nn.Linear(53, 64), nn.ReLU(), nn.Dropout(0.2))
                self.fusion = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, n_classes))
            def forward(self, hand, face):
                h = self.hand_net(hand)
                f = self.face_net(face)
                return self.fusion(torch.cat([h, f], dim=1))
        COMBINED_MODEL = FusionMLP()
        COMBINED_MODEL.load_state_dict(torch.load(path, map_location="cpu"))
        COMBINED_MODEL.eval()
        print(f"  ✅ Combined model loaded")

    # GRU motion
    path = os.path.join(MODEL_DIR, "motion_classifier.pt")
    if os.path.exists(path):
        class GRUClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = nn.GRU(63, 64, batch_first=True, bidirectional=True)
                self.fc = nn.Linear(128, n_classes)
            def forward(self, x, lengths):
                from torch.nn.utils.rnn import pack_padded_sequence
                packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
                _, h_n = self.gru(packed)
                h = torch.cat([h_n[-2], h_n[-1]], dim=1)
                return self.fc(h)
        GRU_MODEL = GRUClassifier()
        GRU_MODEL.load_state_dict(torch.load(path, map_location="cpu"))
        GRU_MODEL.eval()
        print(f"  ✅ GRU motion model loaded")

    # MediaPipe models
    for model_name in ["hand_landmarker", "face_landmarker"]:
        path = os.path.join(BASE_DIR, f"{model_name}.task")
        if not os.path.exists(path):
            print(f"  Downloading {model_name}...")
            import urllib.request
            url = f"https://storage.googleapis.com/mediapipe-models/{model_name}/{model_name}/float16/latest/{model_name}.task"
            urllib.request.urlretrieve(url, path)

    hand_base = python.BaseOptions(model_asset_path=os.path.join(BASE_DIR, "hand_landmarker.task"))
    hand_opts = vision.HandLandmarkerOptions(
        base_options=hand_base, running_mode=vision.RunningMode.IMAGE,
        num_hands=2, min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6, min_tracking_confidence=0.5,
    )
    hand_landmarker = vision.HandLandmarker.create_from_options(hand_opts)
    print(f"  ✅ HandLandmarker ready")

    face_base = python.BaseOptions(model_asset_path=os.path.join(BASE_DIR, "face_landmarker.task"))
    face_opts = vision.FaceLandmarkerOptions(
        base_options=face_base, running_mode=vision.RunningMode.IMAGE,
        num_faces=1, min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5, min_tracking_confidence=0.5,
        output_face_blendshapes=True,
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(face_opts)
    print(f"  ✅ FaceLandmarker ready")

    return True


def normalize_hand(landmarks):
    pts = np.array(landmarks, dtype=np.float32)
    wrist = pts[0]
    centered = pts - wrist
    scale = np.max(np.linalg.norm(centered, axis=1))
    if scale > 0:
        centered /= scale
    return centered.flatten().tolist()


def extract_face(blendshapes):
    if not blendshapes or not blendshapes[0]:
        return None
    bs = [bs.score for bs in blendshapes[0]]
    # Handle old data with 52 blend shapes (missing _neutral)
    if len(bs) == 52:
        bs = [0.0] + bs
    return bs if len(bs) == 53 else None


def classify_frame(rgb_image):
    """Run inference on a single RGB frame. Returns (gesture, confidence, mode, face_data)."""
    global MOTION_BUFFER

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    hand_result = hand_landmarker.detect(mp_image)
    face_result = face_landmarker.detect(mp_image)

    hand_lm = None
    if hand_result.hand_landmarks:
        hand_lm = [(lm.x, lm.y, lm.z) for lm in hand_result.hand_landmarks[0]]

    face_bs = extract_face(face_result.face_blendshapes)

    # Build face expression data for frontend
    face_data = None
    if face_bs is not None:
        face_data = {
            "smile": round(face_bs[44], 3),
            "brow_down": round(face_bs[1], 3),
            "jaw_open": round(face_bs[25], 3),
            "eye_blink_l": round(face_bs[9], 3),
            "eye_blink_r": round(face_bs[10], 3),
        }

    if hand_lm is None:
        return None, 0.0, "none", face_data

    norm = normalize_hand(hand_lm)
    MOTION_BUFFER.append(norm)

    gesture = None
    conf = 0.0
    mode = "none"

    # Try GRU motion
    if GRU_MODEL and len(MOTION_BUFFER) >= 5:
        import torch
        seq = list(MOTION_BUFFER)
        x = torch.tensor([seq], dtype=torch.float32)
        lengths = [len(seq)]
        with torch.no_grad():
            out = GRU_MODEL(x, lengths)
            probs = torch.softmax(out, dim=1)
            c, p = probs.max(dim=1)
        if c.item() >= 0.7:
            gesture, conf, mode = ID_TO_LABEL.get(p.item(), "unknown"), c.item(), "GRU"

    # Try combined
    if conf < 0.5 and COMBINED_MODEL and face_bs is not None:
        import torch
        h = torch.tensor([norm], dtype=torch.float32)
        f = torch.tensor([face_bs], dtype=torch.float32)
        with torch.no_grad():
            out = COMBINED_MODEL(h, f)
            probs = torch.softmax(out, dim=1)
            c, p = probs.max(dim=1)
        if c.item() > conf:
            gesture, conf, mode = ID_TO_LABEL.get(p.item(), "unknown"), c.item(), "COMBINED"

    # Try hand MLP
    if conf < 0.5 and HAND_MODEL:
        import torch
        x = torch.tensor([norm], dtype=torch.float32)
        with torch.no_grad():
            out = HAND_MODEL(x)
            probs = torch.softmax(out, dim=1)
            c, p = probs.max(dim=1)
        if c.item() > conf:
            gesture, conf, mode = ID_TO_LABEL.get(p.item(), "unknown"), c.item(), "HAND"

    # Try face MLP
    if conf < 0.5 and FACE_MODEL and face_bs is not None:
        import torch
        x = torch.tensor([face_bs], dtype=torch.float32)
        with torch.no_grad():
            out = FACE_MODEL(x)
            probs = torch.softmax(out, dim=1)
            c, p = probs.max(dim=1)
        if c.item() > conf:
            gesture, conf, mode = ID_TO_LABEL.get(p.item(), "unknown"), c.item(), "FACE"

    return gesture, conf, mode, face_data


# ── Routes ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(INDEX_HTML)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"gesture": None, "confidence": 0.0, "mode": "none", "face": None})

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gesture, confidence, mode, face_data = classify_frame(rgb)

    return JSONResponse({
        "gesture": gesture,
        "confidence": round(confidence, 3),
        "mode": mode,
        "face": face_data,
    })


@app.get("/memes/{name}")
async def get_meme(name: str):
    for ext in [".png", ".jpg", ".jpeg", ".gif"]:
        path = os.path.join(MEME_DIR, f"{name}{ext}")
        if os.path.exists(path):
            return FileResponse(path, media_type=f"image/{ext[1:]}")
    for fname in os.listdir(MEME_DIR):
        if fname.lower().startswith(name.lower()):
            path = os.path.join(MEME_DIR, fname)
            ext = os.path.splitext(fname)[1][1:]
            return FileResponse(path, media_type=f"image/{ext}")
    return JSONResponse({"error": "not found"}, status_code=404)


@app.get("/gestures")
async def list_gestures():
    if ID_TO_LABEL:
        return JSONResponse(list(ID_TO_LABEL.values()))
    return JSONResponse([])


@app.get("/memes")
async def list_memes():
    memes = []
    if os.path.exists(MEME_DIR):
        for fname in sorted(os.listdir(MEME_DIR)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                name = os.path.splitext(fname)[0]
                ext = os.path.splitext(fname)[1][1:]
                memes.append({"name": name, "file": fname, "url": f"/memes/{name}", "type": f"image/{ext}"})
    return JSONResponse(memes)


# ── HTML frontend ────────────────────────────────────────────────────────

INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Mememic</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      background: #111; color: #eee; font-family: 'Segoe UI', sans-serif;
      display: flex; flex-direction: column; align-items: center; min-height: 100vh;
    }
    h1 { margin: 20px 0 10px; font-size: 2em; background: linear-gradient(90deg, #f55, #5f5, #55f); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .container { position: relative; width: 640px; height: 480px; border-radius: 12px; overflow: hidden; border: 2px solid #333; }
    #webcam { width: 100%; height: 100%; object-fit: cover; transform: scaleX(-1); }
    #meme-overlay {
      position: absolute; top: 20px; right: 20px; width: 200px; height: 150px;
      border-radius: 8px; border: 2px solid rgba(255,255,255,0.3);
      display: none; object-fit: contain; background: rgba(0,0,0,0.5);
    }
    #gesture-label {
      position: absolute; bottom: 20px; left: 20px;
      background: rgba(0,0,0,0.7); padding: 8px 16px; border-radius: 8px;
      font-size: 1.2em; font-weight: bold;
    }
    #face-info {
      position: absolute; top: 20px; left: 20px;
      background: rgba(0,0,0,0.6); padding: 6px 12px; border-radius: 6px;
      font-size: 0.75em; color: #ffc864; display: none;
    }
    .info { margin-top: 20px; text-align: center; color: #888; }
    #status { color: #888; margin-top: 10px; font-size: 0.9em; }
    .controls { margin: 15px 0; display: flex; gap: 10px; }
    button {
      background: #333; color: #eee; border: 1px solid #555;
      padding: 8px 20px; border-radius: 6px; cursor: pointer; font-size: 0.9em;
    }
    button:hover { background: #444; }
    button.active { background: #4a4; border-color: #4a4; }
    .gallery { max-width: 800px; margin: 30px auto; text-align: center; }
    .gallery h2 { color: #888; font-size: 1.2em; margin-bottom: 15px; }
    .meme-grid { display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; }
    .meme-card {
      width: 120px; border-radius: 8px; overflow: hidden;
      background: #1a1a1a; border: 1px solid #333;
      transition: transform 0.2s;
    }
    .meme-card:hover { transform: scale(1.05); border-color: #5af; }
    .meme-card img { width: 100%; height: 90px; object-fit: cover; display: block; }
    .meme-card .label {
      padding: 4px 6px; font-size: 0.7em; color: #aaa;
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
  </style>
</head>
<body>
  <h1>Mememic</h1>
  <div class="controls">
    <button id="start-btn" class="active">▶ Start</button>
    <button id="stop-btn">■ Stop</button>
  </div>
  <div class="container">
    <video id="webcam" autoplay playsinline></video>
    <img id="meme-overlay" />
    <div id="face-info">No face</div>
    <div id="gesture-label">Waiting...</div>
  </div>
  <div class="info">
    <p>Connect from any device on your network</p>
    <p id="status">Loading...</p>
  </div>

  <div class="gallery">
    <h2>Dataset — All Memes</h2>
    <div id="meme-grid" class="meme-grid"></div>
  </div>

  <script>
    fetch('/memes').then(r => r.json()).then(memes => {
      const grid = document.getElementById('meme-grid');
      memes.forEach(m => {
        const card = document.createElement('div');
        card.className = 'meme-card';
        card.innerHTML = `<img src="${m.url}" alt="${m.name}" loading="lazy"/><div class="label">${m.name}</div>`;
        grid.appendChild(card);
      });
    });

    const video = document.getElementById('webcam');
    const overlay = document.getElementById('meme-overlay');
    const label = document.getElementById('gesture-label');
    const faceInfo = document.getElementById('face-info');
    const status = document.getElementById('status');
    let stream = null;
    let interval = null;
    let lastGesture = null;
    let stableCount = 0;
    const STABLE_THRESHOLD = 3;

    async function startCamera() {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        status.textContent = '❌ Browser does not support camera access. Try Chrome/Firefox on localhost or HTTPS.';
        return;
      }
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' } });
        video.srcObject = stream;
        status.textContent = 'Camera active';
        startPolling();
      } catch (e) {
        status.textContent = 'Camera error: ' + e.message;
      }
    }

    function stopCamera() {
      if (interval) { clearInterval(interval); interval = null; }
      if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
      video.srcObject = null;
      status.textContent = 'Stopped';
    }

    function startPolling() {
      if (interval) clearInterval(interval);
      interval = setInterval(async () => {
        if (!video.videoWidth) return;

        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(video, 0, 0);
        canvas.getContext('2d').setTransform(1, 0, 0, 1, 0, 0);

        canvas.toBlob(async (blob) => {
          const formData = new FormData();
          formData.append('file', blob, 'frame.jpg');

          try {
            const res = await fetch('/predict', { method: 'POST', body: formData });
            const data = await res.json();
            const gesture = data.gesture;
            const mode = data.mode;
            const face = data.face;

            // Show face info
            if (face) {
              faceInfo.style.display = 'block';
              faceInfo.textContent = `smile:${face.smile} brow:${face.brow_down} jaw:${face.jaw_open}`;
            } else {
              faceInfo.style.display = 'none';
            }

            if (gesture) {
              if (gesture === lastGesture) {
                stableCount++;
              } else {
                lastGesture = gesture;
                stableCount = 0;
                overlay.style.display = 'none';
              }

              if (stableCount >= STABLE_THRESHOLD) {
                label.textContent = gesture + ' (' + (data.confidence * 100).toFixed(0) + '%) [' + mode + ']';
                overlay.src = '/memes/' + encodeURIComponent(gesture);
                overlay.style.display = 'block';
              } else {
                label.textContent = gesture + ' [' + mode + ']';
              }
            } else {
              lastGesture = null;
              stableCount = 0;
              label.textContent = 'No hand detected';
              overlay.style.display = 'none';
            }
          } catch (e) {
            status.textContent = 'Error: ' + e.message;
          }
        }, 'image/jpeg', 0.8);
      }, 200);
    }

    document.getElementById('start-btn').onclick = startCamera;
    document.getElementById('stop-btn').onclick = stopCamera;
    startCamera();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Mememic Server")
    parser.add_argument("--port", type=int, default=8001, help="Port to run the server on (default: 8001)")
    args = parser.parse_args()
    port = args.port

    print("=" * 60)
    print("  Mememic Server — Hand + Face")
    print("=" * 60)
    print()

    if not load_models():
        print("❌ Failed to load models. Run python train.py first.")
        sys.exit(1)

    print()
    print(f"  Server starting...")
    print(f"  Open http://localhost:{port} in a browser")
    print(f"  Other devices on your network: http://<YOUR_IP>:{port}")
    print()
    print("  Press Ctrl+C to stop")
    print()

    uvicorn.run(app, host="0.0.0.0", port=port)
