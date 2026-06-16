"""
Mememic Server — FastAPI backend for gesture recognition.

Receives webcam frames from a client, runs MediaPipe + trained model,
returns the predicted gesture name.

Usage:
  source venv/bin/activate
  python server.py

Then open http://<your-ip>:8000 in a browser on any device on the same network.
"""

import io
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
from fastapi.staticfiles import StaticFiles
import uvicorn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MEME_DIR = os.path.join(BASE_DIR, "memes")

app = FastAPI(title="Mememic")

# ── Load models ──────────────────────────────────────────────────────────
ID_TO_LABEL = None
MLP_MODEL = None
GRU_MODEL = None
MOTION_BUFFER = deque(maxlen=60)
landmarker = None


def load_models():
    global ID_TO_LABEL, MLP_MODEL, GRU_MODEL, landmarker

    # Labels
    label_path = os.path.join(MODEL_DIR, "labels.json")
    if os.path.exists(label_path):
        with open(label_path) as f:
            ID_TO_LABEL = json.load(f)
            ID_TO_LABEL = {int(k): v for k, v in ID_TO_LABEL.items()}
        print(f"  Labels loaded: {len(ID_TO_LABEL)} classes")
    else:
        print("  ⚠️  No labels.json found")
        return False

    # MLP
    mlp_path = os.path.join(MODEL_DIR, "gesture_classifier.pt")
    if os.path.exists(mlp_path):
        import torch
        import torch.nn as nn
        n_classes = len(ID_TO_LABEL)
        MLP_MODEL = nn.Sequential(
            nn.Linear(63, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )
        MLP_MODEL.load_state_dict(torch.load(mlp_path, map_location="cpu"))
        MLP_MODEL.eval()
        print(f"  MLP loaded ({n_classes} classes)")

    # GRU
    gru_path = os.path.join(MODEL_DIR, "motion_classifier.pt")
    if os.path.exists(gru_path):
        import torch
        import torch.nn as nn

        class GRUClassifier(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_classes):
                super().__init__()
                self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
                self.fc = nn.Linear(hidden_dim * 2, num_classes)
            def forward(self, x, lengths):
                from torch.nn.utils.rnn import pack_padded_sequence
                packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
                _, h_n = self.gru(packed)
                h_fwd = h_n[-2, :, :]
                h_bwd = h_n[-1, :, :]
                h = torch.cat([h_fwd, h_bwd], dim=1)
                return self.fc(h)

        n_classes = len(ID_TO_LABEL)
        GRU_MODEL = GRUClassifier(63, 64, n_classes)
        GRU_MODEL.load_state_dict(torch.load(gru_path, map_location="cpu"))
        GRU_MODEL.eval()
        print(f"  GRU loaded")

    # MediaPipe HandLandmarker
    model_path = os.path.join(BASE_DIR, "hand_landmarker.task")
    if not os.path.exists(model_path):
        print("  Downloading hand_landmarker model...")
        import urllib.request
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        urllib.request.urlretrieve(url, model_path)
        print("  Downloaded.")

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    print(f"  MediaPipe HandLandmarker ready")

    return True


def normalize_landmarks(landmarks):
    pts = np.array(landmarks, dtype=np.float32)
    wrist = pts[0]
    centered = pts - wrist
    scale = np.max(np.linalg.norm(centered, axis=1))
    if scale > 0:
        centered /= scale
    return centered.flatten().tolist()


def classify_frame(rgb_image):
    """Run inference on a single RGB frame. Returns (gesture_name, confidence)."""
    global MOTION_BUFFER

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None, 0.0

    # Use first detected hand
    hand = result.hand_landmarks[0]
    landmarks = [(lm.x, lm.y, lm.z) for lm in hand]
    norm = normalize_landmarks(landmarks)

    # Feed into motion buffer
    MOTION_BUFFER.append(norm)

    # Try GRU (motion) first if confident
    if GRU_MODEL is not None and len(MOTION_BUFFER) >= 5:
        import torch
        seq = list(MOTION_BUFFER)
        x = torch.tensor([seq], dtype=torch.float32)
        lengths = [len(seq)]
        with torch.no_grad():
            out = GRU_MODEL(x, lengths)
            probs = torch.softmax(out, dim=1)
            conf, pred = probs.max(dim=1)
        if conf.item() >= 0.7:
            name = ID_TO_LABEL.get(pred.item(), "unknown")
            return name, conf.item()

    # Fall back to MLP (static)
    if MLP_MODEL is not None:
        import torch
        x = torch.tensor([norm], dtype=torch.float32)
        with torch.no_grad():
            out = MLP_MODEL(x)
            probs = torch.softmax(out, dim=1)
            conf, pred = probs.max(dim=1)
        name = ID_TO_LABEL.get(pred.item(), "unknown")
        return name, conf.item()

    return None, 0.0


# ── Routes ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(INDEX_HTML)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Receive a webcam frame, return predicted gesture."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"gesture": None, "confidence": 0.0})

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gesture, confidence = classify_frame(rgb)

    return JSONResponse({
        "gesture": gesture,
        "confidence": round(confidence, 3),
    })


@app.get("/memes/{name}")
async def get_meme(name: str):
    """Serve meme images."""
    # Try exact match
    for ext in [".png", ".jpg", ".jpeg", ".gif"]:
        path = os.path.join(MEME_DIR, f"{name}{ext}")
        if os.path.exists(path):
            return FileResponse(path, media_type=f"image/{ext[1:]}")
    # Try case-insensitive
    for fname in os.listdir(MEME_DIR):
        if fname.lower().startswith(name.lower()):
            path = os.path.join(MEME_DIR, fname)
            ext = os.path.splitext(fname)[1][1:]
            return FileResponse(path, media_type=f"image/{ext}")
    return JSONResponse({"error": "not found"}, status_code=404)


@app.get("/gestures")
async def list_gestures():
    """Return list of all known gestures."""
    if ID_TO_LABEL:
        return JSONResponse(list(ID_TO_LABEL.values()))
    return JSONResponse([])


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
    .info { margin-top: 20px; text-align: center; color: #888; }
    .info a { color: #5af; }
    #status { color: #888; margin-top: 10px; font-size: 0.9em; }
    .controls { margin: 15px 0; display: flex; gap: 10px; }
    button {
      background: #333; color: #eee; border: 1px solid #555;
      padding: 8px 20px; border-radius: 6px; cursor: pointer; font-size: 0.9em;
    }
    button:hover { background: #444; }
    button.active { background: #4a4; border-color: #4a4; }
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
    <div id="gesture-label">Waiting...</div>
  </div>
  <div class="info">
    <p>Connect from any device on your network</p>
    <p id="status">Loading...</p>
  </div>

  <script>
    const video = document.getElementById('webcam');
    const overlay = document.getElementById('meme-overlay');
    const label = document.getElementById('gesture-label');
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
        // Flip horizontally to match mirror view
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

            if (gesture) {
              if (gesture === lastGesture) {
                stableCount++;
              } else {
                lastGesture = gesture;
                stableCount = 0;
                overlay.style.display = 'none';
              }

              if (stableCount >= STABLE_THRESHOLD) {
                label.textContent = gesture + ' (' + (data.confidence * 100).toFixed(0) + '%)';
                overlay.src = '/memes/' + encodeURIComponent(gesture);
                overlay.style.display = 'block';
              } else {
                label.textContent = gesture;
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

    // Auto-start
    startCamera();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    print("=" * 60)
    print("  Mememic Server")
    print("=" * 60)
    print()

    if not load_models():
        print("❌ Failed to load models. Run python train.py first.")
        sys.exit(1)

    print()
    print("  Server starting...")
    print("  Open http://localhost:8001 in a browser")
    print("  Other devices on your network: http://<YOUR_IP>:8001")
    print()
    print("  Press Ctrl+C to stop")
    print()

    uvicorn.run(app, host="0.0.0.0", port=8001)
