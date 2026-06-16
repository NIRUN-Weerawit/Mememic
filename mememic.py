"""
Mememic — YOLO hand detection + gesture recognition → meme display.

Uses MediaPipe HandLandmarker (new API, v0.10+) for hand tracking.
Gesture recognition:
  - Static poses: trained MLP if available, rule-based fallback
  - Motion gestures: trained GRU on sliding window of landmark sequences
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import os
import sys
import threading
import json
from collections import deque

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEME_DIR = os.path.join(BASE_DIR, "memes")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MEME_DIR, exist_ok=True)

# ── Meme loading (dynamic from memes/ directory) ─────────────────────────
# If a model is trained, its labels define the gesture set.
# Meme images are loaded from memes/<gesture_name>.png
# If no model, fall back to built-in templates.

MEME_TEMPLATES = {}  # populated dynamically


def load_memes():
    """Load meme images from memes/ directory. Returns {name: cv2_img}."""
    global MEME_TEMPLATES
    memes = {}
    if not os.path.exists(MEME_DIR):
        os.makedirs(MEME_DIR)
        return memes

    for fname in os.listdir(MEME_DIR):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            name = os.path.splitext(fname)[0]
            path = os.path.join(MEME_DIR, fname)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                memes[name] = img
    return memes


def ensure_meme_for_label(label, memes):
    """If a gesture name from the model has no meme image, generate one."""
    if label not in memes:
        # Generate a simple text meme
        img = generate_meme_image(label.replace("_", " ").title(), (100, 100, 200))
        path = os.path.join(MEME_DIR, f"{label}.png")
        img.save(path)
        memes[label] = cv2.imread(path)
    return memes


def generate_meme_image(text, color, size=(400, 300)):
    """Create a meme-style image with bold text on a dark background."""
    img = Image.new("RGB", size, (20, 20, 30))
    draw = ImageDraw.Draw(img)

    font = None
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]:
        if os.path.exists(candidate):
            font = ImageFont.truetype(candidate, 36)
            break
    if font is None:
        font = ImageFont.load_default()

    draw.rectangle([0, 0, size[0], 8], fill=color)

    lines = text.split("\n")
    y = 50
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        x = (size[0] - tw) // 2
        draw.text((x + 2, y + 2), line, fill=(0, 0, 0), font=font)
        draw.text((x, y), line, fill=(255, 255, 255), font=font)
        y += 50

    draw.text((10, size[1] - 30), "Mememic", fill=(80, 80, 80), font=font)
    return img


def pregenerate_memes():
    """Generate all meme images at startup."""
    memes = {}
    for name, (text, color) in MEME_TEMPLATES.items():
        path = os.path.join(MEME_DIR, f"{name}.png")
        if not os.path.exists(path):
            img = generate_meme_image(text, color)
            img.save(path)
        memes[name] = cv2.imread(path)
        if memes[name] is None:
            img = generate_meme_image(text, color)
            img.save(path)
            memes[name] = cv2.imread(path)
    return memes


# ── Landmark normalisation ────────────────────────────────────────────────
def normalize_landmarks(landmarks):
    """Normalize 21 landmarks to be translation- and scale-invariant. Returns 63-dim list."""
    pts = np.array(landmarks, dtype=np.float32)
    wrist = pts[0]
    centered = pts - wrist
    scale = np.max(np.linalg.norm(centered, axis=1))
    if scale > 0:
        centered /= scale
    return centered.flatten().tolist()


# ── MLP classifier (static poses) ────────────────────────────────────────
ID_TO_LABEL = None
MLP_MODEL = None


def load_mlp():
    """Load trained MLP model if available."""
    global ID_TO_LABEL, MLP_MODEL

    label_path = os.path.join(MODEL_DIR, "labels.json")
    model_path = os.path.join(MODEL_DIR, "gesture_classifier.pt")

    if not os.path.exists(model_path) or not os.path.exists(label_path):
        return False

    with open(label_path) as f:
        ID_TO_LABEL = json.load(f)
        ID_TO_LABEL = {int(k): v for k, v in ID_TO_LABEL.items()}

    import torch
    import torch.nn as nn

    n_classes = len(ID_TO_LABEL)
    MLP_MODEL = nn.Sequential(
        nn.Linear(63, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, n_classes),
    )
    MLP_MODEL.load_state_dict(torch.load(model_path, map_location="cpu"))
    MLP_MODEL.eval()
    return True


def classify_mlp(landmarks):
    """Classify static pose using trained MLP."""
    if MLP_MODEL is None:
        return None
    import torch
    norm = normalize_landmarks(landmarks)
    x = torch.tensor([norm], dtype=torch.float32)
    with torch.no_grad():
        out = MLP_MODEL(x)
        pred = out.argmax(dim=1).item()
    return ID_TO_LABEL.get(pred, "FIST")


# ── GRU classifier (motion gestures) ──────────────────────────────────────
GRU_MODEL = None
MOTION_BUFFER = deque(maxlen=60)  # sliding window of normalized 63-dim frames


def load_gru():
    """Load trained GRU motion model if available."""
    global GRU_MODEL

    model_path = os.path.join(MODEL_DIR, "motion_classifier.pt")
    if not os.path.exists(model_path):
        return False

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

    n_classes = len(ID_TO_LABEL) if ID_TO_LABEL else 13
    GRU_MODEL = GRUClassifier(63, 64, n_classes)
    GRU_MODEL.load_state_dict(torch.load(model_path, map_location="cpu"))
    GRU_MODEL.eval()
    return True


def classify_motion():
    """Classify motion from sliding window buffer. Returns (gesture_name, confidence) or None."""
    if GRU_MODEL is None or len(MOTION_BUFFER) < 5:
        return None

    import torch
    seq = list(MOTION_BUFFER)
    x = torch.tensor([seq], dtype=torch.float32)  # (1, T, 63)
    lengths = [len(seq)]

    with torch.no_grad():
        out = GRU_MODEL(x, lengths)
        probs = torch.softmax(out, dim=1)
        conf, pred = probs.max(dim=1)

    if conf.item() < 0.7:  # confidence threshold
        return None

    name = ID_TO_LABEL.get(pred.item(), "FIST")
    return name, conf.item()


# ── Rule-based classifier (fallback) ──────────────────────────────────────
def classify_gesture(landmarks):
    """Rule-based gesture classifier. Used as fallback when no trained MLP."""
    lm = landmarks

    def is_extended(tip, pip):
        return lm[tip][1] < lm[pip][1]

    thumb_up = lm[4][1] < lm[3][1]
    index_up = is_extended(8, 6)
    middle_up = is_extended(12, 10)
    ring_up = is_extended(16, 14)
    pinky_up = is_extended(20, 18)

    fingers = [index_up, middle_up, ring_up, pinky_up]
    count = sum(fingers)

    if thumb_up and count == 0:
        return "THUMBS_UP"
    if not thumb_up and count == 0:
        if (lm[4][1] - lm[2][1]) > 0.10:
            return "THUMBS_DOWN"
    if not index_up and middle_up and not ring_up and not pinky_up and not thumb_up:
        return "MIDDLE_FINGER"
    if index_up and not middle_up and not ring_up and pinky_up and not thumb_up:
        return "ROCK_ON"
    if thumb_up and index_up and not middle_up and not ring_up and pinky_up:
        return "SPIDERMAN"
    if index_up and not middle_up and not ring_up and not pinky_up and not thumb_up:
        return "POINTING"
    dist_ti = np.linalg.norm(np.array(lm[4][:2]) - np.array(lm[8][:2]))
    if dist_ti < 0.08 and index_up and not middle_up and not ring_up and not pinky_up:
        return "OK"
    if count == 0:
        return "FIST"
    elif count == 1:
        return "ONE"
    elif count == 2:
        return "PEACE"
    elif count == 3:
        return "THREE"
    elif count == 4:
        return "FOUR"
    elif count == 5:
        return "FIVE"
    return "FIST"


# ── Thread-safe shared state ──────────────────────────────────────────────
current_landmarks = None
landmarks_lock = threading.Lock()


def update_landmarks(lm_list):
    global current_landmarks
    with landmarks_lock:
        current_landmarks = lm_list


def get_landmarks():
    global current_landmarks
    with landmarks_lock:
        return current_landmarks


# ── Main loop ────────────────────────────────────────────────────────────
def main():
    print("Mememic — YOLO + MediaPipe gesture → meme")
    print("Press Q to quit.")

    # Load meme images from memes/ directory
    memes = load_memes()
    print(f"  Loaded {len(memes)} meme images from memes/")

    # Load models
    use_mlp = load_mlp()
    use_gru = load_gru()
    if use_mlp:
        print(f"  ✅ MLP loaded ({len(ID_TO_LABEL)} classes)")
        # Ensure every model label has a meme image
        for label in ID_TO_LABEL.values():
            memes = ensure_meme_for_label(label, memes)
        print(f"  Total memes available: {len(memes)}")
    if use_gru:
        print(f"  ✅ GRU motion model loaded")
    if not use_mlp and not use_gru:
        print("  ℹ️  No trained models — using rule-based classifier")
        print("     Run: python record.py  →  python train.py")

    print("Loading YOLOv8n...")
    yolo = YOLO("yolov8n.pt")
    print("  YOLO ready.")

    # MediaPipe HandLandmarker
    print("Initialising MediaPipe HandLandmarker...")
    model_path = "hand_landmarker.task"
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
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
        result_callback=lambda r, o, t: _callback(r, o, t),
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    print("  MediaPipe ready.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        sys.exit(1)

    cv2.namedWindow("Mememic", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Mememic", 1280, 720)

    current_gesture = None
    gesture_stable_frames = 0
    STABLE_THRESHOLD = 5
    meme_overlay = None
    meme_alpha = 0.7
    frame_count = 0
    motion_cooldown = 0  # frames to wait after motion trigger

    print("\n🎥 Running — show a hand gesture to the camera!\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        frame_count += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        landmarker.detect_async(mp_image, frame_count)

        lm_list = get_landmarks()
        gesture = "FIST"
        motion_conf = None

        if lm_list is not None:
            # Feed landmarks into motion buffer
            norm = normalize_landmarks(lm_list[0])
            MOTION_BUFFER.append(norm)

            # Classify: motion model takes priority if confident
            if use_gru and motion_cooldown <= 0:
                motion_result = classify_motion()
                if motion_result is not None:
                    gesture, motion_conf = motion_result
                    motion_cooldown = 15  # debounce
                else:
                    # Fall back to MLP or rule
                    if use_mlp:
                        gesture = classify_mlp(lm_list[0])
                    else:
                        gesture = classify_gesture(lm_list[0])
            else:
                if use_mlp:
                    gesture = classify_mlp(lm_list[0])
                else:
                    gesture = classify_gesture(lm_list[0])

            if motion_cooldown > 0:
                motion_cooldown -= 1

            # Draw landmarks
            for hand_lm in lm_list:
                for lx, ly, _ in hand_lm:
                    cx, cy = int(lx * w), int(ly * h)
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                connections = [
                    (0,1),(1,2),(2,3),(3,4),
                    (0,5),(5,6),(6,7),(7,8),
                    (0,9),(9,10),(10,11),(11,12),
                    (0,13),(13,14),(14,15),(15,16),
                    (0,17),(17,18),(18,19),(19,20),
                    (5,9),(9,13),(13,17),
                ]
                for a, b in connections:
                    if a < len(hand_lm) and b < len(hand_lm):
                        p1 = (int(hand_lm[a][0] * w), int(hand_lm[a][1] * h))
                        p2 = (int(hand_lm[b][0] * w), int(hand_lm[b][1] * h))
                        cv2.line(frame, p1, p2, (0, 255, 0), 2)

            cx = int(lm_list[0][0][0] * w)
            cy = int(lm_list[0][0][1] * h)
            label = f"{gesture}"
            if motion_conf:
                label += f" ({motion_conf:.2f})"
            cv2.putText(frame, label, (cx - 30, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Gesture stability
        if gesture == current_gesture:
            gesture_stable_frames += 1
        else:
            current_gesture = gesture
            gesture_stable_frames = 0
            meme_overlay = None

        if gesture_stable_frames >= STABLE_THRESHOLD and current_gesture in memes:
            meme_overlay = memes[current_gesture]

        # Overlay meme
        if meme_overlay is not None:
            meme_resized = cv2.resize(meme_overlay, (400, 300))
            x_offset = w - 420
            y_offset = 20
            roi = frame[y_offset:y_offset+300, x_offset:x_offset+400]
            blended = cv2.addWeighted(roi, 1 - meme_alpha, meme_resized, meme_alpha, 0)
            frame[y_offset:y_offset+300, x_offset:x_offset+400] = blended

        # Info bar
        mode = "GRU" if use_gru else ("MLP" if use_mlp else "RULE")
        cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, f"[{mode}] Gesture: {current_gesture or '...'}  |  Q to quit",
                    (20, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("Mememic", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("\n👋 Mememic says goodbye.")


def _callback(result, output_image, timestamp_ms):
    """MediaPipe LIVE_STREAM callback."""
    lm_list = []
    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            pts = [(lm.x, lm.y, lm.z) for lm in hand]
            lm_list.append(pts)
    update_landmarks(lm_list if lm_list else None)


if __name__ == "__main__":
    main()
