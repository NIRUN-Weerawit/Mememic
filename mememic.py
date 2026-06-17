"""
Mememic — Hand gesture + face expression → meme display.

Runs MediaPipe HandLandmarker + FaceLandmarker simultaneously.
Uses trained models: hand MLP, face MLP, combined fusion, motion GRU.
Falls back to rule-based if no models available.
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEME_DIR = os.path.join(BASE_DIR, "memes")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MEME_DIR, exist_ok=True)

# ── Meme loading ─────────────────────────────────────────────────────────
MEME_TEMPLATES = {}


def load_memes():
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


def generate_meme_image(text, color, size=(400, 300)):
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


def ensure_meme_for_label(label, memes):
    if label not in memes:
        img = generate_meme_image(label.replace("_", " ").title(), (100, 100, 200))
        path = os.path.join(MEME_DIR, f"{label}.png")
        img.save(path)
        memes[label] = cv2.imread(path)
    return memes


# ── Feature extraction ────────────────────────────────────────────────────
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


# ── Model loading ─────────────────────────────────────────────────────────
ID_TO_LABEL = None
HAND_MODEL = None
FACE_MODEL = None
COMBINED_MODEL = None
GRU_MODEL = None
MOTION_BUFFER = deque(maxlen=60)


def load_models():
    global ID_TO_LABEL, HAND_MODEL, FACE_MODEL, COMBINED_MODEL, GRU_MODEL

    label_path = os.path.join(MODEL_DIR, "labels.json")
    if not os.path.exists(label_path):
        return False

    with open(label_path) as f:
        ID_TO_LABEL = json.load(f)
        ID_TO_LABEL = {int(k): v for k, v in ID_TO_LABEL.items()}

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

    return True


def classify_hand(landmarks):
    if HAND_MODEL is None or ID_TO_LABEL is None:
        return None, 0.0
    import torch
    norm = normalize_hand(landmarks)
    x = torch.tensor([norm], dtype=torch.float32)
    with torch.no_grad():
        out = HAND_MODEL(x)
        probs = torch.softmax(out, dim=1)
        conf, pred = probs.max(dim=1)
    return ID_TO_LABEL.get(pred.item(), "unknown"), conf.item()


def classify_face(blendshapes):
    if FACE_MODEL is None or ID_TO_LABEL is None:
        return None, 0.0
    import torch
    x = torch.tensor([blendshapes], dtype=torch.float32)
    with torch.no_grad():
        out = FACE_MODEL(x)
        probs = torch.softmax(out, dim=1)
        conf, pred = probs.max(dim=1)
    return ID_TO_LABEL.get(pred.item(), "unknown"), conf.item()


def classify_combined(hand_lm, face_bs):
    if COMBINED_MODEL is None or ID_TO_LABEL is None:
        return None, 0.0
    import torch
    h = torch.tensor([normalize_hand(hand_lm)], dtype=torch.float32)
    f = torch.tensor([face_bs], dtype=torch.float32)
    with torch.no_grad():
        out = COMBINED_MODEL(h, f)
        probs = torch.softmax(out, dim=1)
        conf, pred = probs.max(dim=1)
    return ID_TO_LABEL.get(pred.item(), "unknown"), conf.item()


def classify_motion():
    if GRU_MODEL is None or ID_TO_LABEL is None or len(MOTION_BUFFER) < 5:
        return None, 0.0
    import torch
    seq = list(MOTION_BUFFER)
    x = torch.tensor([seq], dtype=torch.float32)
    lengths = [len(seq)]
    with torch.no_grad():
        out = GRU_MODEL(x, lengths)
        probs = torch.softmax(out, dim=1)
        conf, pred = probs.max(dim=1)
    if conf.item() < 0.7:
        return None, 0.0
    return ID_TO_LABEL.get(pred.item(), "unknown"), conf.item()


# ── Rule-based fallback ──────────────────────────────────────────────────
def classify_gesture(landmarks):
    lm = landmarks
    def is_extended(tip, pip): return lm[tip][1] < lm[pip][1]
    thumb_up = lm[4][1] < lm[3][1]
    index_up = is_extended(8, 6)
    middle_up = is_extended(12, 10)
    ring_up = is_extended(16, 14)
    pinky_up = is_extended(20, 18)
    fingers = [index_up, middle_up, ring_up, pinky_up]
    count = sum(fingers)
    if thumb_up and count == 0: return "THUMBS_UP"
    if not thumb_up and count == 0:
        if (lm[4][1] - lm[2][1]) > 0.10: return "THUMBS_DOWN"
    if not index_up and middle_up and not ring_up and not pinky_up and not thumb_up: return "MIDDLE_FINGER"
    if index_up and not middle_up and not ring_up and pinky_up and not thumb_up: return "ROCK_ON"
    if thumb_up and index_up and not middle_up and not ring_up and pinky_up: return "SPIDERMAN"
    if index_up and not middle_up and not ring_up and not pinky_up and not thumb_up: return "POINTING"
    dist_ti = np.linalg.norm(np.array(lm[4][:2]) - np.array(lm[8][:2]))
    if dist_ti < 0.08 and index_up and not middle_up and not ring_up and not pinky_up: return "OK"
    if count == 0: return "FIST"
    elif count == 1: return "ONE"
    elif count == 2: return "PEACE"
    elif count == 3: return "THREE"
    elif count == 4: return "FOUR"
    elif count == 5: return "FIVE"
    return "FIST"


# ── Thread-safe state ──────────────────────────────────────────────────
current_hands = None  # list of hands, each is list of 21 (x,y,z) tuples
current_face = None
state_lock = threading.Lock()


def update_state(hands, face_bs):
    global current_hands, current_face
    with state_lock:
        current_hands = hands
        current_face = face_bs


def get_state():
    global current_hands, current_face
    with state_lock:
        return current_hands, current_face


def hand_callback(result, output_image, timestamp_ms):
    hands = []
    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            pts = [(lm.x, lm.y, lm.z) for lm in hand]
            hands.append(pts)
    _, f = get_state()
    update_state(hands if hands else None, f)


def face_callback(result, output_image, timestamp_ms):
    bs = result.face_blendshapes if result.face_blendshapes else None
    h, _ = get_state()
    update_state(h, bs)


# ── Main loop ────────────────────────────────────────────────────────────
def main():
    print("Mememic — Hand + Face → meme")
    print("Press Q to quit.")

    memes = load_memes()
    print(f"  Loaded {len(memes)} meme images from memes/")

    models_loaded = load_models()
    if models_loaded:
        print(f"  ✅ Models loaded ({len(ID_TO_LABEL)} classes)")
        for label in ID_TO_LABEL.values():
            memes = ensure_meme_for_label(label, memes)
        print(f"  Total memes: {len(memes)}")
    else:
        print("  ℹ️  No trained models — using rule-based classifier")

    # MediaPipe models
    hand_model = os.path.join(BASE_DIR, "hand_landmarker.task")
    face_model = os.path.join(BASE_DIR, "face_landmarker.task")
    for path, name in [(hand_model, "hand_landmarker"), (face_model, "face_landmarker")]:
        if not os.path.exists(path):
            print(f"  Downloading {name}...")
            import urllib.request
            url = f"https://storage.googleapis.com/mediapipe-models/{name}/{name}/float16/latest/{name}.task"
            urllib.request.urlretrieve(url, path)

    hand_base = python.BaseOptions(model_asset_path=hand_model)
    hand_opts = vision.HandLandmarkerOptions(
        base_options=hand_base, running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=2, min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6, min_tracking_confidence=0.5,
        result_callback=hand_callback,
    )
    hand_landmarker = vision.HandLandmarker.create_from_options(hand_opts)

    face_base = python.BaseOptions(model_asset_path=face_model)
    face_opts = vision.FaceLandmarkerOptions(
        base_options=face_base, running_mode=vision.RunningMode.LIVE_STREAM,
        num_faces=1, min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5, min_tracking_confidence=0.5,
        output_face_blendshapes=True,
        result_callback=face_callback,
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(face_opts)

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
    motion_cooldown = 0

    print("\n🎥 Running — show hand + face to the camera!\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        frame_count += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        hand_landmarker.detect_async(mp_image, frame_count)
        face_landmarker.detect_async(mp_image, frame_count)

        hands, face_bs = get_state()
        gesture = "FIST"
        conf = 0.0
        mode_label = "RULE"

        if hands is not None and len(hands) > 0:
            # Use the first detected hand for inference
            primary_hand = hands[0]

            # Feed primary hand into motion buffer
            norm = normalize_hand(primary_hand)
            MOTION_BUFFER.append(norm)

            # Try motion GRU first
            if GRU_MODEL and motion_cooldown <= 0:
                g, c = classify_motion()
                if g is not None:
                    gesture, conf = g, c
                    mode_label = "GRU"
                    motion_cooldown = 15

            # Try combined (hand + face)
            if conf < 0.5 and COMBINED_MODEL and face_bs is not None:
                g, c = classify_combined(primary_hand, extract_face(face_bs))
                if g is not None and c > conf:
                    gesture, conf = g, c
                    mode_label = "COMBINED"

            # Try hand MLP
            if conf < 0.5 and HAND_MODEL:
                g, c = classify_hand(primary_hand)
                if g is not None and c > conf:
                    gesture, conf = g, c
                    mode_label = "HAND"

            # Try face MLP
            if conf < 0.5 and FACE_MODEL and face_bs is not None:
                g, c = classify_face(extract_face(face_bs))
                if g is not None and c > conf:
                    gesture, conf = g, c
                    mode_label = "FACE"

            # Fallback to rule
            if conf < 0.5:
                gesture = classify_gesture(primary_hand)
                mode_label = "RULE"

            if motion_cooldown > 0:
                motion_cooldown -= 1

            # Draw all hand landmarks
            for hand in hands:
                for lx, ly, _ in hand:
                    cx, cy = int(lx * w), int(ly * h)
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                connections = [
                    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
                    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
                ]
                for a, b in connections:
                    if a < len(hand) and b < len(hand):
                        p1 = (int(hand[a][0] * w), int(hand[a][1] * h))
                        p2 = (int(hand[b][0] * w), int(hand[b][1] * h))
                        cv2.line(frame, p1, p2, (0, 255, 0), 2)

            # Draw face expression
            if face_bs is not None:
                bs = extract_face(face_bs)
                if bs:
                    smile = bs[44]  # mouthSmileLeft
                    brow = bs[1]    # browDownLeft
                    jaw = bs[25]    # jawOpen
                    cv2.putText(frame, f"smile:{smile:.2f} brow:{brow:.2f} jaw:{jaw:.2f}",
                                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)

            # Label at first hand's wrist
            cx = int(primary_hand[0][0] * w)
            cy = int(primary_hand[0][1] * h)
            label = f"{gesture} ({conf:.2f}) [{mode_label}]"
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

        if meme_overlay is not None:
            meme_resized = cv2.resize(meme_overlay, (400, 300))
            x_offset = w - 420
            y_offset = 20
            roi = frame[y_offset:y_offset+300, x_offset:x_offset+400]
            blended = cv2.addWeighted(roi, 1 - meme_alpha, meme_resized, meme_alpha, 0)
            frame[y_offset:y_offset+300, x_offset:x_offset+400] = blended

        cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, f"[{mode_label}] Gesture: {current_gesture or '...'}  |  Q to quit",
                    (20, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("Mememic", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hand_landmarker.close()
    face_landmarker.close()
    print("\n👋 Mememic says goodbye.")


if __name__ == "__main__":
    main()
