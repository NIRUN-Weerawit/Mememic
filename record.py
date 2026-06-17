"""
Mememic — Gesture + Face Expression recording pipeline.

Records hand landmarks AND face blend shapes per frame.
Static mode: SPACE records 1 frame. Motion mode: SPACE to start/stop.

Controls:
  SPACE  — record one frame (static) or start/stop (motion)
  N      — advance to next gesture
  M      — toggle static/motion mode
  R      — re-record (delete all samples for current gesture)
  Q      — quit
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import sys
import json
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "recorded_data")
os.makedirs(DATA_DIR, exist_ok=True)

MEME_TEMPLATES = {
    "67_cat":           ("6..7",          (255, 50, 50)),
    "actually":         ("umm...actually", (200, 200, 50)),
    "bite_finger":      ("bite_finger", (100, 200, 255)),
    "burn_to_ash":      ("burn_to_ash", (255, 150, 50)),
    "cat_laugh":        ("cat_laugh ",   (255, 100, 200)),
    "good":             ("good ", (50, 255, 50)),
    "hmm":              ("hmm ",            (50, 200, 255)),
    "monkey_confused":  ("monkey_confused",       (150, 100, 255)),
    "no_thanks":        ("no_thanks ",   (255, 50, 50)),
    "slap_sandal":      ("slap_sandal",          (200, 50, 200)),
    "thinking":         ("thinking.",   (50, 200, 50)),
    "throw_rose":       ("throw_rose",  (200, 50, 50)),
}

# 53 MediaPipe face blend shape names (in order)
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


def normalize_landmarks(landmarks):
    """Normalize 21 hand landmarks to 63-dim translation/scale-invariant vector."""
    pts = np.array(landmarks, dtype=np.float32)
    wrist = pts[0]
    centered = pts - wrist
    scale = np.max(np.linalg.norm(centered, axis=1))
    if scale > 0:
        centered /= scale
    return centered.flatten().tolist()


def extract_blend_shapes(face_blendshapes):
    """Extract 53 blend shape scores into a flat list. Returns None if no face."""
    if not face_blendshapes or not face_blendshapes[0]:
        return None
    return [bs.score for bs in face_blendshapes[0]]


def make_sample(hand_lm, face_bs):
    """Build a sample dict with hand + face features."""
    sample = {}
    if hand_lm is not None:
        sample["hand"] = normalize_landmarks(hand_lm)
    else:
        sample["hand"] = None
    sample["face"] = extract_blend_shapes(face_bs)
    return sample


def load_existing_data():
    path = os.path.join(DATA_DIR, "gestures.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"_meta": {}}


def save_data(data):
    path = os.path.join(DATA_DIR, "gestures.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    total_static = 0
    total_motion = 0
    meta = data.get("_meta", {})
    for name, samples in data.items():
        if name == "_meta":
            continue
        if meta.get(name) == "motion":
            total_motion += len(samples)
        else:
            total_static += len(samples)
    print(f"  💾 Saved ({total_static} static samples, {total_motion} motion sequences)")


# ── Thread-safe state ──────────────────────────────────────────────────
current_hand = None
current_face = None
state_lock = threading.Lock()


def update_state(hand_lm, face_bs):
    global current_hand, current_face
    with state_lock:
        current_hand = hand_lm
        current_face = face_bs


def get_state():
    global current_hand, current_face
    with state_lock:
        return current_hand, current_face


def hand_callback(result, output_image, timestamp_ms):
    lm_list = []
    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            pts = [(lm.x, lm.y, lm.z) for lm in hand]
            lm_list.append(pts)
    h = lm_list[0] if lm_list else None
    _, f = get_state()
    update_state(h, f)


def face_callback(result, output_image, timestamp_ms):
    bs = result.face_blendshapes if result.face_blendshapes else None
    h, _ = get_state()
    update_state(h, bs)


def main():
    print("=" * 60)
    print("  Mememic — Gesture + Face Recording Pipeline")
    print("=" * 60)
    print()
    print("  Records hand landmarks + face expression per frame.")
    print("  Static mode: SPACE records 1 frame. N to advance.")
    print("  Motion mode: M to toggle, SPACE to start/stop.")
    print()

    data = load_existing_data()
    if "_meta" not in data:
        data["_meta"] = {}
    meta = data["_meta"]

    if data:
        total_static = 0
        total_motion = 0
        for name, samples in data.items():
            if name == "_meta":
                continue
            if meta.get(name) == "motion":
                total_motion += len(samples)
            else:
                total_static += len(samples)
        print(f"  Loaded existing data: {total_static} static samples, {total_motion} motion sequences")
        for name in sorted([k for k in data if k != "_meta"]):
            n = len(data[name])
            t = "motion" if meta.get(name) == "motion" else "static"
            print(f"    {name:15s}: {n} {t} samples")
    print()

    # MediaPipe models
    hand_model = os.path.join(BASE_DIR, "hand_landmarker.task")
    face_model = os.path.join(BASE_DIR, "face_landmarker.task")
    for path, name in [(hand_model, "hand_landmarker"), (face_model, "face_landmarker")]:
        if not os.path.exists(path):
            print(f"  Downloading {name} model...")
            import urllib.request
            url = f"https://storage.googleapis.com/mediapipe-models/{name}/{name}/float16/latest/{name}.task"
            urllib.request.urlretrieve(url, path)
            print("  Downloaded.")

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

    cv2.namedWindow("Mememic Record", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Mememic Record", 1280, 720)

    gesture_names = list(MEME_TEMPLATES.keys())
    current_idx = 0
    frame_count = 0
    recording = False
    record_buffer = []
    motion_mode = False
    status_text = ""
    status_color = (200, 200, 200)
    status_timer = 0

    while current_idx < len(gesture_names):
        name = gesture_names[current_idx]
        text, color = MEME_TEMPLATES[name]
        existing = len(data.get(name, []))
        is_motion = meta.get(name) == "motion"

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

        hand_lm, face_bs = get_state()

        # Draw hand landmarks
        if hand_lm:
            for lm in hand_lm:
                for lx, ly, _ in lm:
                    cx, cy = int(lx * w), int(ly * h)
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                connections = [
                    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
                    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
                ]
                for a, b in connections:
                    if a < len(lm) and b < len(lm):
                        p1 = (int(lm[a][0] * w), int(lm[a][1] * h))
                        p2 = (int(lm[b][0] * w), int(lm[b][1] * h))
                        cv2.line(frame, p1, p2, (0, 255, 0), 2)

        # Draw face mesh
        if face_bs is not None:
            # Show expression labels
            smile = face_bs[44].score  # mouthSmileLeft
            brow = face_bs[1].score    # browDownLeft
            jaw = face_bs[25].score    # jawOpen
            cv2.putText(frame, f"smile:{smile:.2f} brow:{brow:.2f} jaw:{jaw:.2f}",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)

        # ── Recording logic ────────────────────────────────────────────
        if recording and motion_mode:
            sample = make_sample(hand_lm[0] if hand_lm else None, face_bs)
            record_buffer.append(sample)

        # ── Draw UI ──────────────────────────────────────────────────
        preview = np.zeros((300, 400, 3), dtype=np.uint8)
        preview[:] = (20, 20, 30)
        cv2.rectangle(preview, (0, 0), (400, 8), color, -1)
        lines = text.split("\n")
        for i, line in enumerate(lines):
            cv2.putText(preview, line, (20, 60 + i * 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(preview, "Mememic", (10, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
        frame[20:320, w-420:w-20] = preview

        bar_w = w - 40
        progress = current_idx / len(gesture_names)
        cv2.rectangle(frame, (20, h - 80), (20 + int(bar_w * progress), h - 70),
                      (0, 255, 0), -1)
        cv2.rectangle(frame, (20, h - 80), (20 + bar_w, h - 70),
                      (100, 100, 100), 1)

        cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 0), -1)
        if recording and motion_mode:
            cv2.putText(frame, f"🎥 Recording motion... ({len(record_buffer)} frames)",
                        (20, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        elif status_timer > 0:
            cv2.putText(frame, status_text, (20, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            status_timer -= 1
        else:
            mode_label = "MOTION" if motion_mode else "STATIC"
            cv2.putText(frame, f"[{mode_label}] SPACE record  |  N next  |  M toggle  |  R clear  |  Q quit",
                        (20, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        mode_label = "MOTION" if is_motion else "STATIC"
        cv2.putText(frame, f"Gesture: {name}  [{mode_label}]  [{existing} samples]  ({current_idx+1}/{len(gesture_names)})",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Mememic Record", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("m") and not recording:
            motion_mode = not motion_mode
            status_text = f"🔄 Switched to {'MOTION' if motion_mode else 'STATIC'} mode"
            status_color = (255, 200, 50)
            status_timer = 30
            print(f"  🔄 Switched to {'MOTION' if motion_mode else 'STATIC'} mode")
        elif key == ord(" ") and not recording and not motion_mode:
            # STATIC: record 1 frame
            hand_lm, face_bs = get_state()
            sample = make_sample(hand_lm[0] if hand_lm else None, face_bs)
            if name not in data or data[name] is None:
                data[name] = []
            data[name].append(sample)
            meta[name] = "static"
            save_data(data)
            status_text = f"✅ Recorded 1 frame for {name} (total: {len(data[name])})"
            status_color = (0, 255, 0)
            status_timer = 30
            print(f"  ✅ {name}: +1 frame (total: {len(data[name])})")
        elif key == ord(" ") and not recording and motion_mode:
            recording = True
            record_buffer = []
            status_text = f"🎥 Recording motion {name}... press SPACE to stop"
            status_color = (0, 255, 255)
            status_timer = 30
            print(f"  🎥 Recording motion {name}... press SPACE to stop")
        elif key == ord(" ") and recording and motion_mode:
            recording = False
            if len(record_buffer) >= 5:
                if name not in data or data[name] is None:
                    data[name] = []
                data[name].append(record_buffer)
                meta[name] = "motion"
                save_data(data)
                status_text = f"✅ Recorded motion sequence ({len(record_buffer)} frames)"
                status_color = (0, 255, 0)
                status_timer = 30
                print(f"  ✅ {name}: motion sequence ({len(record_buffer)} frames, total: {len(data[name])})")
            else:
                status_text = f"⚠️ Too few frames ({len(record_buffer)}). Need ≥5."
                status_color = (0, 0, 255)
                status_timer = 30
            record_buffer = []
        elif key == ord("n") and not recording:
            current_idx += 1
            status_text = "⏭️  Next gesture"
            status_color = (200, 200, 200)
            status_timer = 20
        elif key == ord("r") and not recording:
            if name in data:
                del data[name]
                save_data(data)
                print(f"  🔄 Cleared all samples for {name}")
            status_text = f"🔄 Cleared {name}"
            status_color = (255, 200, 50)
            status_timer = 30

    cap.release()
    cv2.destroyAllWindows()
    hand_landmarker.close()
    face_landmarker.close()

    print()
    print("=" * 60)
    print("  Recording complete!")
    print("=" * 60)
    for name in sorted([k for k in data if k != "_meta"]):
        n = len(data[name])
        t = "motion" if meta.get(name) == "motion" else "static"
        print(f"    {name:15s}: {n} {t} samples")
    print()
    print(f"  Data saved to: {DATA_DIR}/gestures.json")
    print("  Next step: python train.py")


if __name__ == "__main__":
    main()
