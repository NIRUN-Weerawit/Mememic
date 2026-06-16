"""
Mememic — Gesture recording pipeline.

Static mode (default): hold pose, press SPACE to record 1 frame.
  Press N to advance to next gesture. Collect as many samples as you want.

Motion mode: press M to toggle, SPACE to start/stop recording a sequence.
  Press N to advance to next gesture.

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


def normalize_landmarks(landmarks):
    """Normalize 21 landmarks to be translation- and scale-invariant. Returns 63-dim list."""
    pts = np.array(landmarks, dtype=np.float32)
    wrist = pts[0]
    centered = pts - wrist
    scale = np.max(np.linalg.norm(centered, axis=1))
    if scale > 0:
        centered /= scale
    return centered.flatten().tolist()


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


# ── Thread-safe landmark state ──────────────────────────────────────────
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


def _callback(result, output_image, timestamp_ms):
    lm_list = []
    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            pts = [(lm.x, lm.y, lm.z) for lm in hand]
            lm_list.append(pts)
    update_landmarks(lm_list if lm_list else None)


def main():
    print("=" * 60)
    print("  Mememic — Gesture Recording Pipeline")
    print("=" * 60)
    print()
    print("  Static mode (default):")
    print("    Hold pose, press SPACE to record 1 frame.")
    print("    Press N to advance to next gesture.")
    print("  Motion mode:")
    print("    Press M to toggle, SPACE to start/stop recording.")
    print("    Press N to advance to next gesture.")
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
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
        result_callback=_callback,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

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
        landmarker.detect_async(mp_image, frame_count)

        lm_list = get_landmarks()
        if lm_list:
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

        # ── Recording logic ────────────────────────────────────────────
        if recording and motion_mode:
            # Motion: accumulate frames while recording
            if lm_list:
                norm = normalize_landmarks(lm_list[0])
                record_buffer.append(norm)

        # ── Draw UI ──────────────────────────────────────────────────
        # Meme preview
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

        # Progress bar
        bar_w = w - 40
        progress = current_idx / len(gesture_names)
        cv2.rectangle(frame, (20, h - 80), (20 + int(bar_w * progress), h - 70),
                      (0, 255, 0), -1)
        cv2.rectangle(frame, (20, h - 80), (20 + bar_w, h - 70),
                      (100, 100, 100), 1)

        # Status bar
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
            cv2.putText(frame, f"[{mode_label}] SPACE to record  |  N next  |  M toggle  |  R clear  |  Q quit",
                        (20, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        # Gesture name + sample count
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
            if lm_list:
                norm = normalize_landmarks(lm_list[0])
                if name not in data or data[name] is None:
                    data[name] = []
                data[name].append(norm)
                meta[name] = "static"
                save_data(data)
                status_text = f"✅ Recorded 1 frame for {name} (total: {len(data[name])})"
                status_color = (0, 255, 0)
                status_timer = 30
                print(f"  ✅ {name}: +1 frame (total: {len(data[name])})")
            else:
                status_text = "⚠️ No hand detected!"
                status_color = (0, 0, 255)
                status_timer = 30

        elif key == ord(" ") and not recording and motion_mode:
            # MOTION: start recording
            recording = True
            record_buffer = []
            status_text = f"🎥 Recording motion {name}... press SPACE to stop"
            status_color = (0, 255, 255)
            status_timer = 30
            print(f"  🎥 Recording motion {name}... press SPACE to stop")

        elif key == ord(" ") and recording and motion_mode:
            # MOTION: stop recording
            recording = False
            if len(record_buffer) >= 5:
                if name not in data or data[name] is None:
                    data[name] = []
                data[name].append(record_buffer)
                meta[name] = "motion"
                save_data(data)
                status_text = f"✅ Recorded motion sequence ({len(record_buffer)} frames, total: {len(data[name])})"
                status_color = (0, 255, 0)
                status_timer = 30
                print(f"  ✅ {name}: motion sequence ({len(record_buffer)} frames, total: {len(data[name])} sequences)")
            else:
                status_text = f"⚠️ Too few frames ({len(record_buffer)}). Need ≥5. Try again."
                status_color = (0, 0, 255)
                status_timer = 30
                print(f"  ⚠️ {name}: only {len(record_buffer)} frames, need ≥5")
            record_buffer = []

        elif key == ord("n") and not recording:
            print(f"  ⏭️  Advanced to next gesture")
            current_idx += 1
            status_text = f"⏭️  Next gesture"
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
    landmarker.close()

    print()
    print("=" * 60)
    print("  Recording complete!")
    print("=" * 60)
    total_static = 0
    total_motion = 0
    for name in sorted([k for k in data if k != "_meta"]):
        n = len(data[name])
        t = "motion" if meta.get(name) == "motion" else "static"
        if t == "motion":
            total_motion += n
        else:
            total_static += n
        print(f"    {name:15s}: {n} {t} samples")
    print()
    print(f"  Data saved to: {DATA_DIR}/gestures.json")
    print()
    print("  Next step: python train.py")


if __name__ == "__main__":
    main()
