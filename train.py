"""
Mememic — Train gesture + face expression classifiers.

Supports:
  - Hand MLP: 63 hand landmarks → class
  - Face MLP: 53 face blend shapes → class
  - Combined: fuse hand + face features → class
  - Motion GRU: variable-length sequences → class

Data format (gestures.json):
  Each sample is a dict: {"hand": [63] or null, "face": [53] or null}
  Motion: list of such dicts per sequence
"""

import json
import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "recorded_data", "gestures.json")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

GESTURE_NAMES = []
label_to_id = {}
id_to_label = {}


def load_data():
    """Load recorded data. Returns static and motion datasets."""
    global GESTURE_NAMES, label_to_id, id_to_label

    if not os.path.exists(DATA_PATH):
        print(f"❌ No recorded data found at {DATA_PATH}")
        print("   Run python record.py first!")
        sys.exit(1)

    with open(DATA_PATH) as f:
        raw = json.load(f)

    meta = raw.get("_meta", {})

    discovered = sorted([k for k in raw if k != "_meta"])
    GESTURE_NAMES = discovered
    label_to_id = {name: i for i, name in enumerate(GESTURE_NAMES)}
    id_to_label = {i: name for i, name in enumerate(GESTURE_NAMES)}

    # Static: list of sample dicts
    # Static data: flat list of 63-dim samples
    X_hand, X_face, y_static, y_face = [], [], [], []
    # Aligned hand+face pairs for combined training
    X_hand_aligned, X_face_aligned, y_aligned = [], [], []
    static_counts = {}

    # Motion data: list of sequences, each sequence is list of sample dicts
    X_motion_hand, X_motion_face, y_motion = [], [], []
    motion_counts = {}

    for name, samples in raw.items():
        if name == "_meta":
            continue
        if name not in label_to_id:
            print(f"  ⚠️  Unknown gesture '{name}' — skipping")
            continue

        label = label_to_id[name]
        is_motion = meta.get(name) == "motion"

        if is_motion:
            for seq in samples:
                if len(seq) >= 5:
                    hands = [s.get("hand") for s in seq]
                    faces = [s.get("face") for s in seq]
                    X_motion_hand.append(hands)
                    X_motion_face.append(faces)
                    y_motion.append(label)
            motion_counts[name] = len(samples)
        else:
            for s in samples:
                hand = s.get("hand")
                face = s.get("face")
                if hand is not None:
                    X_hand.append(hand)
                    y_static.append(label)
                if face is not None:
                    # Handle old data with 52 blend shapes (missing _neutral)
                    if len(face) == 52:
                        face = [0.0] + face  # prepend _neutral=0
                    if len(face) == 53:
                        X_face.append(face)
                        y_face.append(label)
                # Aligned pair for combined training
                if hand is not None and face is not None and len(face) == 53:
                    X_hand_aligned.append(hand)
                    X_face_aligned.append(face)
                    y_aligned.append(label)
            static_counts[name] = len(samples)

    X_hand = np.array(X_hand, dtype=np.float32) if X_hand else np.empty((0, 63), dtype=np.float32)
    X_face = np.array(X_face, dtype=np.float32) if X_face else np.empty((0, 53), dtype=np.float32)
    y_static = np.array(y_static, dtype=np.int64) if y_static else np.empty((0,), dtype=np.int64)
    y_face = np.array(y_face, dtype=np.int64) if y_face else np.empty((0,), dtype=np.int64)
    X_hand_aligned = np.array(X_hand_aligned, dtype=np.float32) if X_hand_aligned else np.empty((0, 63), dtype=np.float32)
    X_face_aligned = np.array(X_face_aligned, dtype=np.float32) if X_face_aligned else np.empty((0, 53), dtype=np.float32)
    y_aligned = np.array(y_aligned, dtype=np.int64) if y_aligned else np.empty((0,), dtype=np.int64)

    print(f"\n  Static hand samples: {len(X_hand)}")
    print(f"  Static face samples: {len(X_face)}")
    print(f"  Aligned hand+face pairs: {len(X_hand_aligned)}")
    print(f"  Motion sequences: {len(X_motion_hand)}")
    for name, count in sorted(static_counts.items()):
        print(f"    {name}: {count}")
    print()

    return X_hand, X_face, y_static, y_face, X_hand_aligned, X_face_aligned, y_aligned, X_motion_hand, X_motion_face, y_motion, static_counts, motion_counts


def train_mlp(X, y, name="MLP", input_dim=None):
    """Train a simple MLP classifier."""
    import torch
    import torch.nn as nn
    import torch.optim as optim

    n_classes = len(GESTURE_NAMES)
    if input_dim is None:
        input_dim = X.shape[1]
    n_samples = len(X)

    if n_samples < 5:
        print(f"  ⏭️  Skipping {name}: only {n_samples} samples")
        return None

    perm = np.random.permutation(n_samples)
    split = int(n_samples * 0.8)
    train_idx = perm[:split]
    val_idx = perm[split:]

    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y[train_idx], dtype=torch.long)
    X_val = torch.tensor(X[val_idx], dtype=torch.float32)
    y_val = torch.tensor(y[val_idx], dtype=torch.long)

    if len(val_idx) < 1:
        X_val, y_val = X_train.clone(), y_train.clone()

    model = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, n_classes),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_acc = 0.0
    best_state = None
    patience = 20
    no_improve = 0

    print(f"  Training {name}: {input_dim} → 128 → 64 → {n_classes}")
    print(f"  Samples: {len(X_train)} train, {len(X_val)} val")

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_loss = criterion(val_out, y_val)
            preds = val_out.argmax(dim=1)
            acc = (preds == y_val).float().mean().item()

        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d} | loss: {loss.item():.4f} | val_loss: {val_loss.item():.4f} | val_acc: {acc:.3f}")

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        all_out = model(torch.tensor(X, dtype=torch.float32))
        all_preds = all_out.argmax(dim=1).numpy()
        train_acc = (all_preds == y).mean()

    print(f"\n  ✅ {name} best val accuracy: {best_acc:.3f}")
    print(f"  ✅ {name} train accuracy:   {train_acc:.3f}")

    return model


def train_combined(X_hand, X_face, y):
    """Train a combined model that fuses hand + face features."""
    import torch
    import torch.nn as nn
    import torch.optim as optim

    n_classes = len(GESTURE_NAMES)
    n_samples = min(len(X_hand), len(X_face))

    if n_samples < 5:
        print("  ⏭️  Skipping Combined: too few samples with both hand+face")
        return None

    # Use only samples that have both hand and face
    # (they should be aligned since we record them together)
    X_h = X_hand[:n_samples]
    X_f = X_face[:n_samples]
    y_c = y[:n_samples]

    perm = np.random.permutation(n_samples)
    split = int(n_samples * 0.8)
    train_idx = perm[:split]
    val_idx = perm[split:]

    Xh_train = torch.tensor(X_h[train_idx], dtype=torch.float32)
    Xf_train = torch.tensor(X_f[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y_c[train_idx], dtype=torch.long)
    Xh_val = torch.tensor(X_h[val_idx], dtype=torch.float32)
    Xf_val = torch.tensor(X_f[val_idx], dtype=torch.float32)
    y_val = torch.tensor(y_c[val_idx], dtype=torch.long)

    if len(val_idx) < 1:
        Xh_val, Xf_val, y_val = Xh_train.clone(), Xf_train.clone(), y_train.clone()

    class FusionMLP(nn.Module):
        def __init__(self, hand_dim, face_dim, num_classes):
            super().__init__()
            self.hand_net = nn.Sequential(
                nn.Linear(hand_dim, 64), nn.ReLU(), nn.Dropout(0.2))
            self.face_net = nn.Sequential(
                nn.Linear(face_dim, 64), nn.ReLU(), nn.Dropout(0.2))
            self.fusion = nn.Sequential(
                nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, num_classes))

        def forward(self, hand, face):
            h = self.hand_net(hand)
            f = self.face_net(face)
            x = torch.cat([h, f], dim=1)
            return self.fusion(x)

    model = FusionMLP(63, 53, n_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_acc = 0.0
    best_state = None
    patience = 20
    no_improve = 0

    print(f"  Training Combined: hand(63→64) + face(53→64) → fusion(128→64→{n_classes})")
    print(f"  Samples: {len(Xh_train)} train, {len(Xh_val)} val")

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        outputs = model(Xh_train, Xf_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(Xh_val, Xf_val)
            val_loss = criterion(val_out, y_val)
            preds = val_out.argmax(dim=1)
            acc = (preds == y_val).float().mean().item()

        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d} | loss: {loss.item():.4f} | val_loss: {val_loss.item():.4f} | val_acc: {acc:.3f}")

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        all_out = model(torch.tensor(X_h, dtype=torch.float32),
                        torch.tensor(X_f, dtype=torch.float32))
        all_preds = all_out.argmax(dim=1).numpy()
        train_acc = (all_preds == y_c).mean()

    print(f"\n  ✅ Combined best val accuracy: {best_acc:.3f}")
    print(f"  ✅ Combined train accuracy:   {train_acc:.3f}")

    return model


def train_gru(X_motion_hand, y_motion):
    """Train GRU for motion sequence classification (hand only)."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.nn.utils.rnn import pad_sequence

    n_classes = len(GESTURE_NAMES)
    n_sequences = len(X_motion_hand)

    if n_sequences < 2:
        print("  ⏭️  Skipping GRU: need ≥2 motion sequences")
        return None

    # Filter out None frames and convert to tensors
    tensors = []
    seq_lens = []
    for seq in X_motion_hand:
        valid = [f for f in seq if f is not None]
        if len(valid) >= 5:
            tensors.append(torch.tensor(np.array(valid, dtype=np.float32)))
            seq_lens.append(len(valid))

    if len(tensors) < 2:
        print("  ⏭️  Skipping GRU: too few valid sequences after filtering")
        return None

    padded = pad_sequence(tensors, batch_first=True)
    y_tensor = torch.tensor(y_motion[:len(tensors)], dtype=torch.long)

    perm = np.random.permutation(len(tensors))
    split = max(1, int(len(tensors) * 0.8))
    train_idx = perm[:split]
    val_idx = perm[split:]

    X_train = padded[train_idx]
    y_train = y_tensor[train_idx]
    X_val = padded[val_idx]
    y_val = y_tensor[val_idx]
    train_lens = [seq_lens[i] for i in train_idx]
    val_lens = [seq_lens[i] for i in val_idx]

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

    model = GRUClassifier(63, 64, n_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_acc = 0.0
    best_state = None
    patience = 30
    no_improve = 0

    print(f"\n  Training GRU: 63 → GRU(64, bidirectional) → {n_classes}")
    print(f"  Sequences: {len(X_train)} train, {len(X_val)} val")

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train, train_lens)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(X_val, val_lens)
            val_loss = criterion(val_out, y_val)
            preds = val_out.argmax(dim=1)
            acc = (preds == y_val).float().mean().item()

        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d} | loss: {loss.item():.4f} | val_loss: {val_loss.item():.4f} | val_acc: {acc:.3f}")

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    print(f"\n  ✅ GRU best val accuracy: {best_acc:.3f}")

    return model


def export_models(hand_model=None, face_model=None, combined_model=None, gru_model=None):
    """Export trained models."""
    import torch

    if hand_model is not None:
        path = os.path.join(MODEL_DIR, "hand_classifier.pt")
        torch.save(hand_model.state_dict(), path)
        print(f"  💾 Hand model saved to {path}")

    if face_model is not None:
        path = os.path.join(MODEL_DIR, "face_classifier.pt")
        torch.save(face_model.state_dict(), path)
        print(f"  💾 Face model saved to {path}")

    if combined_model is not None:
        path = os.path.join(MODEL_DIR, "combined_classifier.pt")
        torch.save(combined_model.state_dict(), path)
        print(f"  💾 Combined model saved to {path}")

    if gru_model is not None:
        path = os.path.join(MODEL_DIR, "motion_classifier.pt")
        torch.save(gru_model.state_dict(), path)
        print(f"  💾 GRU model saved to {path}")

    label_path = os.path.join(MODEL_DIR, "labels.json")
    with open(label_path, "w") as f:
        json.dump(id_to_label, f, indent=2)
    print(f"  💾 Labels saved to {label_path}")


def main():
    print("=" * 60)
    print("  Mememic — Gesture + Face Classifier Training")
    print("=" * 60)

    X_hand, X_face, y_static, y_face, X_hand_aligned, X_face_aligned, y_aligned, X_mh, X_mf, y_motion, sc, mc = load_data()

    hand_model = None
    face_model = None
    combined_model = None
    gru_model = None

    if len(X_hand) >= 10:
        hand_model = train_mlp(X_hand, y_static, name="Hand MLP", input_dim=63)
    else:
        print("  ⏭️  Skipping Hand MLP (need ≥10 samples)")

    if len(X_face) >= 10:
        face_model = train_mlp(X_face, y_face, name="Face MLP", input_dim=53)
    else:
        print("  ⏭️  Skipping Face MLP (need ≥10 samples)")

    if len(X_hand_aligned) >= 10:
        combined_model = train_combined(X_hand_aligned, X_face_aligned, y_aligned)
    else:
        print("  ⏭️  Skipping Combined (need ≥10 hand + face samples)")

    if len(X_mh) >= 2:
        gru_model = train_gru(X_mh, y_motion)
    else:
        print("  ⏭️  Skipping GRU (need ≥2 motion sequences)")

    if all(m is None for m in [hand_model, face_model, combined_model, gru_model]):
        print("\n❌ No models trained. Record more data with python record.py")
        sys.exit(1)

    export_models(hand_model, face_model, combined_model, gru_model)

    print()
    print("=" * 60)
    print("  Training complete!")
    print("  Run python mememic.py to use the trained model(s).")
    print("=" * 60)


if __name__ == "__main__":
    main()
