"""
Mememic — Train gesture classifier from recorded data.

Supports:
  - Static poses: individual 63-dim landmark frames → MLP
  - Motion gestures: variable-length sequences → GRU classifier

Usage:
  source venv/bin/activate
  python train.py
"""

import json
import os
import sys
import numpy as np
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "recorded_data", "gestures.json")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

GESTURE_NAMES = [
    "67_cat", "actually", "bite_finger",
    "burn_to_ash", "cat_laugh", "good", 
    "hmm", "monkey_confused", "no_thanks",
    "slap_sandal", "thinking", "throw_rose"
]

label_to_id = {name: i for i, name in enumerate(GESTURE_NAMES)}
id_to_label = {i: name for i, name in enumerate(GESTURE_NAMES)}


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

    # Auto-discover gesture names from data
    discovered = sorted([k for k in raw if k != "_meta"])
    GESTURE_NAMES = discovered
    label_to_id = {name: i for i, name in enumerate(GESTURE_NAMES)}
    id_to_label = {i: name for i, name in enumerate(GESTURE_NAMES)}

    # Static data: flat list of 63-dim samples
    X_static, y_static = [], []
    static_counts = {}

    # Motion data: list of sequences, each sequence is list of 63-dim frames
    X_motion, y_motion = [], []
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
            # samples is a list of sequences
            for seq in samples:
                if len(seq) >= 5:  # minimum sequence length
                    X_motion.append(seq)
                    y_motion.append(label)
            motion_counts[name] = len(samples)
        else:
            # samples is a flat list of 63-dim vectors
            for s in samples:
                X_static.append(s)
                y_static.append(label)
            static_counts[name] = len(samples)

    X_static = np.array(X_static, dtype=np.float32) if X_static else np.empty((0, 63), dtype=np.float32)
    y_static = np.array(y_static, dtype=np.int64) if y_static else np.empty((0,), dtype=np.int64)
    X_motion = [np.array(s, dtype=np.float32) for s in X_motion]
    y_motion = np.array(y_motion, dtype=np.int64) if y_motion else np.empty((0,), dtype=np.int64)

    print(f"\n  Static data:  {len(X_static)} samples")
    for name, count in sorted(static_counts.items()):
        print(f"    {name}: {count}")
    print(f"  Motion data: {len(X_motion)} sequences")
    for name, count in sorted(motion_counts.items()):
        print(f"    {name}: {count}")
    print()

    return X_static, y_static, X_motion, y_motion, static_counts, motion_counts


def train_mlp(X, y):
    """Train MLP for static pose classification."""
    import torch
    import torch.nn as nn
    import torch.optim as optim

    n_classes = len(GESTURE_NAMES)
    n_features = X.shape[1]
    n_samples = len(X)

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
        nn.Linear(n_features, 128),
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

    print(f"  Training MLP: {n_features} → 128 → 64 → {n_classes}")
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

    print(f"\n  ✅ MLP best val accuracy: {best_acc:.3f}")
    print(f"  ✅ MLP train accuracy:   {train_acc:.3f}")

    return model


def train_gru(X_motion, y_motion):
    """Train GRU for motion sequence classification."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

    n_classes = len(GESTURE_NAMES)
    n_features = 63
    n_sequences = len(X_motion)

    if n_sequences < 2:
        print("  ⚠️  Too few motion sequences for GRU training (need ≥2)")
        return None

    # Pad sequences to same length
    seq_lens = [len(s) for s in X_motion]
    tensors = [torch.tensor(s, dtype=torch.float32) for s in X_motion]
    padded = pad_sequence(tensors, batch_first=True)  # (B, T, 63)
    y_tensor = torch.tensor(y_motion, dtype=torch.long)

    # Train/val split
    perm = np.random.permutation(n_sequences)
    split = max(1, int(n_sequences * 0.8))
    train_idx = perm[:split]
    val_idx = perm[split:]

    X_train = padded[train_idx]
    y_train = y_tensor[train_idx]
    X_val = padded[val_idx]
    y_val = y_tensor[val_idx]
    train_lens = [seq_lens[i] for i in train_idx]
    val_lens = [seq_lens[i] for i in val_idx]

    # GRU model
    class GRUClassifier(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_classes):
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hidden_dim * 2, num_classes)

        def forward(self, x, lengths):
            packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            _, h_n = self.gru(packed)
            # h_n: (num_layers * num_directions, B, hidden_dim)
            h_fwd = h_n[-2, :, :]  # last forward
            h_bwd = h_n[-1, :, :]  # last backward
            h = torch.cat([h_fwd, h_bwd], dim=1)
            return self.fc(h)

    model = GRUClassifier(n_features, 64, n_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_acc = 0.0
    best_state = None
    patience = 30
    no_improve = 0

    print(f"\n  Training GRU: 63 → GRU(64, bidirectional) → {n_classes}")
    print(f"  Sequences: {len(X_train)} train, {len(X_val)} val")
    print(f"  Lengths: min={min(seq_lens)}, max={max(seq_lens)}, mean={np.mean(seq_lens):.0f}")

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


def export_models(mlp_model=None, gru_model=None):
    """Export trained models."""
    import torch

    if mlp_model is not None:
        path = os.path.join(MODEL_DIR, "gesture_classifier.pt")
        torch.save(mlp_model.state_dict(), path)
        print(f"  💾 MLP model saved to {path}")

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
    print("  Mememic — Gesture Classifier Training")
    print("=" * 60)

    X_static, y_static, X_motion, y_motion, static_counts, motion_counts = load_data()

    mlp_model = None
    gru_model = None

    if len(X_static) >= 10:
        mlp_model = train_mlp(X_static, y_static)
    else:
        print("  ⏭️  Skipping MLP (need ≥10 static samples)")

    if len(X_motion) >= 2:
        gru_model = train_gru(X_motion, y_motion)
    else:
        print("  ⏭️  Skipping GRU (need ≥2 motion sequences)")

    if mlp_model is None and gru_model is None:
        print("\n❌ No models trained. Record more data with python record.py")
        sys.exit(1)

    export_models(mlp_model, gru_model)

    print()
    print("=" * 60)
    print("  Training complete!")
    print("  Run python mememic.py to use the trained model(s).")
    print("=" * 60)


if __name__ == "__main__":
    main()
