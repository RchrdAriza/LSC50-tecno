"""Training/evaluation for the LSC50 landmark classifier (Stage 1).

Evaluates via leave-subject-out cross-validation: train on 4 volunteers, test on
the 5th. Reports per-fold and mean accuracy.
"""
from __future__ import annotations

import argparse
import collections
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import CLASS_LABELS, KEPT_SIGNS
from .data import (
    list_clips,
    label_of,
    read_landmark_sequence,
    sequences_to_array,
)
from .model import build_model


VOLUNTEERS = {f"{i:04d}" for i in range(5)}


def normalize_fit(X: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-feature mean/std from all train sequences (for standardization)."""
    all_rows = np.concatenate(X, axis=0)
    mean = all_rows.mean(axis=0)
    std = all_rows.std(axis=0) + 1e-6
    return mean, std


def normalize_apply(seq: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (seq - mean) / std


def collate_padded(batch):
    xs, ys, lens = [], [], []
    for x, y, l in batch:
        xs.append(x)
        ys.append(y)
        lens.append(l)
    X = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
    return X, torch.tensor(ys), torch.tensor(lens)


def load_split(clips, mean, std):
    seqs, ys = [], []
    for p in clips:
        seq = read_landmark_sequence(p)
        seqs.append(normalize_apply(seq, mean, std))
        ys.append(label_of(p))
    return seqs, np.asarray(ys, dtype=np.int64)


def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X, y, lengths in loader:
            X, lengths = X.to(device), lengths.to(device)
            logits = model(X, lengths)
            preds.append(logits.argmax(dim=-1).cpu().numpy())
            trues.append(y.numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return preds, trues


def train_one_fold(
    train_clips, val_clips, test_clips, input_dim, num_classes, args, device
):
    # fit normalization on training only
    train_seqs = [read_landmark_sequence(p) for p in train_clips]
    mean, std = normalize_fit(train_seqs)
    del train_seqs

    # build padded tensors
    def to_tensors(clips):
        seqs, ys = load_split(clips, mean, std)
        lens = [s.shape[0] for s in seqs]
        T = max(lens)
        X = np.zeros((len(seqs), T, input_dim), dtype=np.float32)
        for i, s in enumerate(seqs):
            X[i, : s.shape[0]] = s
        return torch.tensor(X), torch.tensor(ys), torch.tensor(lens)

    Xtr, ytr, ltr = to_tensors(train_clips)
    Xva, yva, lva = to_tensors(val_clips)
    Xte, yte, lte = to_tensors(test_clips)

    ds_tr = TensorDataset(Xtr, ytr, ltr)
    ds_va = TensorDataset(Xva, yva, lva)
    ds_te = TensorDataset(Xte, yte, lte)

    train_loader = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ds_va, batch_size=args.batch_size)
    test_loader = DataLoader(ds_te, batch_size=args.batch_size)

    model = build_model(input_dim, num_classes, hidden_dim=args.hidden, num_layers=args.layers)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = torch.nn.CrossEntropyLoss()

    best_val, best_state = -1.0, None
    patience = args.patience
    wait = 0

    for epoch in range(args.epochs):
        model.train()
        for X, y, lengths in train_loader:
            X, y, lengths = X.to(device), y.to(device), lengths.to(device)
            optimizer.zero_grad()
            logits = model(X, lengths)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        vp, vt = evaluate(model, val_loader, device)
        v_acc = (vp == vt).mean()
        if v_acc > best_val:
            best_val = v_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    tp, tt = evaluate(model, test_loader, device)
    return (tp == tt).mean(), best_val, tp, tt


def main():
    ap = argparse.ArgumentParser(description="Train LSC50 landmark classifier")
    ap.add_argument("--landmarks", default="body", choices=["body", "hand_l", "hand_r", "face"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    clips = list_clips(args.landmarks)
    input_dim = read_landmark_sequence(clips[0]).shape[1]
    num_classes = len(KEPT_SIGNS)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"landmarks={args.landmarks} input_dim={input_dim} classes={num_classes} device={device}")

    per_fold = []
    t0 = time.time()
    for vol in sorted(VOLUNTEERS):
        test_clips = [p for p in clips if os.path.basename(p).split("_")[1] == vol]
        rest = [p for p in clips if os.path.basename(p).split("_")[1] != vol]
        # hold out one volunteer from the rest for validation
        val_vol = next(v for v in sorted(VOLUNTEERS) if v != vol)
        val_clips = [p for p in rest if os.path.basename(p).split("_")[1] == val_vol]
        train_clips = [p for p in rest if os.path.basename(p).split("_")[1] != val_vol]

        acc, v_acc, tp, tt = train_one_fold(
            train_clips, val_clips, test_clips, input_dim, num_classes, args, device
        )
        per_fold.append(acc)
        print(f"  fold vol={vol}: test_acc={acc:.3f} (val_best={v_acc:.3f})")

    per_fold = np.array(per_fold)
    print(f"\nLeave-subject-out accuracy: mean={per_fold.mean():.3f} std={per_fold.std():.3f}")
    print(f"total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()