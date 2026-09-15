"""Handcrafted sequence features + robust classifier for LSC50 landmarks.

With only 980 clips (49 classes x 20), a per-frame RNN converges too slowly to
be competitive. This module extracts static per-clip descriptors (summary
statistics of the whole gesture) and trains a sklearn classifier. Much faster
to converge and a strong baseline for Stage 1.

Features (per clip, per landmark channel):
  - mean, std (global and per-axis)
  - first/last frame
  - per-axis min/max (range of motion)
  - mean velocity magnitude
"""
from __future__ import annotations

import numpy as np

from .data import read_landmark_sequence


def extract_features(seq: np.ndarray, pad_to: int | None = None) -> np.ndarray:
    """seq: (T, D). Returns a 1-D feature vector."""
    # center each channel
    m = seq.mean(axis=0)
    centered = seq - m[None, :]

    # velocities between consecutive frames
    vel = np.diff(seq, axis=0)
    vel_mag = np.linalg.norm(vel, axis=1) if vel.shape[0] > 0 else np.zeros(0)

    features = [
        m,                                    # D mean
        seq.std(axis=0),                      # D std
        seq[0],                               # D first frame
        seq[-1],                              # D last frame
        seq.min(axis=0),                      # D min
        seq.max(axis=0),                      # D max
        np.abs(seq - m[None, :]).max(axis=0), # D max abs deviation
    ]
    if vel_mag.size:
        features += [
            vel_mag.mean(keepdims=True),      # 1
            vel_mag.std(keepdims=True),       # 1
        ]
    return np.concatenate(features).astype(np.float32)


def clip_features(clips: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix X (N, F) and labels y (N,) for a list of clips."""
    X, y = [], []
    for p in clips:
        seq = read_landmark_sequence(p)
        X.append(extract_features(seq))
        y.append(label_of(p))
    return np.vstack(X).astype(np.float32), np.asarray(y, dtype=np.int64)


def label_of(path: str) -> int:
    """Integer class id based on KEPT_SIGNS ordering."""
    from .config import KEPT_SIGNS
    from .data import parse_filename
    import os
    sign = parse_filename(os.path.basename(path))[0]
    return KEPT_SIGNS.index(sign)