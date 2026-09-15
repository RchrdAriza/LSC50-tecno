"""Handcrafted sequence features + robust classifier for LSC50 landmarks.

With only 980 clips (49 classes x 20), a per-frame RNN converges too slowly to
be competitive. This module extracts static per-clip descriptors (summary
statistics of the whole gesture) and trains a sklearn classifier. Much faster
to converge and a strong baseline for Stage 1.

Features (per clip, per landmark channel):
  - mean (absolute position)
  - std
  - first/last frame
  - per-axis min/max (range of motion)
  - max absolute deviation
  - mean velocity magnitude

`torso_relative` optionally re-centers body landmarks on the mid-shoulder point
each frame, encoding position relative to the signer's torso instead of the
image frame (marginal effect on this dataset; both variants within noise).
"""
from __future__ import annotations

import numpy as np

from .data import read_landmark_sequence


def extract_features(seq: np.ndarray, pad_to: int | None = None, torso_relative: bool = False) -> np.ndarray:
    """seq: (T, D). Returns a 1-D feature vector.

    If torso_relative, the body sequence is re-centered per-frame on the
    mid-shoulder point (MediaPipe Pose landmarks 11 and 12) before feature
    extraction.
    """
    if torso_relative:
        seq = _torso_center(seq)
    m = seq.mean(axis=0)
    centered = seq - m[None, :]

    # velocities between consecutive frames
    vel = np.diff(seq, axis=0)
    vel_mag = np.linalg.norm(vel, axis=1) if vel.shape[0] > 0 else np.zeros(0)

    features = [
        m,                                    # D mean
        centered.std(axis=0),                 # D std
        centered[0],                          # D first frame (relative)
        centered[-1],                         # D last frame (relative)
        centered.min(axis=0),                 # D min
        centered.max(axis=0),                 # D max
        np.abs(centered).max(axis=0),         # D max abs deviation
    ]
    if vel_mag.size:
        features += [
            vel_mag.mean(keepdims=True),      # 1
            vel_mag.std(keepdims=True),       # 1
        ]
    return np.concatenate(features).astype(np.float32)


def _torso_center(seq: np.ndarray) -> np.ndarray:
    """Re-center each frame on the mid-shoulder point.

    Expects body landmarks in MediaPipe Pose order (33 landmarks, xyz each):
    landmark 11 = left shoulder, 12 = right shoulder. Origin (0,0,0) is placed
    at the shoulder midpoint every frame. Assumes D is a multiple of 3.
    """
    D = seq.shape[1]
    n_landmarks = D // 3
    if n_landmarks <= 12:
        return seq  # not a full body pose; leave unchanged
    ls = seq[:, 11 * 3 : 11 * 3 + 3]  # left shoulder xyz
    rs = seq[:, 12 * 3 : 12 * 3 + 3]  # right shoulder xyz
    origin = (ls + rs) / 2.0          # (T, 3)
    origin_full = np.repeat(origin, n_landmarks, axis=1)  # (T, D)
    return seq - origin_full


def clip_features(clips: list[str], torso_relative: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix X (N, F) and labels y (N,) for a list of clips."""
    X, y = [], []
    for p in clips:
        seq = read_landmark_sequence(p)
        X.append(extract_features(seq, torso_relative=torso_relative))
        y.append(label_of(p))
    return np.vstack(X).astype(np.float32), np.asarray(y, dtype=np.int64)


def label_of(path: str) -> int:
    """Integer class id based on KEPT_SIGNS ordering."""
    from .config import KEPT_SIGNS
    from .data import parse_filename
    import os
    sign = parse_filename(os.path.basename(path))[0]
    return KEPT_SIGNS.index(sign)