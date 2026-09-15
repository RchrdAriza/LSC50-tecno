"""Loaders for LSC50 landmark tracks.

Each landmark CSV is one clip: rows = frames, columns = per-landmark x,y,z
(with a leading unnamed frame index column). The sign label is taken from the
first field of the filename, NOT from any annotation file.
"""
from __future__ import annotations

import glob
import os
import random

import numpy as np

from .config import LANDMARK_DIRS, CLASS_LABELS, KEPT_SIGNS, is_included, parse_filename


def read_landmark_sequence(path: str) -> np.ndarray:
    """Load a landmark CSV into a (T, D) float array, dropping the frame index column."""
    # skip header; data is numeric
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    if arr.ndim == 1:
        arr = arr[None, :]
    # drop leading unnamed frame-index column
    return arr[:, 1:].astype(np.float32)


def list_clips(landmark_key: str = "body") -> list[str]:
    """Return all CSV paths for a landmark set, filtering excluded signs."""
    directory = LANDMARK_DIRS[landmark_key]
    return [p for p in glob.glob(str(directory / "*.csv")) if is_included(parse_filename(os.path.basename(p))[0])]


def subject_split(
    clips: list[str],
    test_volunteers: set[str],
    val_volunteers: set[str] | None = None,
    random_state: int = 0,
    val_ratio: float = 0.2,
) -> dict[str, list[str]]:
    """Partition clips into train/val/test by volunteer (subject-exclusive split).

    test = clips from test_volunteers. If val_volunteers is None, a random subset
    of the remaining volunteers is held out for validation (or random clips if too
    few volunteers remain). Returns {'train': [...], 'val': [...], 'test': [...]}.
    """
    def vol_of(p):
        return parse_filename(os.path.basename(p))[1]

    test = [p for p in clips if vol_of(p) in test_volunteers]
    rest = [p for p in clips if vol_of(p) not in test_volunteers]

    if val_volunteers is not None:
        val = [p for p in rest if vol_of(p) in val_volunteers]
        train = [p for p in rest if vol_of(p) not in val_volunteers]
    else:
        # random clip-level split from the remaining pool
        rng = random.Random(random_state)
        rng.shuffle(rest)
        n_val = int(len(rest) * val_ratio)
        val, train = rest[:n_val], rest[n_val:]

    return {"train": train, "val": val, "test": test}


def label_of(path: str) -> int:
    """Integer class id for a clip, based on KEPT_SIGNS ordering."""
    sign = parse_filename(os.path.basename(path))[0]
    return KEPT_SIGNS.index(sign)


def sequences_to_array(clips: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) where X is a list-compatible object (np.ndarray of float32),
    y is integer labels. Sequences have variable length; pad externally."""
    xs, ys = [], []
    for p in clips:
        xs.append(read_landmark_sequence(p))
        ys.append(label_of(p))
    return xs, np.asarray(ys, dtype=np.int64)


def pad_sequences(
    sequences: list[np.ndarray], pad_value: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Pad a list of (T_i, D) arrays to a single (N, T_max, D) tensor.

    Returns (padded, lengths).
    """
    lengths = np.asarray([s.shape[0] for s in sequences], dtype=np.int64)
    T_max = int(lengths.max())
    D = sequences[0].shape[1]
    out = np.full((len(sequences), T_max, D), pad_value, dtype=np.float32)
    for i, s in enumerate(sequences):
        out[i, : s.shape[0]] = s
    return out, lengths