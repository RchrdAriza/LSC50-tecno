"""Feature-based leave-subject-out evaluation for LSC50 (Stage 1 baseline).

Trains a classifier on per-clip summary features, evaluates on a held-out
volunteer. Reports mean/std accuracy across the 5 folds.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .data import list_clips, parse_filename, read_landmark_sequence
from .features import clip_features, extract_features, label_of


MODELS = {
    "logreg": lambda: LogisticRegression(max_iter=2000),
    "rf": lambda: RandomForestClassifier(n_estimators=300, random_state=0),
    "mlp": lambda: MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=2000, random_state=0),
}


def clip_features_combined(base_clips: list[str], landmarks: list[str], torso_relative: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate per-clip features across multiple landmark sets.

    base_clips are the resolved paths for the first modality; the same
    (sign, volunteer, repetition) is resolved into each other landmark set's
    directory. Returns (X, y) with X = concat of each modality's features.
    """
    from .data import LANDMARK_DIRS, parse_filename

    feat_blocks = []
    for base in base_clips:
        sign, vol, rep = parse_filename(os.path.basename(base))
        row_blocks = []
        for key in landmarks:
            path = LANDMARK_DIRS[key] / f"{sign}_{vol}_{rep}.csv"
            # torso-relative centering only applies to body poses
            rel = torso_relative and key == "body"
            row_blocks.append(extract_features(read_landmark_sequence(str(path)), torso_relative=rel))
        feat_blocks.append(np.concatenate(row_blocks))
    X = np.vstack(feat_blocks).astype(np.float32)
    # labels from the first modality
    y = np.asarray([label_of(p) for p in base_clips], dtype=np.int64)
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmarks", default="body", choices=["body", "hand_l", "hand_r", "face"])
    ap.add_argument("--combine", nargs="*", default=None,
                    help="Concatenate multiple landmark sets, e.g. --combine body hand_l hand_r")
    ap.add_argument("--model", default="mlp", choices=list(MODELS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--torso-relative", action="store_true",
                    help="Re-center body landmarks on mid-shoulder point each frame")
    args = ap.parse_args()

    keys = args.combine if args.combine else [args.landmarks]
    clips = list_clips(keys[0])
    volunteers = sorted({parse_filename(os.path.basename(p))[1] for p in clips})
    print(f"landmarks={keys} clips={len(clips)} volunteers={volunteers}")

    per_fold = []
    for vol in volunteers:
        test = [p for p in clips if parse_filename(os.path.basename(p))[1] == vol]
        train = [p for p in clips if parse_filename(os.path.basename(p))[1] != vol]

        Xtr, ytr = clip_features_combined(train, keys, torso_relative=args.torso_relative)
        Xte, yte = clip_features_combined(test, keys, torso_relative=args.torso_relative)

        sc = StandardScaler().fit(Xtr)
        Xtr_s = sc.transform(Xtr)
        Xte_s = sc.transform(Xte)

        clf = MODELS[args.model]()
        clf.fit(Xtr_s, ytr)
        acc = (clf.predict(Xte_s) == yte).mean()
        per_fold.append(acc)
        print(f"  fold vol={vol}: test_acc={acc:.3f}")

    per_fold = np.array(per_fold)
    print(f"\nLeave-subject-out accuracy ({args.model}, {keys}): mean={per_fold.mean():.3f} std={per_fold.std():.3f}")


if __name__ == "__main__":
    main()