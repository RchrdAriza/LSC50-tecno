"""Interactive demo for Stage 1: predict the sign of individual landmark clips.

Trains the RandomForest (body + both hands) on 4 volunteers, then lets the user
type a filename and shows the top-3 predicted signs with probabilities, plus
whether the prediction was correct. The test clip is held out from training by
volunteer.

Usage:
    uv run python -m lsc.demo
    uv run python -m lsc.demo --volunteer 0002
    uv run python -m lsc.demo --sign 0030 --rep 0
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from .config import CLASS_LABELS
from .data import LANDMARK_DIRS, list_clips, parse_filename, read_landmark_sequence
from .eval_features import clip_features_combined
from .features import extract_features, label_of

KEYS = ["body", "hand_l", "hand_r"]


def train_model(train_clips, keys=KEYS):
    Xtr, ytr = clip_features_combined(train_clips, keys)
    sc = StandardScaler().fit(Xtr)
    clf = RandomForestClassifier(n_estimators=300, random_state=0)
    clf.fit(sc.transform(Xtr), ytr)
    return sc, clf


def predict_one(sc, clf, sign, vol, rep, keys=KEYS):
    row_blocks = []
    for key in keys:
        path = LANDMARK_DIRS[key] / f"{sign}_{vol}_{rep}.csv"
        row_blocks.append(extract_features(read_landmark_sequence(str(path))))
    x = np.array(np.concatenate(row_blocks), dtype=np.float32).reshape(1, -1)
    probs = clf.predict_proba(sc.transform(x))[0]
    order = np.argsort(probs)[::-1]
    return order, probs


def show_prediction(sc, clf, sign, vol, rep):
    name = f"{sign}_{vol}_{rep}"
    order, probs = predict_one(sc, clf, sign, vol, rep)
    true = CLASS_LABELS.get(sign, sign)

    print(f"\n{name}  (signo real: {sign} = {true})")
    hit = order[0] == label_of(f"{sign}_{vol}_{rep}.csv")
    for rank in range(3):
        c = order[rank]
        mark = "  <-- top" if rank == 0 else ""
        print(f"  #{rank+1}  {c:02d} {CLASS_LABELS[c]:<14} {probs[c]:.1%}{mark}")
    print(f"  {'CORRECTO' if hit else 'INCORRECTO'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volunteer", default=None,
                    help="Hold out a specific volunteer for testing (default: interactive over all)")
    ap.add_argument("--sign", default=None, help="Test a single sign id, e.g. 0030")
    ap.add_argument("--rep", default=None, help="Repetition to test, e.g. 0")
    args = ap.parse_args()

    # all clips from the first modality
    clips = list_clips("body")
    vols = sorted({parse_filename(os.path.basename(p))[1] for p in clips})

    if args.sign is not None and args.volunteer is not None and args.rep is not None:
        # single prediction mode
        vols_train = [v for v in vols if v != args.volunteer]
        train = [p for p in clips if parse_filename(os.path.basename(p))[1] in vols_train]
        sc, clf = train_model(train)
        show_prediction(sc, clf, args.sign, args.volunteer, args.rep)
        return

    print("Entrenando RandomForest (body + manos) sobre 4 voluntarios por prueba...")
    for vol in vols:
        train = [p for p in clips if parse_filename(os.path.basename(p))[1] != vol]
        test = [p for p in clips if parse_filename(os.path.basename(p))[1] == vol]
        sc, clf = train_model(train)

        n = len(test)
        correct = 0
        print(f"\n=== Voluntario de prueba: {vol} ({n} clips) ===")
        for i, p in enumerate(test):
            sign, v, rep = parse_filename(os.path.basename(p))
            order, probs = predict_one(sc, clf, sign, v, rep)
            if order[0] == label_of(f"{sign}_{v}_{rep}.csv"):
                correct += 1
            show_prediction(sc, clf, sign, v, rep)
            if (i + 1) % 20 == 0:
                print(f"  ... {i+1}/{n} (aciertos hasta ahora: {correct})")
        print(f"  -> Voluntario {vol}: {correct}/{n} correctos ({correct/n:.1%})")

    # interactive: user types a clip filename to test
    print("\n=== Modo interactivo ===")
    print("Escribe un nombre de clip (ej. 0030_0002_0) para predecirlo, o 'q' para salir.")
    while True:
        raw = input("clip> ").strip()
        if raw.lower() in ("q", "quit", "exit", ""):
            break
        # accept both "0030_0002_0" and "0030_0002_0000"
        parts = raw.replace(".csv", "").split("_")
        if len(parts) != 3:
            print("  Formato inválido. Usa SIGN_VOLUNTEER_REP (ej. 0030_0002_0).")
            continue
        sign, vol, rep = (p.zfill(4) for p in parts)
        if vol not in vols:
            print(f"  Voluntario {vol} no válido. Usa uno de: {vols}")
            continue
        train = [p for p in clips if parse_filename(os.path.basename(p))[1] != vol]
        sc, clf = train_model(train)
        show_prediction(sc, clf, sign, vol, rep)


if __name__ == "__main__":
    main()