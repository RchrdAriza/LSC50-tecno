# LSC50-tecno

Colombian Sign Language (LSC) motion-capture research dataset: inertial (IMU) + OpenSim kinematics + MediaPipe landmark tracks across 5 volunteers.

## Layout

- `IMU/` — per-volunteer inertial and biomechanics data (`RAW`, `STO`, `OUT_OPENSIM`, `INFO`).
- `LANDMARKS/` — MediaPipe landmark tracks (`BODY_LANDMARKS`, `FACE_LANDMARKS`, `HANDS_LANDMARKS`), 1000 CSVs each.

See `AGENTS.md` for detailed data semantics and format gotchas.

## Stage 1 — Sign classification (`lsc/`)

Recognizes the 49 kept LSC signs from MediaPipe landmark tracks, evaluated with
**leave-subject-out** cross-validation (train on 4 volunteers, test on the 5th).

### Setup

```sh
uv sync               # creates .venv with pinned deps (Python 3.11)
```

### Usage

```sh
# feature-based baseline, RandomForest on body + both hands (best config)
uv run python -m lsc.eval_features --combine body hand_l hand_r --model rf

# other models / single modality
uv run python -m lsc.eval_features --landmarks body --model mlp
uv run python -m lsc.eval_features --combine body hand_l hand_r face --model rf

# optional: re-center body landmarks on mid-shoulder point each frame
uv run python -m lsc.eval_features --combine body hand_l hand_r --model rf --torso-relative

# interactive demo: predict individual clips, show top-3 with probabilities
uv run python -m lsc.demo                                  # walk through all volunteers
uv run python -m lsc.demo --volunteer 0002 --sign 0030 --rep 0000   # single clip
```

Modules: `config.py` (50-sign dictionary + exclusion of `0049`), `data.py`
(landmark loaders + subject split), `features.py` (per-clip descriptors),
`model.py` (temporal GRU), `eval_features.py` (leave-subject-out benchmark),
`train.py` (GRU training loop), `demo.py` (interactive single-clip predictor).

### Results (leave-subject-out, 49 classes, random chance = 2%)

| Input                          | Model       | LSO accuracy |
|--------------------------------|-------------|--------------|
| body                           | RandomForest | ~21%        |
| body + hand_l + hand_r         | RandomForest | ~34%        |
| body + hand_l + hand_r + face  | RandomForest | ~26%        |
| body + hand_l + hand_r         | MLP          | ~21%        |

Findings:
- **Combining body + both hands is the most informative**; adding face hurts
  (high dimensionality, noisy, few samples).
- **RandomForest beats deep models** (GRU/MLP) on this small dataset (980 clips,
  20 per class) — ensembles generalize better across signers.
- Inter-volunteer variance is high (fold std ~0.15); this is the main challenge.

## Stage 2 — Rigs and 3D avatars (`lsc/rig.py`)

Builds a hierarchical skeleton (the **rig**) that drives 3D avatars from the
OpenSim inverse-kinematics data, plus per-sign motion clips.

### What the rig is

- **Topology** from the OpenSim model: pelvis → torso → arms/hands and
  pelvis → legs (14 bones), matching the `.osim` joint hierarchy.
- **Bone lengths** from the per-volunteer anthropometrics (`Volunteer_N.txt`;
  humerus/radius/back measured, the rest scaled from stature).
- **Joint channels** mapped to the 42 `.mot` columns: each bone rotates by its
  joint angles (degrees) in the documented axis order.
- **Segmentation** of each volunteer's continuous `.mot` into one clip per sign
  using `Timestamps.xlsx`, skipping corrupt zero-duration windows (V5/0049).

The **equivalent** orientation source is `IMU/STO/*.sto`: raw XSENS dot
world-frame quaternions for 6 IMU segments (torso, both humeri/radii, right
hand) — orientation-only, no relative joints, no left hand.

### Usage

```sh
# export a volunteer's rig + per-sign animations as JSON (renderer-agnostic)
uv run python -m lsc.export_rig --volunteer 1 --out AVATAR
uv run python -m lsc.export_rig --volunteer all --out AVATAR   # all 5

# python API
uv run python -c "from lsc.rig import segment, bone_lengths, forward_kinematics;
print(len(segment(1)), bone_lengths(1)['humerus_r']);
print(forward_kinematics(1, '0030', 0)['hand_r_tip'])"
```

Output per volunteer: `AVATAR/Volunteer_N/rig.json` (skeleton + channel
definitions) and one `<SIGN>_<GLOSS>.json` per sign (per-frame angles).
Generated under `AVATAR/` (gitignored, reproducible).

## Dataset

This project uses the **LSC50: Colombian Sign Language Video and Inertial Measurement Dataset**:

> Flórez-Sierra, A. F., Solórzano, B. D., Segura-Quijano, F., Cortés-Bello, Y. M., Cubillos, L., Giraldo, L. F., & Cifuentes-De la Portilla, C. (2024). *LSC50: Colombian Sign Language Video and Inertial Measurement dataset*. Scientific Data, 11, 1347.
> https://doi.org/10.1038/s41597-024-04172-5

The dataset was developed by the authors cited above and is used in this project for research and development purposes.