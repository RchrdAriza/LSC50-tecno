# AGENTS.md

Raw research dataset (motion capture + IMU + OpenSim kinematics). No source code, build system, tests, or CI. This file documents the data layout only.

## Layout

Two top-level data domains:

- `IMU/` — per-volunteer inertial + biomechanics data (5 volunteers).
- `LANDMARKS/` — MediaPipe-style 2D landmark tracks per camera, per frame.

Not a git repo; treat the tree as a fixed dataset, not a codebase to build.

## IMU/ structure (5 volunteers)

- `RAW/Volunteer_N.xlsx` — raw sensor time series (large sheets, ~88k rows).
- `STO/Volunteer_N.sto` — OpenSim storage file, quaternion IMU orientations. Header block ends at `endheader`; columns: `time` + 6 IMU sensors (`hand_r_imu`, `humerus_l/r_imu`, `radius_l/r_imu`, `torso_imu`). Quaternions are comma-separated 4-tuples within each cell. `DataRate=120`.
- `OUT_OPENSIM/Volunteer_N.mot` — OpenSim inverse-kinematics results (42 columns, ~90k rows, `inDegrees=yes`).
- `INFO/Volunteer_N.txt` — subject anthropometrics (height, segment lengths) + sampling rate (125 Hz). Units: cm, Hz.
- `INFO/OpenSim_Model.osim` — the OpenSim model used for IK.
- `INFO/Timestamps.xlsx` — 5 sheets (one per volunteer); rows are camera index pairs `(start, end)` timestamps in seconds.

## LANDMARKS/ structure

Four landmark sets, each with 1000 CSV files named `CAMERA_TRIAL_SEGMENT.csv` (three zero-padded integers):

- `BODY_LANDMARKS/` — 33 landmarks (MediaPipe Pose). 100 columns: unnamed index + `landmark_{0..32}_{x,y,z}`.
- `FACE_LANDMARKS/` — 468 landmarks (MediaPipe FaceMesh). 1405 columns.
- `HANDS_LANDMARKS/LEFT_HAND_LANDMARKS/` and `RIGHT_HAND_LANDMARKS/` — 21 landmarks (MediaPipe Hands). 64 columns.

Row-count semantics (highly non-uniform, don't assume fixed length):
- `BODY_LANDMARKS` and hand CSVs have a leading unnamed row number column; first data row is frame `0`.
- `FACE_LANDMARKS` also has a leading unnamed row index column.
- Data rows per file vary widely (body ~25–100; face ~100–430). Row count is the number of frames in the clip, not a marker count.

Coordinates are raw MediaPipe output (normalized 0–1 range, x/y/z per landmark). Verify units/ranges before use; there is no schema or README to confirm conventions.

## Gotchas

- CSV cells contain comma-separated quaternions in `.sto` files — they are tab-delimited fields but the quaternion value itself is comma-joined.
- `.mot` output includes `inDegrees=yes`; angles are degrees, not radians.
- IMU sampling is 125 Hz (`INFO/*.txt`) while OpenSim outputs are 120 Hz (`DataRate` / `.sto`). Do not assume they are aligned or equal.
- No scripts, notebooks, or processing code exist in the repo; any analysis is entirely new work.