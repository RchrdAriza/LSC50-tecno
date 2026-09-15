# AGENTS.md

Raw research dataset for Colombian Sign Language (LSC). Sources: the LSC50 dataset (Flórez-Sierra et al. 2024, Scientific Data 11:1347, doi:10.1038/s41597-024-04172-5) and its original code repo `BiomecanicaUniandes/LSC50`. **No source code lives in this repo** — treat the tree as a fixed dataset.

The dataset contains **50 LSC signs** performed by **5 volunteers** (3 native, 2 non-native), each with 4 repetitions.

## File naming — THE MOST IMPORTANT THING

Every data file is tagged `SIGN_VOLUNTEER_REPETITION` (three zero-padded 4-digit fields):

- `SIGN` (0000–0049) = **class label** → your training target.
- `VOLUNTEER` (0000–0004) = subject index.
- `REPETITION` (0000–0003) = repetition number.

This applies to `LANDMARKS/**/*.csv` AND to the `IMU/OUT_OPENSIM/*.mot` timestamps.

The sign→gloss dictionary (50 classes) is **only** in `IMU/INFO/Timestamps.xlsx` (col B of each sheet; e.g. 0000=Gracias, 0001=Buenos Días, ..., 0049=Mucho). It is NOT stored anywhere else. If you need labels, parse this file.

## Layout

- `IMU/` — per-volunteer inertial + biomechanics data (5 volunteers).
- `LANDMARKS/` — MediaPipe landmark tracks, 1000 CSVs each (50 signs × 5 vol × 4 reps):
  - `BODY_LANDMARKS/` — 33 landmarks (MediaPipe Pose). 100 cols.
  - `FACE_LANDMARKS/` — 468 landmarks (MediaPipe FaceMesh). 1405 cols.
  - `HANDS_LANDMARKS/{LEFT,RIGHT}_HAND_LANDMARKS/` — 21 landmarks (MediaPipe Hands). 64 cols.

## IMU/ structure

- `RAW/Volunteer_N.xlsx` — raw XSENS DOT sensor output. Multiple sheets named by body segment (`hand_l_imu`, `hand_r_imu`, `humerus_l/r_imu`, `radius_l_imu`, `raidus_r_imu` [sic, typo], `torso_imu`). Columns are numeric indices, not descriptive names.
- `STO/Volunteer_N.sto` — OpenSim storage, quaternion IMU orientations. Header ends at `endheader`; cols `time` + 6 sensors (`hand_r_imu`, `humerus_l/r_imu`, `radius_l/r_imu`, `torso_imu`). Quaternions are comma-separated 4-tuples inside each (otherwise tab-delimited) cell. `DataRate=120`.
- `OUT_OPENSIM/Volunteer_N.mot` — OpenSim inverse-kinematics, 42 cols, ~90k rows, `inDegrees=yes`. **Each `.mot` is ONE continuous session per volunteer containing all 50 signs** — you must segment it using the timestamps.
- `INFO/Volunteer_N.txt` — anthropometrics (cm) + sampling rate (125 Hz). Note: `.txt` says 125 Hz but sensors/OpenSim run at 120 Hz; reconcile if it matters.
- `INFO/OpenSim_Model.osim` — generic OpenSim model used for IK.
- `INFO/Timestamps.xlsx` — **5 sheets** (one per volunteer). Col A = sign id (shared-string index), col B = sign id `NNNN`, col C = gloss (Spanish word), col D = `Time End (s)`. Wait — see below; actual layout is A=shared index, B=sign id string, C=Time Start, D=Time End. Parsing note: column values mix shared-string refs (for sign id and gloss) and inline numeric cells (for times).

## Known data issues (verified)

- **`IMU/INFO/Timestamps.xlsx`, sheet5 (Volunteer 5), sign `0049` "Mucho":** `Time Start = Time End = 833.126 s` (zero duration). The `.mot`/`.sto` for V5 run to ~855.66 s and the landmarks for `0049_0004_000*` all exist with full frames, so this is a corrupt/missing timestamp entry — NOT missing data. Any segmentation relying on timestamps for this (sign,volunteer) pair will fail.
- Otherwise the 5 volunteers share the **identical ordered list of 50 signs**, no duplicated sign ids, all start<end, no overlaps, durations ~2.4–6.6 s (median ~4 s).
- Landmark files are complete: exactly 4 repetitions for every (sign, volunteer) in all four landmark sets.

## Landmark CSV row semantics (highly non-uniform)

- All landmark CSVs have a leading unnamed row index; first data row is frame `0`. Row count = frames in the clip (not marker count).
- Row counts vary widely per file: body ~25–100 frames, face ~100–430 frames. Hand files ~similar to body.
- Coordinates are raw MediaPipe output: body/hand x,y normalized 0–1; face x,y normalized 0–1, z = depth relative to head/wrist origin. Verify ranges before use.

## Gotchas

- `.sto` quaternion values are comma-joined 4-tuples inside tab-delimited cells.
- `.mot` angles are degrees, not radians.
- Sampling rates: `.txt` claims 125 Hz, OpenSim/`.sto` are 120 Hz, RGB body video is 24 FPS, face video 50 FPS, depth/IR 30 FPS. Do not assume alignment.
- No analysis code, notebooks, or scripts exist in this repo; any processing is new work. The upstream code repo (with `LoadFiles.ipynb`) is at `github.com/BiomecanicaUniandes/LSC50`.