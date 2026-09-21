"""Stage 2: Rig for 3D avatars from the LSC50 OpenSim output.

The rig is the hierarchical skeleton that drives an avatar. It is built from:

  * the OpenSim model topology (IMU/INFO/OpenSim_Model.osim): pelvis -> torso
    -> arms/hands and pelvis -> legs,
  * the per-volunteer anthropometrics (IMU/INFO/Volunteer_N.txt): humerus,
    radius and back lengths (measured), the rest derived from stature,
  * the OpenSim IK angles (IMU/OUT_OPENSIM/Volunteer_N.mot, 42 columns),
    segmented into one clip per sign with IMU/INFO/Timestamps.xlsx.

The "equivalent" driver is the raw XSENS dot orientations (IMU/STO/*.sto):
absolute world-frame quaternions for 6 IMU segments, available as an
orientation-only source (no relative joint angles, no left hand).

The rig is expressed with nested local rotations: each bone applies one or
more joint rotations around the parent bone's local axes, in the order given
by the joint definition. Angles in the .mot are degrees; forward kinematics
converts to radians.
"""
from __future__ import annotations

import os
import re
import zipfile
from dataclasses import dataclass, field

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INFO_DIR = os.path.join(ROOT_DIR, "IMU", "INFO")
OPENSIM_DIR = os.path.join(ROOT_DIR, "IMU", "OUT_OPENSIM")
STO_DIR = os.path.join(ROOT_DIR, "IMU", "STO")

# Column names of the OpenSim IK storage file, in order. Index 0 => .mot row.
MOT_COLUMNS = [
    "time", "pelvis_tilt", "pelvis_list", "pelvis_rotation",
    "pelvis_tx", "pelvis_ty", "pelvis_tz",
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r",
    "knee_angle_r", "knee_angle_r_beta", "ankle_dorsiflexion_r",
    "ankle_inversion_r", "subtalar_angle_r", "mtp_angle_r",
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l",
    "knee_angle_l", "knee_angle_l_beta", "ankle_dorsiflexion_l",
    "ankle_inversion_l", "subtalar_angle_l", "mtp_angle_l",
    "lumbar_extension", "lumbar_bending", "lumbar_rotation",
    "arm_flex_r", "arm_add_r", "arm_rot_r", "elbow_flex_r",
    "pro_sup_r", "wrist_flex_r", "wrist_dev_r",
    "arm_flex_l", "arm_add_l", "arm_rot_l", "elbow_flex_l",
    "pro_sup_l", "wrist_flex_l", "wrist_dev_l",
]

# Every bone: name -> {parent, length source, joints}. Each joint is
# (joint_name, [(mot_column, axis), ...]) applied around the parent bone's
# LOCAL axes, in the given order. Length source is a key of the anthropometry
# file, 'root' (no length), or a derived fraction of stature.
BONES = {
    "pelvis":    {"parent": None,      "length": "root",   "axis": None,
                  "joints": [("ground_pelvis", [("pelvis_tilt", "x"), ("pelvis_list", "y"), ("pelvis_rotation", "z")])]},
    "torso":     {"parent": "pelvis",  "length": "back",   "axis": np.array([0.0, 1.0, 0.0]),
                  "joints": [("back", [("lumbar_extension", "x"), ("lumbar_bending", "y"), ("lumbar_rotation", "z")])]},
    # right arm
    "humerus_r": {"parent": "torso",   "length": "humerus", "axis": np.array([0.0, -1.0, 0.0]),
                  "joints": [("acromial_r", [("arm_flex_r", "x"), ("arm_add_r", "y"), ("arm_rot_r", "z")])]},
    "radius_r":  {"parent": "humerus_r", "length": "radius", "axis": np.array([0.0, -1.0, 0.0]),
                  "joints": [("elbow_r", [("elbow_flex_r", "y")]),
                             ("radioulnar_r", [("pro_sup_r", "z")])]},
    "hand_r":    {"parent": "radius_r", "length": "hand",   "axis": np.array([0.0, -1.0, 0.0]),
                  "joints": [("radius_hand_r", [("wrist_flex_r", "x"), ("wrist_dev_r", "y")])]},
    # left arm
    "humerus_l": {"parent": "torso",    "length": "humerus", "axis": np.array([0.0, -1.0, 0.0]),
                  "joints": [("acromial_l", [("arm_flex_l", "x"), ("arm_add_l", "y"), ("arm_rot_l", "z")])]},
    "radius_l":  {"parent": "humerus_l", "length": "radius", "axis": np.array([0.0, -1.0, 0.0]),
                  "joints": [("elbow_l", [("elbow_flex_l", "y")]),
                             ("radioulnar_l", [("pro_sup_l", "z")])]},
    "hand_l":    {"parent": "radius_l", "length": "hand",   "axis": np.array([0.0, -1.0, 0.0]),
                  "joints": [("radius_hand_l", [("wrist_flex_l", "x"), ("wrist_dev_l", "y")])]},
    # right leg
    "femur_r":   {"parent": "pelvis",   "length": "femur",  "axis": np.array([0.0, -1.0, 0.0]),
                  "joints": [("hip_r", [("hip_flexion_r", "x"), ("hip_adduction_r", "y"), ("hip_rotation_r", "z")])]},
    "tibia_r":   {"parent": "femur_r",  "length": "tibia",  "axis": np.array([0.0, -1.0, 0.0]),
                  "joints": [("knee_r", [("knee_angle_r", "x")])]},
    "foot_r":    {"parent": "tibia_r",  "length": "foot",   "axis": np.array([1.0, 0.0, 0.0]),
                  "joints": [("ankle_r", [("ankle_dorsiflexion_r", "x"), ("ankle_inversion_r", "y"), ("subtalar_angle_r", "z")])]},
    # left leg
    "femur_l":   {"parent": "pelvis",   "length": "femur",  "axis": np.array([0.0, -1.0, 0.0]),
                  "joints": [("hip_l", [("hip_flexion_l", "x"), ("hip_adduction_l", "y"), ("hip_rotation_l", "z")])]},
    "tibia_l":   {"parent": "femur_l",  "length": "tibia",  "axis": np.array([0.0, -1.0, 0.0]),
                  "joints": [("knee_l", [("knee_angle_l", "x")])]},
    "foot_l":    {"parent": "tibia_l",  "length": "foot",   "axis": np.array([1.0, 0.0, 0.0]),
                  "joints": [("ankle_l", [("ankle_dorsiflexion_l", "x"), ("ankle_inversion_l", "y"), ("subtalar_angle_l", "z")])]},
}

# Fraction of stature for segments not measured per volunteer (Leva 1996).
PROPORTIONS = {
    "femur": 0.245,
    "tibia": 0.245,
    "foot": 0.039,
    "hand": 0.108,
}


# --------------------------------------------------------------------------- #
# Anthropometrics
# --------------------------------------------------------------------------- #
def load_anthropometrics(volunteer: int) -> dict[str, float]:
    """Measured lengths (m) for a volunteer index 1..5."""
    path = os.path.join(INFO_DIR, f"Volunteer_{volunteer}.txt")
    with open(path) as f:
        text = f.read()
    data = {}
    for m in re.finditer(r"([A-Za-z0-9 ()]+):\s*([0-9.]+)\s*cm", text):
        key, val = m.group(1).strip(), float(m.group(2)) / 100.0
        if key == "back (from coccyx to cervical vertebrae C7)":
            key = "back"
        elif key == "sampling frecuency":
            continue
        data[key] = val
    return data


def bone_lengths(volunteer: int) -> dict[str, float]:
    """Per-bone length table (m): measured lengths + derived proportions."""
    anthro = load_anthropometrics(volunteer)
    lengths = {}
    for bone, spec in BONES.items():
        src = spec["length"]
        if src == "root":
            lengths[bone] = 0.0
        elif src in anthro:
            lengths[bone] = anthro[src]
        else:
            lengths[bone] = anthro["height"] * PROPORTIONS[src]
    return lengths


# --------------------------------------------------------------------------- #
# Storage readers (mot / sto)
# --------------------------------------------------------------------------- #
def _read_storage(path: str) -> tuple[list[str], np.ndarray]:
    """Read an OpenSim/STO storage file. Returns (columns, data)."""
    with open(path) as f:
        lines = f.readlines()
    idx = next(i for i, l in enumerate(lines) if l.strip().startswith("endheader"))
    header = lines[idx + 1].strip().split("\t")
    data = np.loadtxt(lines[idx + 2:], delimiter="\t")
    return header, data.astype(np.float64)


def load_mot(volunteer: int) -> tuple[list[str], np.ndarray]:
    """OpenSim IK sequence for a volunteer: (columns, rows with time col 0)."""
    cols, data = _read_storage(os.path.join(OPENSIM_DIR, f"Volunteer_{volunteer}.mot"))
    return cols, data


def load_sto(volunteer: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """XSENS dot IMU orientations: {sensor: (t, quats (N,4) w,x,y,z)}.

    Quaternion cells are comma-joined, so the storage cannot be read with
    np.loadtxt; each row is split on tabs and each sensor cell on commas.
    """
    path = os.path.join(STO_DIR, f"Volunteer_{volunteer}.sto")
    with open(path) as f:
        lines = f.readlines()
    idx = next(i for i, l in enumerate(lines) if l.strip().startswith("endheader"))
    header = lines[idx + 1].strip().split("\t")
    n_sensors = len(header) - 1
    t, quats_all = [], []
    for line in lines[idx + 2:]:
        cells = line.strip().split("\t")
        t.append(float(cells[0]))
        quats_all.append([[float(p) for p in cells[1 + i].split(",")[:4]] for i in range(n_sensors)])
    t = np.asarray(t)
    quats_all = np.asarray(quats_all)  # (N, n_sensors, 4)
    out = {}
    for i, sensor in enumerate(header[1:]):
        out[sensor] = (t, quats_all[:, i, :])
    return out


# --------------------------------------------------------------------------- #
# Timestamps (per-sign segmentation source)
# --------------------------------------------------------------------------- #
def parse_timestamps() -> dict[int, list[tuple[str, str, float, float]]]:
    """Read Timestamps.xlsx -> {volunteer_index 1..5: [(sign, gloss, t0, t1)]}.

    Columns: A = shared-string ref to sign id, B = shared-string ref to gloss,
    C = Time Start (s), D = Time End (s).
    """
    path = os.path.join(INFO_DIR, "Timestamps.xlsx")
    z = zipfile.ZipFile(path)
    shared = []
    for name in z.namelist():
        if name.startswith("xl/sharedStrings"):
            body = z.read(name).decode("utf-8", "replace")
            for m in re.finditer(r"<t[^>]*>([^<]*)</t>", body):
                shared.append(m.group(1))
    sheets = sorted(n for n in z.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml", n))
    out = {}
    for sheet_idx, sheet in enumerate(sheets, start=1):
        body = z.read(sheet).decode("utf-8", "replace")
        entries = []
        for row in re.findall(r"<row[^>]*>(.*?)</row>", body, re.S):
            cells = re.findall(r'<c r="([A-Z]+)\d+"(?:[^>]*?t="([^"]*)")?[^>]*>(?:<v>(.*?)</v>)?', row)
            vals = {}
            for col, ttype, v in cells:
                vals[col] = shared[int(v)] if ttype == "s" else v
            if {"A", "B", "C", "D"} <= set(vals) and vals["A"].isdigit():
                entries.append((vals["A"], vals["B"], float(vals["C"]), float(vals["D"])))
        out[sheet_idx] = entries
    return out


# --------------------------------------------------------------------------- #
# Clips
# --------------------------------------------------------------------------- #
@dataclass
class Clip:
    """A per-sign segment extracted from a volunteer's IK sequence."""
    volunteer: int
    sign: str
    gloss: str
    t0: float
    t1: float
    columns: list[str] = field(repr=False)
    data: np.ndarray = field(repr=False)  # (F, ncols); col 0 = time

    @property
    def frames(self) -> int:
        return self.data.shape[0]

    def angle(self, column: str) -> np.ndarray:
        return self.data[:, self.columns.index(column)]


def segment(volunteer: int, include_bad: bool = False) -> list[Clip]:
    """Slice a volunteer's .mot into one Clip per sign using the timestamps.

    Corrupt zero-duration windows (V5 sign 0049) are skipped unless
    include_bad=True.
    """
    cols, data = load_mot(volunteer)
    clips = []
    for sign, gloss, t0, t1 in parse_timestamps()[volunteer]:
        if t1 <= t0 and not include_bad:
            continue
        mask = (data[:, 0] >= t0) & (data[:, 0] <= t1)
        if not mask.any():
            continue
        clips.append(Clip(volunteer, sign, gloss, t0, t1, cols, data[mask]))
    return clips


# --------------------------------------------------------------------------- #
# Forward kinematics
# --------------------------------------------------------------------------- #
def _rot(axis: str, deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _bone_orientation(bone: str, clip: Clip, frame: int, parent_R: np.ndarray) -> np.ndarray:
    """World orientation of a bone given the clip angles for one frame."""
    R = parent_R
    for _, dofs in BONES[bone]["joints"]:
        for col, axis in dofs:
            R = _rot(axis, clip.data[frame][clip.columns.index(col)]) @ R
    return R


def forward_kinematics(volunteer: int, sign: str, frame: int = 0) -> dict[str, np.ndarray]:
    """3-D positions (m) of every bone origin+tip for one IK frame of a sign.

    Returns a dict {bone: origin (3,), 'bone_tip': tip (3,)} where bone also
    has an entry 'bone_axis' with the world-space bone direction.
    """
    clip = next(c for c in segment(volunteer) if c.sign == sign)
    lengths = bone_lengths(volunteer)

    pelvis = clip.data[frame]
    root_R = np.eye(3)
    for col, axis in [("pelvis_tilt", "x"), ("pelvis_list", "y"), ("pelvis_rotation", "z")]:
        root_R = _rot(axis, pelvis[clip.columns.index(col)]) @ root_R
    root_p = pelvis[[clip.columns.index("pelvis_tx"),
                     clip.columns.index("pelvis_ty"),
                     clip.columns.index("pelvis_tz")]]

    positions: dict[str, np.ndarray] = {}
    stack = [("pelvis", root_R, root_p)]
    while stack:
        bone, R, origin = stack.pop()
        # apply the bone's own joints relative to its parent frame
        if bone != "pelvis":
            R = _bone_orientation(bone, clip, frame, R)
        tip = origin + R @ (BONES[bone]["axis"] * lengths[bone]) if BONES[bone]["axis"] is not None else origin.copy()
        positions[bone] = origin.copy()
        positions[f"{bone}_tip"] = tip.copy()
        positions[f"{bone}_axis"] = R @ BONES[bone]["axis"] if BONES[bone]["axis"] is not None else R[:, 1].copy()
        for child, spec in BONES.items():
            if spec["parent"] == bone:
                stack.append((child, R, tip))
    return positions