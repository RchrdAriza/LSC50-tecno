"""Export the LSC50 rig and per-sign animations to JSON.

Produces, for every volunteer:
  * a rig.json describing the skeleton hierarchy, bone lengths and the joint
    angle channels (which .mot columns rotate which bone, and in which order),
  * one <SIGN>_<GLOSS>.json per sign with the per-frame joint angles (deg)
    sampled from the .mot for that sign's time window.

The output is renderer-agnostic: a 3D engine applies each bone's joint angles
as nested local rotations in the documented axis order.

Usage:
    uv run python -m lsc.export_rig --volunteer 1 --out AVATAR
    uv run python -m lsc.export_rig --volunteer all --out AVATAR
"""
from __future__ import annotations

import argparse
import json
import os

from .rig import BONES, MOT_COLUMNS, PROPORTIONS, bone_lengths, load_anthropometrics, segment


def rig_schema(volunteer: int) -> dict:
    lengths = bone_lengths(volunteer)
    bones = []
    for bone, spec in BONES.items():
        entry = {
            "name": bone,
            "parent": spec["parent"],
            "length": lengths[bone],
            "axis": spec["axis"].tolist() if spec["axis"] is not None else None,
            "joints": [
                {"joint": jname, "dofs": [(col, axis) for col, axis in dofs]}
                for jname, dofs in spec["joints"]
            ],
        }
        bones.append(entry)
    anthro = load_anthropometrics(volunteer)
    return {
        "dataset": "LSC50",
        "stage": 2,
        "volunteer": volunteer,
        "schema": "lsc50-rig",
        "units": "m, degrees",
        "anthropometrics": anthro,
        "length_sources": {"measured": ["humerus", "radius", "back"], "proportions": PROPORTIONS},
        "mot_columns": MOT_COLUMNS,
        "bones": bones,
    }


def sign_animation(volunteer: int, sign: str) -> dict:
    for clip in segment(volunteer):
        if clip.sign == sign:
            break
    else:
        raise KeyError(f"sign {sign} not found for volunteer {volunteer}")
    frames = []
    for row in clip.data:
        frames.append({col: float(row[i]) for i, col in enumerate(clip.columns)})
    return {
        "dataset": "LSC50",
        "stage": 2,
        "volunteer": clip.volunteer,
        "sign": clip.sign,
        "gloss": clip.gloss,
        "t0": clip.t0,
        "t1": clip.t1,
        "fps": 120.0,
        "frames": frames,
    }


def export(volunteer: int, outdir: str) -> int:
    os.makedirs(f"{outdir}/Volunteer_{volunteer}", exist_ok=True)
    with open(f"{outdir}/Volunteer_{volunteer}/rig.json", "w") as f:
        json.dump(rig_schema(volunteer), f, indent=1)
    n = 0
    for clip in segment(volunteer):
        name = f"{clip.sign}_{clip.gloss.replace(' ', '_')}.json"
        with open(f"{outdir}/Volunteer_{volunteer}/{name}", "w") as f:
            json.dump(sign_animation(volunteer, clip.sign), f)
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volunteer", help="Volunteer index (1..5) or 'all'")
    ap.add_argument("--out", default="AVATAR", help="Output directory")
    args = ap.parse_args()

    vols = [int(args.volunteer)] if args.volunteer.isdigit() else range(1, 6)
    for v in vols:
        n = export(v, args.out)
        print(f"Volunteer {v}: rig + {n} sign animations -> {args.out}/Volunteer_{v}")


if __name__ == "__main__":
    main()