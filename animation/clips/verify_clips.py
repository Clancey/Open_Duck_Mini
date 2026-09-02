"""Validate generated starter clips against the authoritative robot URDF."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from xml.etree import ElementTree

CLIPS = Path(__file__).parent
URDF = CLIPS.parents[1] / "mini_bdx/robots/open_duck_mini_v2/robot.urdf"
LEGS = {"left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
        "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"}


def limits():
    return {joint.get("name"): (float(joint.find("limit").get("lower")), float(joint.find("limit").get("upper")))
            for joint in ElementTree.parse(URDF).findall("joint") if joint.find("limit") is not None}


def main():
    known = limits()
    errors = []
    for path in sorted(CLIPS.glob("*.duckanim.json")):
        data = json.loads(path.read_text())
        joints, frames = data["joints"], data["frames"]
        if len(frames) != round(data["duration"] * data["fps"]):
            errors.append(f"{path.name}: incorrect frame count")
        if any(len(row) != len(joints) for row in frames):
            errors.append(f"{path.name}: row width does not match joints")
        if any(joint not in known for joint in joints):
            errors.append(f"{path.name}: unknown joint")
        if data["layer"] == "override" and set(joints) & LEGS:
            errors.append(f"{path.name}: head-only override clip includes leg joints")
        for row in frames:
            for joint, value in zip(joints, row):
                if not math.isfinite(value) or not known[joint][0] <= value <= known[joint][1]:
                    errors.append(f"{path.name}: {joint} angle outside limit or non-finite")
    if errors:
        raise AssertionError("\n".join(errors))
    print(f"Validated {len(list(CLIPS.glob('*.duckanim.json')))} clips")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Clip verification failed: {error}", file=sys.stderr)
        raise SystemExit(1)
