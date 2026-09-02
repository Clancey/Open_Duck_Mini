"""Generate the Open Duck Mini starter .duckanim.json clip library."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

FPS = 50
LIMITS = {
    "left_ankle": (-math.pi / 2, math.pi / 2), "left_knee": (-math.pi / 2, math.pi / 2),
    "left_hip_pitch": (-1.2217304763960306, math.pi / 6), "left_hip_roll": (-math.pi / 7.2, math.pi / 7.2),
    "right_ankle": (-math.pi / 2, math.pi / 2), "right_knee": (-math.pi / 2, math.pi / 2),
    "right_hip_pitch": (-math.pi / 6, 1.2217304763960306), "right_hip_roll": (-math.pi / 7.2, math.pi / 7.2),
    "neck_pitch": (-math.pi / 9, 1.1344640137963142), "head_pitch": (-math.pi / 4, math.pi / 4),
    "head_yaw": (-2.792526803190927, 2.792526803190927), "head_roll": (-math.pi / 6, math.pi / 6),
    "left_antenna": (-math.pi / 2, math.pi / 2), "right_antenna": (-math.pi / 2, math.pi / 2),
}
LEGS = {"left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
        "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"}


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def curve(times, points):
    """Interpolate deliberately posed key points with smoothstep easing."""
    key_times, key_values = zip(*points)
    return np.interp(times, key_times, key_values) if len(points) == 1 else np.array([
        key_values[min(np.searchsorted(key_times, t, side="right") - 1, len(points) - 2)]
        + (key_values[min(np.searchsorted(key_times, t, side="right") - 1, len(points) - 2) + 1]
           - key_values[min(np.searchsorted(key_times, t, side="right") - 1, len(points) - 2)])
        * smoothstep((t - key_times[min(np.searchsorted(key_times, t, side="right") - 1, len(points) - 2)])
                     / (key_times[min(np.searchsorted(key_times, t, side="right") - 1, len(points) - 2) + 1]
                        - key_times[min(np.searchsorted(key_times, t, side="right") - 1, len(points) - 2)]))
        for t in times
    ])


def clip(name, duration, joints, values, *, loop=False, blend=0.25, priority=10, layer="override", tags=()):
    n = round(duration * FPS)
    frames = np.column_stack([values[joint] for joint in joints])
    for index, joint in enumerate(joints):
        low, high = LIMITS[joint]
        frames[:, index] = np.clip(frames[:, index], low + 0.002, high - 0.002)
    assert frames.shape == (n, len(joints)) and np.all(np.isfinite(frames))
    return {"format_version": 1, "name": name, "fps": FPS, "loop": loop, "duration": duration,
            "blend_in": blend, "blend_out": blend, "priority": priority, "layer": layer,
            "joints": joints, "joint_weights": {joint: 1.0 for joint in joints},
            "frames": frames.tolist(), "metadata": {"tags": list(tags), "author": "", "source_blend": ""}}


def make_clips():
    def t(duration): return np.arange(round(duration * FPS)) / FPS
    clips = []
    time = t(4.0)
    clips.append(clip("idle_breathe", 4.0, ["neck_pitch", "head_pitch"],
                      {"neck_pitch": 0.035 * np.sin(2 * math.pi * time / 4), "head_pitch": 0.045 * np.sin(2 * math.pi * time / 4 + .3)},
                      loop=True, blend=.4, priority=0, tags=("idle", "baseline")))
    clips.append(clip("look_around", 4.0, ["head_yaw", "head_pitch"],
                      {"head_yaw": curve(time, [(0, 0), (.7, .65), (1.2, .65), (2.0, -.65), (2.5, -.65), (3.4, 0), (4, 0)]),
                       "head_pitch": curve(time, [(0, 0), (.7, -.10), (1.2, -.10), (2, .06), (2.5, .06), (3.4, 0), (4, 0)])},
                      blend=.25, tags=("idle", "gaze")))
    time = t(1.2)
    clips.append(clip("nod_yes", 1.2, ["head_pitch"],
                      {"head_pitch": curve(time, [(0, 0), (.18, -.22), (.34, .12), (.52, -.22), (.7, .12), (.95, 0), (1.2, 0)])},
                      blend=.15, tags=("gesture", "affirmative")))
    time = t(1.4)
    clips.append(clip("shake_no", 1.4, ["head_yaw", "head_roll"],
                      {"head_yaw": curve(time, [(0, 0), (.2, .38), (.45, -.38), (.7, .38), (.95, -.38), (1.2, 0), (1.4, 0)]),
                       "head_roll": curve(time, [(0, 0), (.2, -.06), (.45, .06), (.7, -.06), (.95, .06), (1.2, 0), (1.4, 0)])},
                      blend=.15, tags=("gesture", "negative")))
    time = t(2.0)
    clips.append(clip("curious_tilt", 2.0, ["head_roll", "head_pitch", "left_antenna", "right_antenna"],
                      {"head_roll": curve(time, [(0, 0), (.55, .22), (1.45, .22), (2, 0)]),
                       "head_pitch": curve(time, [(0, 0), (.55, -.13), (1.45, -.13), (2, 0)]),
                       "left_antenna": curve(time, [(0, 0), (.55, .25), (1.45, .25), (2, 0)]),
                       "right_antenna": curve(time, [(0, 0), (.55, .25), (1.45, .25), (2, 0)])},
                      blend=.25, tags=("gesture", "curious")))
    time = t(.8)
    clips.append(clip("alert_perk", .8, ["head_pitch", "left_antenna", "right_antenna"],
                      {"head_pitch": curve(time, [(0, 0), (.15, .18), (.5, .18), (.8, 0)]),
                       "left_antenna": curve(time, [(0, 0), (.15, .42), (.5, .42), (.8, 0)]),
                       "right_antenna": curve(time, [(0, 0), (.15, .42), (.5, .42), (.8, 0)])},
                      blend=.15, priority=20, tags=("alert", "gesture")))
    time = t(2.5)
    clips.append(clip("sad_droop", 2.5, ["neck_pitch", "head_pitch", "left_antenna", "right_antenna"],
                      {"neck_pitch": curve(time, [(0, 0), (1.2, -.18), (2, -.18), (2.5, 0)]),
                       "head_pitch": curve(time, [(0, 0), (1.2, -.28), (2, -.28), (2.5, 0)]),
                       "left_antenna": curve(time, [(0, 0), (1.2, -.35), (2, -.35), (2.5, 0)]),
                       "right_antenna": curve(time, [(0, 0), (1.2, -.35), (2, -.35), (2.5, 0)])},
                      blend=.35, tags=("gesture", "sad")))
    time = t(1.0)
    clips.append(clip("antenna_wiggle", 1.0, ["left_antenna", "right_antenna"],
                      {"left_antenna": curve(time, [(0, 0), (.18, .32), (.38, -.22), (.62, .22), (.82, 0), (1, 0)]),
                       "right_antenna": curve(time, [(0, 0), (.18, -.32), (.38, .22), (.62, -.22), (.82, 0), (1, 0)])},
                      blend=.15, tags=("gesture", "antenna")))
    time = t(1.0); bob = .10 * (1 - np.cos(2 * math.pi * time))
    clips.append(clip("happy_bounce", 1.0, ["left_hip_pitch", "right_hip_pitch", "left_knee", "right_knee", "left_ankle", "right_ankle", "head_pitch"],
                      {"left_hip_pitch": -.25 * bob, "right_hip_pitch": -.25 * bob, "left_knee": .55 * bob, "right_knee": .55 * bob,
                       "left_ankle": -.3 * bob, "right_ankle": -.3 * bob, "head_pitch": -.15 * bob},
                      loop=True, blend=.2, layer="additive", tags=("full_body", "happy")))
    time = t(2.0); sway = .10 * np.sin(2 * math.pi * time / 2)
    clips.append(clip("dance_sway", 2.0, ["left_hip_roll", "right_hip_roll", "left_ankle", "right_ankle", "head_roll"],
                      {"left_hip_roll": sway, "right_hip_roll": sway, "left_ankle": -.7 * sway, "right_ankle": -.7 * sway, "head_roll": -.65 * sway},
                      loop=True, blend=.3, layer="additive", tags=("full_body", "dance")))
    return clips


if __name__ == "__main__":
    directory = Path(__file__).parent
    for item in make_clips():
        (directory / f"{item['name']}.duckanim.json").write_text(json.dumps(item, indent=2) + "\n")
    print(f"Generated {len(make_clips())} clips in {directory}")
