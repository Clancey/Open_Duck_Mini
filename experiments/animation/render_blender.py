"""Blender authoring-view renderer for the Open Duck animation library.

Runs *inside* Blender (headless) and drives the 49-bone rig's FK bones from a
clip's 16 canonical joints, then renders a fixed 3/4 camera view per clip.

This is the *authoring* view (what the animator posed). It is deliberately
complementary to ``render_clips.py`` (MuJoCo), which shows the *physical* view.
A disagreement between the two is itself a bug signal.

Usage (do NOT run with plain python -- it needs Blender's bpy):

    /Applications/Blender.app/Contents/MacOS/Blender --background \
        /path/to/open-duck-mini.blend --enable-autoexec \
        --python experiments/animation/render_blender.py -- \
        --clips experiments/animation/clips \
        --out /path/to/renders/blender \
        --ref /path/to/standing_wiggle.json

Only the Blender render happens here (clean PNG per clip). Labelling and the
contact sheet are assembled afterwards by base-python PIL (see
``build_blender_contact_sheet`` in this file, called via a separate plain-python
invocation, because Blender does not bundle Pillow).
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

# ---------------------------------------------------------------------------
# Calibrated bone -> joint transform table, hardcoded to avoid needing the
# open_duck_anim package on Blender's sys.path. Kept in lock-step with
# blender/open_duck_anim_blender/transform_table.py (JOINT_ORDER_16 order).
# Each row: (bone_name, axis[0=X,1=Y,2=Z], sign, zero_offset_rad)
# Inverse used to drive bones:  euler[axis] = (joint - zero_offset) / sign
# ---------------------------------------------------------------------------
DEG10 = math.radians(10.0)
X, Y, Z = 0, 1, 2
JOINT_TRANSFORMS = (
    ("hip_yaw_fk.l", Y, 1.0, 0.0),      # 0  left_hip_yaw
    ("hip_roll_fk.l", Z, 1.0, 0.0),     # 1  left_hip_roll
    ("hip_pitch_fk.l", X, 1.0, 0.0),    # 2  left_hip_pitch
    ("knee_fk.l", X, 1.0, -DEG10),      # 3  left_knee
    ("ankle_fk.l", X, 1.0, DEG10),      # 4  left_ankle
    ("neck_pitch", X, 1.0, 0.0),        # 5  neck_pitch
    ("head_pitch", X, 1.0, 0.0),        # 6  head_pitch
    ("head_yaw", Z, 1.0, 0.0),          # 7  head_yaw
    ("head_roll", Z, 1.0, 0.0),         # 8  head_roll
    ("antenna.l", Z, 1.0, 0.0),         # 9  left_antenna
    ("antenna.r", Z, 1.0, 0.0),         # 10 right_antenna
    ("hip_yaw_fk.r", Y, 1.0, 0.0),      # 11 right_hip_yaw
    ("hip_roll_fk.r", Z, 1.0, 0.0),     # 12 right_hip_roll
    ("hip_pitch_fk.r", X, 1.0, 0.0),    # 13 right_hip_pitch
    ("knee_fk.r", X, 1.0, -DEG10),      # 14 right_knee
    ("ankle_fk.r", X, 1.0, DEG10),      # 15 right_ankle
)

RES = 480
CAM_TARGET = (0.0, 0.0, 0.26)
CAM_DIST = 1.35
CAM_AZ_DEG = 135.0   # 3/4 left, matches MuJoCo contact_sheet_3q_left
CAM_EL_DEG = 12.0


def _load_clip_joint_frames(path):
    """Return (name, list-of-16-float-frames) for a .duckanim or ref59 .json."""
    d = json.load(open(path))
    name = d.get("name") or os.path.splitext(os.path.basename(path))[0]
    if d.get("format") == "duckanim" or "joints" in d:
        frames = d["joints"]["frames"]
        return name, [list(f)[:16] for f in frames]
    # ref59 training reference: frames are 59-float, joints at 7:23
    frames = d["frames"] if "frames" in d else d.get("Frames") or d.get("data")
    return name, [list(f)[7:23] for f in frames]


def _representative_frame(frames):
    """Pick the frame with the largest total deviation from the mean pose."""
    n = len(frames)
    if n == 0:
        return []
    if n == 1:
        return frames[0]
    dim = len(frames[0])
    mean = [sum(fr[j] for fr in frames) / n for j in range(dim)]
    best_i, best_d = 0, -1.0
    for i, fr in enumerate(frames):
        d = sum(abs(fr[j] - mean[j]) for j in range(dim))
        if d > best_d:
            best_d, best_i = d, i
    return frames[best_i]


def _setup_scene(bpy):
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = RES
    scene.render.resolution_y = RES
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    try:
        shading = scene.display.shading
        shading.light = "STUDIO"
        shading.color_type = "SINGLE"
        shading.single_color = (0.72, 0.74, 0.78)
        shading.show_shadows = True
        shading.show_cavity = True
    except Exception:
        pass

    import mathutils

    az = math.radians(CAM_AZ_DEG)
    el = math.radians(CAM_EL_DEG)
    tgt = mathutils.Vector(CAM_TARGET)
    direction = mathutils.Vector(
        (math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))
    )
    cam_loc = tgt + CAM_DIST * direction

    cam = scene.camera
    if cam is None:
        cam_data = bpy.data.cameras.new("RenderCam")
        cam = bpy.data.objects.new("RenderCam", cam_data)
        scene.collection.objects.link(cam)
        scene.camera = cam
    cam.location = cam_loc
    look = (tgt - cam_loc)
    cam.rotation_euler = look.to_track_quat("-Z", "Y").to_euler()
    cam.data.lens = 55.0
    return scene


def _apply_pose(armature, joints16):
    for i, (bone_name, axis, sign, zero_off) in enumerate(JOINT_TRANSFORMS):
        pb = armature.pose.bones.get(bone_name)
        if pb is None:
            continue
        pb.rotation_mode = "XYZ"
        euler = list(pb.rotation_euler)
        euler[axis] = (joints16[i] - zero_off) / sign
        pb.rotation_euler = euler


def render_library(bpy, clips_dir, out_dir, refs):
    os.makedirs(out_dir, exist_ok=True)
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    armature = bpy.data.objects.get("Armature")
    if armature is None:
        raise SystemExit("Armature object not found in blend")
    # ensure FK drives the deform (fk_ik=0 == FK)
    try:
        armature.pose.bones["fk_ik_controller"]["fk_ik"] = 0.0
    except Exception:
        pass

    _setup_scene(bpy)

    paths = sorted(glob.glob(os.path.join(clips_dir, "*.duckanim")))
    paths += list(refs)

    manifest = []
    for path in paths:
        name, frames = _load_clip_joint_frames(path)
        if not frames:
            continue
        rep = _representative_frame(frames)
        _apply_pose(armature, rep)
        bpy.context.view_layer.update()
        out_png = os.path.join(frames_dir, name + ".png")
        bpy.context.scene.render.filepath = out_png
        bpy.ops.render.render(write_still=True)
        manifest.append({"name": name, "png": out_png, "n_frames": len(frames)})
        print("[blender] rendered", name, "->", out_png)

    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print("[blender] wrote manifest with", len(manifest), "clips")


def _parse_args(argv):
    # args after the "--" separator
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    clips = "experiments/animation/clips"
    out = "blender_out"
    refs = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--clips":
            clips = argv[i + 1]; i += 2
        elif a == "--out":
            out = argv[i + 1]; i += 2
        elif a == "--ref":
            refs.append(argv[i + 1]); i += 2
        else:
            i += 1
    return clips, out, refs


def main():
    try:
        import bpy  # noqa: F401
    except ImportError:
        raise SystemExit("render_blender.py must be run inside Blender (bpy missing)")
    import bpy
    clips, out, refs = _parse_args(sys.argv)
    render_library(bpy, clips, out, refs)


if __name__ == "__main__":
    main()
