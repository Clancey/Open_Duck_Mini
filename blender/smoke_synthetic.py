"""Headless Blender smoke test for the open_duck_anim_blender shims.

Builds a SYNTHETIC minimal armature (root + 16 FK bones + toe.l/toe.r), then
exercises the Blender-facing shims that the bpy-free unit tests cannot reach:
constraints.apply_limit_rotation_constraints and DataRecorder.record (the D4
deterministic loop), then export_and_compile. Run with:

    /Applications/Blender.app/Contents/MacOS/Blender --background --python \
        blender/smoke_synthetic.py -- <repo_root> <out_dir>

Exit code 0 == smoke passed. This does NOT need the real .blend and is meant to
run on Blender 4.1.1 to see how much of the pipeline that (older) version
tolerates. Whatever needs Blender >= 4.3.2 is reported by the caller.
"""

import os
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
repo_root = argv[0] if argv else os.getcwd()
out_dir = argv[1] if len(argv) > 1 else os.path.join(repo_root, "blender", "_smoke_out")
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, "blender"))

from open_duck_anim_blender import constraints, export as export_mod, recorder
from open_duck_anim_blender.metadata import ClipMetadata
from open_duck_anim_blender.transform_table import REQUIRED_BONES

BONES = list(REQUIRED_BONES) + ["root", "toe.l", "toe.r"]


def build_armature():
    # Fresh scene.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    arm_data = bpy.data.armatures.new("Armature")
    obj = bpy.data.objects.new("Armature", arm_data)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj

    bpy.ops.object.mode_set(mode="EDIT")
    for i, name in enumerate(BONES):
        eb = arm_data.edit_bones.new(name)
        eb.head = (i * 0.1, 0.0, 0.0)
        eb.tail = (i * 0.1, 0.0, 0.1)
    bpy.ops.object.mode_set(mode="POSE")
    for pb in obj.pose.bones:
        pb.rotation_mode = "XYZ"
    return obj


def main():
    obj = build_armature()
    print("SMOKE: built synthetic armature with", len(obj.pose.bones), "bones")

    # 1) constraints (idempotent — apply twice, expect no duplicates).
    touched = constraints.apply_limit_rotation_constraints("Armature")
    again = constraints.apply_limit_rotation_constraints("Armature")
    n_con = sum(len(pb.constraints) for pb in obj.pose.bones)
    assert touched == again, "constraint apply not idempotent"
    assert n_con == 14, "expected 14 constraints, got %d" % n_con
    print("SMOKE: applied", len(touched), "Limit Rotation constraints (idempotent, total=%d)" % n_con)

    # 2) deterministic recording over a short range.
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = 5
    # a tiny keyframe so the timeline actually varies
    hy = obj.pose.bones["head_yaw"]
    scene.frame_set(1)
    hy.rotation_euler[2] = 0.0
    hy.keyframe_insert(data_path="rotation_euler", index=2, frame=1)
    hy.rotation_euler[2] = 0.4
    hy.keyframe_insert(data_path="rotation_euler", index=2, frame=5)

    rec = recorder.DataRecorder(fps=50, contacts_valid=False)
    ep1 = rec.record()
    rec2 = recorder.DataRecorder(fps=50, contacts_valid=False)
    ep2 = rec2.record()
    assert ep1["Frames"] == ep2["Frames"], "D4 determinism failed: frames differ"
    assert len(ep1["Frames"]) == 5, "expected 5 frames"
    assert len(ep1["Frames"][0]) == 59, "frame not 59 floats"
    print("SMOKE: recorded", len(ep1["Frames"]), "frames; byte-identical re-record OK (D4)")

    # 3) export + compile (reuses open_duck_anim.compiler). 5 frames @50 = 0.1s,
    # so use short blend times to satisfy the compiler's blend<=duration rule.
    meta = ClipMetadata(
        name="smoke", layer_mask="head", blend_in_s=0.02, blend_out_s=0.02,
        show_blend_in_s=0.01, show_blend_out_s=0.01, contacts_valid=False,
    ).to_compiler_meta()
    os.makedirs(out_dir, exist_ok=True)
    src = os.path.join(out_dir, "smoke.source.json")
    out = os.path.join(out_dir, "smoke.duckanim")
    result = export_mod.export_and_compile(ep1, meta, src, out)
    assert os.path.exists(out), "no .duckanim produced"
    print("SMOKE: exported", os.path.basename(out), "sha", result["source_sha256"][:12])
    print("SMOKE: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
