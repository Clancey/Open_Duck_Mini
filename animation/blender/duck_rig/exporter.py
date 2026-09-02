"""Bake Blender actions to the versioned Open Duck Mini clip format."""

from __future__ import annotations

import json
from pathlib import Path

import bpy
from bpy.props import StringProperty


def _action_settings(action):
    return {
        "loop": bool(action.get("duck_loop", False)),
        "blend_in": float(action.get("duck_blend_in", 0.25)),
        "blend_out": float(action.get("duck_blend_out", 0.25)),
        "priority": int(action.get("duck_priority", 10)),
        "layer": str(action.get("duck_layer", "override")),
        "tags": [tag.strip() for tag in str(action.get("duck_tags", "")).split(",") if tag.strip()],
    }


def _keyed_bones(action) -> set[str]:
    result = set()
    for fcurve in action.fcurves:
        path = fcurve.data_path
        if path.startswith('pose.bones["') and '"]' in path:
            result.add(path.split('"')[1])
    return result


def export_action(rig, action, output_directory: str, export_fps: float = 50.0) -> Path:
    """Evaluate *action* on *rig* and write one .duckanim.json file."""
    if export_fps <= 0:
        raise ValueError("Export fps must be positive")
    frame_start, frame_end = action.frame_range
    source_fps = bpy.context.scene.render.fps / bpy.context.scene.render.fps_base
    duration = (frame_end - frame_start + 1) / source_fps
    sample_count = round(duration * export_fps)
    previous_action = rig.animation_data.action if rig.animation_data else None
    previous_frame = scene.frame_current if (scene := bpy.context.scene) else 1
    if rig.animation_data is None:
        rig.animation_data_create()
    rig.animation_data.action = action
    keyed = _keyed_bones(action)
    values: dict[str, list[float]] = {name: [] for name in rig.pose.bones.keys()}
    for index in range(sample_count):
        scene.frame_set(frame_start + index * source_fps / export_fps)
        for name, bone in rig.pose.bones.items():
            values[name].append(float(bone.rotation_euler.z))
    rig.animation_data.action = previous_action
    scene.frame_set(previous_frame)

    joints = [
        name for name, samples in values.items()
        if name in keyed or (max(samples) - min(samples) > 1e-6)
    ]
    frames = [[values[name][index] for name in joints] for index in range(sample_count)]
    settings = _action_settings(action)
    if settings["layer"] not in {"override", "additive"}:
        raise ValueError("duck_layer must be 'override' or 'additive'")
    output = {
        "format_version": 1, "name": action.name, "fps": export_fps,
        "loop": settings["loop"], "duration": duration,
        "blend_in": settings["blend_in"], "blend_out": settings["blend_out"],
        "priority": settings["priority"], "layer": settings["layer"],
        "joints": joints, "joint_weights": {name: 1.0 for name in joints},
        "frames": frames,
        "metadata": {"tags": settings["tags"], "author": "", "source_blend": Path(bpy.data.filepath).name},
    }
    directory = Path(bpy.path.abspath(output_directory))
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{action.name}.duckanim.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return path


class DUCK_OT_export_clip(bpy.types.Operator):
    bl_idname = "duck.export_clip"
    bl_label = "Export Duck Clip"

    output_directory: StringProperty(name="Output directory", subtype="DIR_PATH")

    def execute(self, context):
        rig = context.active_object
        if rig is None or rig.type != "ARMATURE":
            self.report({"ERROR"}, "Select an armature")
            return {"CANCELLED"}
        strips = [
            strip for track in rig.animation_data.nla_tracks for strip in track.strips
            if strip.select and strip.action
        ] if rig.animation_data else []
        actions = [strip.action for strip in strips] or (
            [rig.animation_data.action] if rig.animation_data and rig.animation_data.action else []
        )
        if not actions:
            self.report({"ERROR"}, "Select NLA strips or assign an active Action")
            return {"CANCELLED"}
        try:
            paths = [export_action(rig, action, self.output_directory) for action in actions]
        except (OSError, ValueError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Exported {len(paths)} clip(s)")
        return {"FINISHED"}
