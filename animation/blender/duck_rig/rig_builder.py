"""Build a constrained Blender armature from Open Duck Mini's URDF."""

from __future__ import annotations

from pathlib import Path

import bpy
from bpy.props import BoolProperty, StringProperty
from mathutils import Euler, Matrix, Vector

from .urdf_import import UrdfJoint, parse_urdf

GROUPS = {
    "legs": {"left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
             "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"},
    "head": {"neck_pitch", "head_pitch", "head_yaw", "head_roll"},
    "antennas": {"left_antenna", "right_antenna"},
}


def _urdf_matrix(joint: UrdfJoint) -> Matrix:
    return Matrix.Translation(joint.origin_xyz) @ Euler(joint.origin_rpy, "XYZ").to_matrix().to_4x4()


def _world_matrices(joints: list[UrdfJoint]) -> dict[str, Matrix]:
    by_parent: dict[str, list[UrdfJoint]] = {}
    for joint in joints:
        by_parent.setdefault(joint.parent, []).append(joint)
    result: dict[str, Matrix] = {}

    def visit(joint: UrdfJoint, parent_matrix: Matrix) -> None:
        matrix = parent_matrix @ _urdf_matrix(joint)
        result[joint.name] = matrix
        for child in by_parent.get(joint.child, []):
            visit(child, matrix)

    children = {joint.child for joint in joints}
    for joint in joints:
        if joint.parent not in children:
            visit(joint, Matrix.Identity(4))
    return result


def _collection_for(joint_name: str) -> str:
    for name, members in GROUPS.items():
        if joint_name in members:
            return name
    return "other"


def build_rig(urdf_path: str, attach_meshes: bool = False) -> bpy.types.Object:
    joints = parse_urdf(urdf_path)
    if not joints:
        raise ValueError(f"No revolute joints found in {urdf_path}")
    armature = bpy.data.armatures.new("OpenDuckMiniRig")
    rig = bpy.data.objects.new("OpenDuckMiniRig", armature)
    bpy.context.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")

    worlds = _world_matrices(joints)
    joint_by_child = {joint.child: joint for joint in joints}
    edit_bones = {}
    for joint in joints:
        world = worlds[joint.name]
        axis = (world.to_3x3() @ Vector(joint.axis)).normalized()
        head = world.translation
        bone = armature.edit_bones.new(joint.name)
        bone.head = head
        bone.tail = head + axis * 0.035
        edit_bones[joint.name] = bone
        parent_joint = joint_by_child.get(joint.parent)
        if parent_joint:
            bone.parent = edit_bones.get(parent_joint.name)
            bone.use_connect = False
    bpy.ops.object.mode_set(mode="POSE")

    for name in GROUPS:
        armature.collections.new(name)
    for joint in joints:
        pose_bone = rig.pose.bones[joint.name]
        pose_bone.rotation_mode = "XYZ"
        pose_bone.lock_rotation = (True, True, False)
        constraint = pose_bone.constraints.new("LIMIT_ROTATION")
        constraint.name = "URDF servo limits"
        constraint.owner_space = "LOCAL"
        constraint.use_limit_x = True
        constraint.min_x = constraint.max_x = 0.0
        constraint.use_limit_y = True
        constraint.min_y = constraint.max_y = 0.0
        constraint.use_limit_z = True
        constraint.min_z = joint.lower
        constraint.max_z = joint.upper
        armature.collections[_collection_for(joint.name)].assign(armature.bones[joint.name])
        for key, value in {
            "urdf_axis": list(joint.axis),
            "urdf_lower": joint.lower,
            "urdf_upper": joint.upper,
            "urdf_velocity": joint.velocity,
        }.items():
            pose_bone[key] = value
    bpy.ops.object.mode_set(mode="OBJECT")
    rig["urdf_path"] = str(Path(urdf_path).resolve())

    if attach_meshes:
        _attach_visual_meshes(rig, Path(urdf_path), joint_by_child)
    return rig


def _attach_visual_meshes(rig, urdf_path: Path, joint_by_child: dict[str, UrdfJoint]) -> None:
    """Best-effort STL import. A bad or absent visual mesh must not abort rigging."""
    from xml.etree import ElementTree

    root = ElementTree.parse(urdf_path).getroot()
    for link in root.findall("link"):
        joint = joint_by_child.get(link.get("name", ""))
        if joint is None:
            continue
        for mesh in link.findall("./visual/geometry/mesh"):
            filename = mesh.get("filename", "")
            path = urdf_path.parent / Path(filename).name
            if not path.exists() or path.suffix.lower() != ".stl":
                continue
            try:
                bpy.ops.wm.stl_import(filepath=str(path))
                imported = bpy.context.selected_objects[:]
                for obj in imported:
                    obj.parent = rig
                    obj.parent_type = "BONE"
                    obj.parent_bone = joint.name
            except RuntimeError:
                continue


class DUCK_OT_build_rig(bpy.types.Operator):
    bl_idname = "duck.build_rig"
    bl_label = "Build Open Duck Rig"
    bl_options = {"REGISTER", "UNDO"}

    urdf_path: StringProperty(name="URDF", subtype="FILE_PATH")
    attach_meshes: BoolProperty(name="Attach STL Visuals", default=False)

    def execute(self, context):
        path = bpy.path.abspath(self.urdf_path)
        try:
            build_rig(path, self.attach_meshes)
        except (OSError, ValueError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        return {"FINISHED"}
