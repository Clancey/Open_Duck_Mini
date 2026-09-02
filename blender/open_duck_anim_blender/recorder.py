"""Blender-facing recorder shim: deterministic ``frame_set()`` loop (defect D4).

This is the ONLY recorder module that touches Blender, and it is a thin shim:
all the maths (bone->joint transform, foot contacts, 59-float assembly) lives in
the bpy-free modules and is reused here, never reimplemented.

``import bpy`` is guarded so this module can be *imported* on a machine without
Blender (CI). :class:`DataRecorder` still needs a running Blender to actually
record; the guard only keeps import-time side effects away from the test host.

Defect D4 — determinism. Upstream drove recording from a wall-clock timer
(``bpy.ops.screen.animation_play()`` + ``bpy.app.timers.register`` returning
``1/FPS``), so a slow scene dropped or duplicated frames. :meth:`DataRecorder.record`
replaces that with an explicit ``scene.frame_set(f)`` loop that steps the
timeline frame by frame and captures exactly one sample per frame, independent of
playback speed. Re-recording the same scene twice therefore yields byte-identical
frames (Phase 2 acceptance).

Defects D2/D11 — joint ordering and calibration are handled entirely by
:mod:`.transform_table` (imported, not duplicated). Defect D3 — contacts come
from :mod:`.contacts`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from . import contacts as contacts_mod
from . import export as export_mod
from .transform_table import REQUIRED_BONES, joints_from_bone_eulers

try:  # guarded so CI (no Blender) can import this module
    import bpy  # type: ignore
except Exception:  # pragma: no cover - exercised only inside Blender
    bpy = None  # type: ignore


# --- bpy-free root-orientation maths (unit-tested in tests/) ----------------
def euler_to_quaternion_xyzw(yaw, pitch, roll):
    """ZYX (yaw-pitch-roll) Euler → quaternion in scipy ``as_quat`` XYZW order.

    This is the exact emitted convention consumed downstream at ``frame[3:7]``
    (``R.from_quat`` / ``R.from_matrix(...).as_quat()`` in the reference-motion
    pipeline — XYZW, *not* the stale ``qw,qx,qy,qz`` comments in that code).
    """
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    return np.array(
        [
            sr * cp * cy - cr * sp * sy,  # x
            cr * sp * cy + sr * cp * sy,  # y
            cr * cp * sy - sr * sp * cy,  # z
            cr * cp * cy + sr * sp * sy,  # w
        ]
    )


def root_frame_euler_rpy(rx, ry, rz):
    """Map the root bone's local Euler ``(rx, ry, rz)`` to blender-frame RPY.

    Reproduces the upstream axis convention exactly: roll is negated, pitch and
    yaw pass through. Returns ``[roll, pitch, yaw]``.
    """
    return [-rx, ry, rz]


def root_frame_quat_xyzw(rx, ry, rz):
    """Root bone local Euler → blender-frame orientation quaternion (XYZW)."""
    roll_b, pitch_b, yaw_b = root_frame_euler_rpy(rx, ry, rz)
    return euler_to_quaternion_xyzw(yaw_b, pitch_b, roll_b)


class DataRecorder:
    """Deterministic recorder for the Open Duck Mini rig.

    Args:
        armature_name: name of the armature object (default ``"Armature"``).
        fps: sample rate written into the authoring JSON (default 50).
        ground_z: world Z of the ground plane for contact detection.
        contact_threshold: contact band above the ground (metres).
        contacts_valid: when False, every frame's contacts are forced to the
            explicit invalid sentinel ``[0, 0]`` and ``FootContactValid=false``
            is stamped into the episode (D3), rather than emitting bogus values.
    """

    def __init__(
        self,
        armature_name: str = "Armature",
        fps: int = 50,
        ground_z: float = 0.0,
        contact_threshold: float = contacts_mod.DEFAULT_CONTACT_THRESHOLD_M,
        contacts_valid: bool = True,
    ) -> None:
        if bpy is None:  # pragma: no cover - only hit outside Blender
            raise RuntimeError(
                "DataRecorder requires Blender (bpy). The bpy-free logic lives in "
                "transform_table / contacts / export and is tested separately."
            )
        self.armature_name = armature_name
        self.fps = int(fps)
        self.target_frame_time = 1.0 / self.fps
        self.ground_z = float(ground_z)
        self.contact_threshold = float(contact_threshold)
        self.contacts_valid = bool(contacts_valid)
        self.obj = bpy.data.objects[armature_name]
        self.pose = self.obj.pose

    # --- frame-mapping helpers (kept from upstream, unchanged behaviour) ------
    def blender_frame_to_robot_frame(self, x: float, y: float, z: float):
        return [-y, x, z]

    def euler_to_quaternion(self, yaw, pitch, roll):
        return euler_to_quaternion_xyzw(yaw, pitch, roll)

    def _root_local_euler_rpy(self):
        """Read the ``root`` bone's local rotation as ``(rx, ry, rz)`` Euler,
        independent of its ``rotation_mode``.

        The rig's ``root`` bone is in QUATERNION mode, where ``rotation_euler``
        is a stale, always-identity channel — reading it silently exports an
        identity root orientation and corrupts any motion that turns the body.
        Branch on the actual mode and convert to the same XYZ Euler components
        the downstream axis remap expects.
        """
        pb = self.pose.bones["root"]
        mode = pb.rotation_mode
        if mode == "QUATERNION":
            e = pb.rotation_quaternion.to_euler("XYZ")
            return (e.x, e.y, e.z)
        if mode == "AXIS_ANGLE":
            import mathutils  # type: ignore

            aa = pb.rotation_axis_angle  # (angle, x, y, z)
            e = mathutils.Quaternion(aa[1:4], aa[0]).to_euler("XYZ")
            return (e.x, e.y, e.z)
        # Any Euler mode: components are per-axis angles (order affects only
        # composition, which the component-wise remap below already assumes).
        e = pb.rotation_euler
        return (e[0], e[1], e[2])

    def get_root_position(self):
        x_root_frame, y_root_frame, z_root_frame = self.pose.bones["root"].location
        x_blender_frame = -x_root_frame
        y_blender_frame = z_root_frame
        z_blender_frame = y_root_frame
        return self.blender_frame_to_robot_frame(
            x_blender_frame, y_blender_frame, z_blender_frame
        )

    def get_root_orientation(self, return_quat=True):
        rx, ry, rz = self._root_local_euler_rpy()
        if not return_quat:
            return root_frame_euler_rpy(rx, ry, rz)
        return root_frame_quat_xyzw(rx, ry, rz)

    # --- joint reading (D2 + D11 fixed via the transform table) ---------------
    def get_bone_eulers(self) -> Dict[str, Sequence[float]]:
        """Read ``rotation_euler`` for every bone the transform table needs."""
        out: Dict[str, Sequence[float]] = {}
        for bone in REQUIRED_BONES:
            e = self.pose.bones[bone].rotation_euler
            out[bone] = (e[0], e[1], e[2])
        return out

    def get_joints_positions(self) -> List[float]:
        """16 canonical joint angles (rad) — antenna order & offsets corrected."""
        return joints_from_bone_eulers(self.get_bone_eulers())

    # --- toe / foot geometry --------------------------------------------------
    def get_toe_position(self, side: str):
        matrix_world = self.obj.matrix_world
        toe_matrix_world = matrix_world @ self.pose.bones[f"toe.{side[0]}"].matrix
        root_matrix_world = matrix_world @ self.pose.bones["root"].matrix
        return toe_matrix_world.translation - root_matrix_world.translation

    def get_toe_world_z(self, side: str) -> float:
        """World-space Z of the toe bone (for contact detection, D3)."""
        matrix_world = self.obj.matrix_world
        toe_matrix_world = matrix_world @ self.pose.bones[f"toe.{side[0]}"].matrix
        return float(toe_matrix_world.translation.z)

    def compute_frame_contacts(self) -> List[int]:
        """Per-frame ``[left, right]`` contacts (D3), honouring contacts_valid."""
        if not self.contacts_valid:
            return contacts_mod.invalid_contacts()
        return contacts_mod.compute_foot_contacts(
            self.get_toe_world_z("left"),
            self.get_toe_world_z("right"),
            ground_z=self.ground_z,
            threshold=self.contact_threshold,
        )

    @staticmethod
    def _velocity(prev: Sequence[float], curr: Sequence[float], dt: float) -> List[float]:
        return [(c - p) / dt for c, p in zip(curr, prev)]

    def _sample(self) -> Dict[str, object]:
        """Capture every quantity for the current frame (no stepping here)."""
        return {
            "root_position": list(self.get_root_position()),
            "root_orientation_quat": self.get_root_orientation().tolist(),
            "root_orientation_euler": list(self.get_root_orientation(return_quat=False)),
            "joints_positions": self.get_joints_positions(),
            "left_toe_pos": list(self.get_toe_position("left")),
            "right_toe_pos": list(self.get_toe_position("right")),
            "foot_contacts": self.compute_frame_contacts(),
        }

    def record(self) -> Dict[str, object]:
        """Deterministically record ``frame_start..frame_end`` (D4).

        Steps the timeline explicitly with ``frame_set`` (one sample per frame,
        no wall clock), then returns a fully-populated episode dict ready for
        :func:`open_duck_anim_blender.export.export_and_compile`.
        """
        scene = bpy.context.scene
        first, last = int(scene.frame_start), int(scene.frame_end)
        if last < first:
            raise ValueError("frame_end (%d) < frame_start (%d)" % (last, first))

        episode = export_mod.new_episode(fps=self.fps, contacts_valid=self.contacts_valid)
        dt = self.target_frame_time

        # Establish the previous-state baseline AT frame_start so the first
        # frame's finite-difference velocities are exactly zero and reproducible.
        scene.frame_set(first)
        bpy.context.view_layer.update()
        prev = self._sample()

        for f in range(first, last + 1):
            scene.frame_set(f)
            bpy.context.view_layer.update()
            cur = self._sample()

            frame = export_mod.assemble_frame(
                root_position=cur["root_position"],
                root_quaternion=cur["root_orientation_quat"],
                joint_positions=cur["joints_positions"],
                left_toe_pos=cur["left_toe_pos"],
                right_toe_pos=cur["right_toe_pos"],
                world_linear_vel=self._velocity(
                    prev["root_position"], cur["root_position"], dt
                ),
                world_angular_vel=self._velocity(
                    prev["root_orientation_euler"], cur["root_orientation_euler"], dt
                ),
                joint_velocities=self._velocity(
                    prev["joints_positions"], cur["joints_positions"], dt
                ),
                left_toe_vel=self._velocity(
                    prev["left_toe_pos"], cur["left_toe_pos"], dt
                ),
                right_toe_vel=self._velocity(
                    prev["right_toe_pos"], cur["right_toe_pos"], dt
                ),
                foot_contacts=cur["foot_contacts"],
            )
            episode["Frames"].append(frame)
            prev = cur

        return episode
