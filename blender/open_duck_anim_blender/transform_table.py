"""Calibrated bone -> joint transform table (defects D2 and D11).

This is the **single, explicit, readable** declaration of how every one of the
16 canonical joints is derived from a Blender pose bone. It replaces the
scattered inline arithmetic in the upstream ``data_recording.py`` (the baked
``± np.deg2rad(10)`` knee/ankle offsets, defect **D11**) and the swapped antenna
indices (defect **D2**).

**bpy-free by design.** Nothing in this module imports :mod:`bpy`. The Blender
shim (:mod:`open_duck_anim_blender.recorder`) reads
``pose.bones[bone].rotation_euler`` into a plain ``{bone: (rx, ry, rz)}`` mapping
and hands it to :func:`joints_from_bone_eulers`, so every bit of the actual
maths is unit-testable on the CI machine without Blender >= 4.3.2.

Each row declares, for one canonical joint (``JOINT_ORDER_16`` order):

* ``bone``          — the Blender pose-bone name.
* ``axis``          — which Euler component carries the joint angle (0=X,1=Y,2=Z).
* ``sign``          — +1 or -1, the rotation direction convention.
* ``zero_offset``   — a constant added AFTER the sign, in radians. This is where
  the old baked knee/ankle calibration now lives, in the open, one value per row.

Forward transform (export):  ``joint = sign * euler[axis] + zero_offset``.
Inverse (round-trip):        ``euler[axis] = (joint - zero_offset) / sign``.

D2 (antenna swap): the upstream code wrote ``antenna.r`` to canonical index 9
(``left_antenna``) and ``antenna.l`` to index 10 (``right_antenna``). Here index
9 = ``antenna.l`` and index 10 = ``antenna.r``, matching ``JOINT_ORDER_16`` /
``poly_reference_motion.py``. **No sign inversion is applied** to the antennas:
the ``LEFT_SIGN=+1`` / ``RIGHT_SIGN=-1`` convention lives downstream in the
runtime PWM (``antennas.py``) and in the compiler's ``antenna_calibration`` — so
recording the raw bone angle in canonical order here, with ``sign=+1``, avoids a
double inversion (task requirement).

D11 (baked offsets): ``left_knee`` / ``right_knee`` carry ``zero_offset =
-DEG10`` and ``left_ankle`` / ``right_ankle`` carry ``zero_offset = +DEG10``,
reproducing the exact numeric behaviour of the upstream
``euler[0] - np.deg2rad(10)`` / ``euler[0] + np.deg2rad(10)`` — now declared, not
buried.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from open_duck_anim.joint_order import JOINT_ORDER_16, N_JOINTS_16

# Baked leg calibration constant (upstream ``np.deg2rad(10)``). Declared once.
DEG10: float = float(np.deg2rad(10.0))

# Euler component indices, named for readability in the table below.
X, Y, Z = 0, 1, 2


@dataclass(frozen=True)
class JointTransform:
    """One row: how a single canonical joint is read from a pose bone."""

    joint_name: str
    bone: str
    axis: int
    sign: float
    zero_offset: float

    def forward(self, euler: Sequence[float]) -> float:
        """``euler`` is the bone's ``(rx, ry, rz)``; return the joint angle."""
        return self.sign * float(euler[self.axis]) + self.zero_offset

    def inverse(self, joint_value: float) -> float:
        """Recover the bone Euler component on ``axis`` from a joint angle."""
        return (float(joint_value) - self.zero_offset) / self.sign


# --- The table, in canonical JOINT_ORDER_16 order (indices 0..15) -------------
# NOTE: kept in list form so the ordering is visually auditable against
# Appendix A. A post-construction assert (below) guarantees it stays in lockstep
# with ``JOINT_ORDER_16`` — no silent drift.
JOINT_TRANSFORMS: Tuple[JointTransform, ...] = (
    JointTransform("left_hip_yaw",    "hip_yaw_fk.l",   Y, 1.0,  0.0),      # 0
    JointTransform("left_hip_roll",   "hip_roll_fk.l",  Z, 1.0,  0.0),      # 1
    JointTransform("left_hip_pitch",  "hip_pitch_fk.l", X, 1.0,  0.0),      # 2
    JointTransform("left_knee",       "knee_fk.l",      X, 1.0, -DEG10),    # 3  (D11)
    JointTransform("left_ankle",      "ankle_fk.l",     X, 1.0,  DEG10),    # 4  (D11)
    JointTransform("neck_pitch",      "neck_pitch",     X, 1.0,  0.0),      # 5
    JointTransform("head_pitch",      "head_pitch",     X, 1.0,  0.0),      # 6
    JointTransform("head_yaw",        "head_yaw",       Z, 1.0,  0.0),      # 7
    JointTransform("head_roll",       "head_roll",      Z, 1.0,  0.0),      # 8
    JointTransform("left_antenna",    "antenna.l",      Z, 1.0,  0.0),      # 9  (D2)
    JointTransform("right_antenna",   "antenna.r",      Z, 1.0,  0.0),      # 10 (D2)
    JointTransform("right_hip_yaw",   "hip_yaw_fk.r",   Y, 1.0,  0.0),      # 11
    JointTransform("right_hip_roll",  "hip_roll_fk.r",  Z, 1.0,  0.0),      # 12
    JointTransform("right_hip_pitch", "hip_pitch_fk.r", X, 1.0,  0.0),      # 13
    JointTransform("right_knee",      "knee_fk.r",      X, 1.0, -DEG10),    # 14 (D11)
    JointTransform("right_ankle",     "ankle_fk.r",     X, 1.0,  DEG10),    # 15 (D11)
)

# Fail loudly at import time if the table ever drifts from the canonical order.
assert len(JOINT_TRANSFORMS) == N_JOINTS_16, "transform table must have 16 rows"
assert [t.joint_name for t in JOINT_TRANSFORMS] == list(JOINT_ORDER_16), (
    "JOINT_TRANSFORMS order must match open_duck_anim.JOINT_ORDER_16"
)

# Convenience lookups.
TRANSFORM_BY_JOINT: Dict[str, JointTransform] = {t.joint_name: t for t in JOINT_TRANSFORMS}
REQUIRED_BONES: Tuple[str, ...] = tuple(t.bone for t in JOINT_TRANSFORMS)


def joints_from_bone_eulers(euler_map: Dict[str, Sequence[float]]) -> List[float]:
    """Export the 16 canonical joint angles from a bone-Euler mapping.

    Args:
        euler_map: ``{bone_name: (rx, ry, rz)}`` for at least every bone in
            :data:`REQUIRED_BONES`. Extra keys are ignored.

    Returns a list of 16 floats in :data:`JOINT_ORDER_16` order (radians).

    Raises ``KeyError`` (with the missing bone name) if a required bone is
    absent — this surfaces a rig/name mismatch immediately instead of silently
    exporting a wrong vector.
    """
    out: List[float] = []
    for t in JOINT_TRANSFORMS:
        if t.bone not in euler_map:
            raise KeyError(
                "bone %r (joint %r) missing from euler_map" % (t.bone, t.joint_name)
            )
        out.append(t.forward(euler_map[t.bone]))
    return out


def bone_eulers_from_joints(joints16: Sequence[float]) -> Dict[str, float]:
    """Inverse of :func:`joints_from_bone_eulers` for round-trip testing.

    Returns ``{bone_name: euler_component_on_its_axis}`` — one scalar per bone,
    the value that, placed on that bone's declared ``axis``, reproduces the joint
    angle. Used by the D11 regression test to assert a clean round-trip.
    """
    if len(joints16) != N_JOINTS_16:
        raise ValueError("joints16 must have length 16, got %d" % len(joints16))
    return {t.bone: t.inverse(joints16[i]) for i, t in enumerate(JOINT_TRANSFORMS)}


def zero_pose_joints() -> List[float]:
    """The 16-joint vector exported when every bone is at Euler ``(0, 0, 0)``.

    Equals each row's ``zero_offset`` — i.e. zeros everywhere except the four
    baked knee/ankle offsets. This is the expected value the D11 regression test
    pins down (the rig's rest/zero pose).
    """
    return [t.zero_offset for t in JOINT_TRANSFORMS]
