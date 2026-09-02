"""MJCF ``jnt_range`` table and the derived Blender Limit-Rotation spec.

The upstream rig lets an author pose joints outside the robot's physical range;
those poses then export to clips that the policy has never seen. Phase 2 mirrors
the MJCF ``jnt_range`` into **Limit Rotation** bone constraints so out-of-range
poses are literally unauthorable.

**bpy-free by design** — this module only encodes the numbers and does the
range-to-Euler algebra; :mod:`open_duck_anim_blender.constraints` applies them to
the rig. That keeps the (testable) arithmetic off Blender.

Source: ``Open_Duck_Playground/playground/open_duck_mini_v2/xmls/
open_duck_mini_v2.xml`` (the MJCF ``<joint ... range="lo hi">`` attributes).
Antennas have **no** MJCF joint (they are open-loop PWM hobby servos, not on the
Feetech bus), so they get no ``jnt_range`` constraint here — see the note below.

Range -> bone Euler limit.  A joint angle is ``joint = sign*euler + zero_offset``
(:mod:`.transform_table`), so a jnt_range ``[lo, hi]`` on the *joint* maps to a
limit on the *bone Euler component*:

    e = (j - zero_offset) / sign

For ``sign = +1`` this is ``[lo - zero_offset, hi - zero_offset]``; for a
hypothetical ``sign = -1`` the endpoints also swap to stay ordered. Doing it via
the transform table means the baked knee/ankle offset (D11) is automatically and
correctly folded into the constraint bounds — one source of truth.
"""

from __future__ import annotations

from typing import Dict, Tuple

from .transform_table import TRANSFORM_BY_JOINT

# --- MJCF jnt_range, radians, verbatim from open_duck_mini_v2.xml -------------
# 14 DOF (no antennas). Keyed by canonical joint name.
JNT_RANGE: Dict[str, Tuple[float, float]] = {
    "left_hip_yaw":    (-0.5235987755982979, 0.5235987755982997),
    "left_hip_roll":   (-0.4363323129985815, 0.43633231299858327),
    "left_hip_pitch":  (-1.2217304763960306, 0.5235987755982988),
    "left_knee":       (-1.5707963267948966, 1.5707963267948966),
    "left_ankle":      (-1.5707963267948957, 1.5707963267948974),
    "neck_pitch":      (-0.3490658503988437, 1.1344640137963364),
    "head_pitch":      (-0.7853981633974483, 0.7853981633974483),
    "head_yaw":        (-2.792526803190927, 2.792526803190927),
    "head_roll":       (-0.523598775598218, 0.5235987755983796),
    "right_hip_yaw":   (-0.523598775598297, 0.5235987755983006),
    "right_hip_roll":  (-0.4363323129985797, 0.43633231299858505),
    "right_hip_pitch": (-0.5235987755982988, 1.2217304763960306),
    "right_knee":      (-1.5707963267948966, 1.5707963267948966),
    "right_ankle":     (-1.5707963267948957, 1.5707963267948974),
}
# (Antennas intentionally absent — no MJCF joint / no bus range; see module doc.)


def euler_limit_for_joint(joint_name: str) -> Tuple[int, float, float]:
    """Return ``(axis, min_euler, max_euler)`` for a joint's Limit Rotation.

    Converts the MJCF joint ``jnt_range`` into a limit on the bone's Euler
    component using that joint's calibrated transform (sign + zero_offset), so
    the constraint bounds already account for the baked knee/ankle offset (D11).

    Raises ``KeyError`` if the joint has no ``jnt_range`` (e.g. the antennas).
    """
    if joint_name not in JNT_RANGE:
        raise KeyError("no jnt_range for joint %r" % joint_name)
    t = TRANSFORM_BY_JOINT[joint_name]
    lo, hi = JNT_RANGE[joint_name]
    e_a = (lo - t.zero_offset) / t.sign
    e_b = (hi - t.zero_offset) / t.sign
    e_min, e_max = (e_a, e_b) if e_a <= e_b else (e_b, e_a)
    return t.axis, e_min, e_max


def constrained_joints() -> Tuple[str, ...]:
    """The joints that receive a Limit Rotation constraint (the 14 DOF)."""
    return tuple(JNT_RANGE.keys())
