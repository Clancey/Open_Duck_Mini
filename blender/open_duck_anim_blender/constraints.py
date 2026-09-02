"""Apply Limit Rotation bone constraints mirroring the MJCF ``jnt_range``.

Blender shim (imports :mod:`bpy` guarded). The range→Euler algebra is in the
bpy-free :mod:`.jnt_range` module and is reused here.

Run this as a script against the rig (Phase 2 task d). It is **idempotent**: a
uniquely-named ``Limit Rotation`` constraint is created per constrained bone the
first time and updated in place on subsequent runs, so re-running never stacks
duplicate constraints. This matters because we may not be able to save a modified
``.blend`` — the author re-applies the constraints on load.

Constraints are authored in the bone's *local* space so the limits match the
Euler components the recorder reads.
"""

from __future__ import annotations

from typing import List

from .jnt_range import constrained_joints, euler_limit_for_joint
from .transform_table import TRANSFORM_BY_JOINT

try:  # guarded so CI (no Blender) can import this module
    import bpy  # type: ignore
except Exception:  # pragma: no cover - exercised only inside Blender
    bpy = None  # type: ignore

# Prefix so our constraints are greppable and idempotently replaceable.
CONSTRAINT_PREFIX = "duckanim_limit_"


def _axis_flags(axis: int):
    """Return (use_x, use_y, use_z) with only ``axis`` enabled."""
    return (axis == 0, axis == 1, axis == 2)


def apply_limit_rotation_constraints(armature_name: str = "Armature") -> List[str]:
    """Create/update Limit Rotation constraints for all 14 constrained joints.

    Returns the list of bone names that received a constraint. Idempotent.
    """
    if bpy is None:  # pragma: no cover - only outside Blender
        raise RuntimeError("apply_limit_rotation_constraints requires Blender (bpy)")

    obj = bpy.data.objects[armature_name]
    pose = obj.pose
    touched: List[str] = []

    for joint_name in constrained_joints():
        bone_name = TRANSFORM_BY_JOINT[joint_name].bone
        axis, e_min, e_max = euler_limit_for_joint(joint_name)
        pbone = pose.bones[bone_name]

        cname = CONSTRAINT_PREFIX + joint_name
        con = pbone.constraints.get(cname)
        if con is None or con.type != "LIMIT_ROTATION":
            if con is not None:
                pbone.constraints.remove(con)
            con = pbone.constraints.new("LIMIT_ROTATION")
            con.name = cname

        use_x, use_y, use_z = _axis_flags(axis)
        con.use_limit_x, con.use_limit_y, con.use_limit_z = use_x, use_y, use_z
        # Only the active axis gets meaningful min/max; others stay 0 and unused.
        con.min_x = e_min if use_x else 0.0
        con.max_x = e_max if use_x else 0.0
        con.min_y = e_min if use_y else 0.0
        con.max_y = e_max if use_y else 0.0
        con.min_z = e_min if use_z else 0.0
        con.max_z = e_max if use_z else 0.0
        con.owner_space = "LOCAL"
        con.use_transform_limit = True

        touched.append(bone_name)

    return touched


def remove_limit_rotation_constraints(armature_name: str = "Armature") -> int:
    """Remove all constraints we added (by prefix). Returns the count removed."""
    if bpy is None:  # pragma: no cover
        raise RuntimeError("remove_limit_rotation_constraints requires Blender (bpy)")
    obj = bpy.data.objects[armature_name]
    removed = 0
    for pbone in obj.pose.bones:
        for con in list(pbone.constraints):
            if con.name.startswith(CONSTRAINT_PREFIX):
                pbone.constraints.remove(con)
                removed += 1
    return removed


if __name__ == "__main__":  # pragma: no cover - run inside Blender
    print("duckanim: applied constraints to", apply_limit_rotation_constraints())
