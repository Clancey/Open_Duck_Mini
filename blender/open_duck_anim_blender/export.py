"""59-float frame assembly, episode building, and the export→compile path.

**bpy-free.** The recorder shim gathers per-frame quantities from Blender and
passes plain lists to :func:`assemble_frame`; everything here is testable without
Blender. The 59-float layout is authoritative per plan Appendix B and must stay
byte-for-byte compatible with the existing training source of truth (that schema
is *unchanged* — this is the training single source of truth).

Frame layout (Appendix B), inclusive byte ranges:

    0:3   root_position          (3)
    3:7   root_quaternion XYZW   (4)
    7:23  joint_positions        (16, JOINT_ORDER_16, rad)
    23:26 left_toe_pos           (3)
    26:29 right_toe_pos          (3)
    29:32 world_linear_vel       (3)
    32:35 world_angular_vel      (3)
    35:51 joint_velocities       (16)
    51:54 left_toe_vel           (3)
    54:57 right_toe_vel          (3)
    57:59 foot_contacts          (2)                       total = 59

Export path (plan §7 Phase 2 task f): write the 59-float authoring JSON (the
training source of truth), then invoke :mod:`open_duck_anim.compiler` to produce
the ``.duckanim``. We **reuse** the compiler; nothing here reimplements it.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Sequence

from open_duck_anim import compiler

FRAME_SIZE_59 = 59

# Segment lengths in order, used to validate assembly.
_SEGMENTS = (
    ("root_position", 3),
    ("root_quaternion", 4),
    ("joint_positions", 16),
    ("left_toe_pos", 3),
    ("right_toe_pos", 3),
    ("world_linear_vel", 3),
    ("world_angular_vel", 3),
    ("joint_velocities", 16),
    ("left_toe_vel", 3),
    ("right_toe_vel", 3),
    ("foot_contacts", 2),
)


def assemble_frame(
    root_position: Sequence[float],
    root_quaternion: Sequence[float],
    joint_positions: Sequence[float],
    left_toe_pos: Sequence[float],
    right_toe_pos: Sequence[float],
    world_linear_vel: Sequence[float],
    world_angular_vel: Sequence[float],
    joint_velocities: Sequence[float],
    left_toe_vel: Sequence[float],
    right_toe_vel: Sequence[float],
    foot_contacts: Sequence[float],
) -> List[float]:
    """Concatenate the 11 segments into one 59-float frame (Appendix B order).

    Validates every segment length and the total, so a malformed frame fails
    loudly at record time instead of producing a subtly-wrong clip.
    """
    parts = [
        root_position,
        root_quaternion,
        joint_positions,
        left_toe_pos,
        right_toe_pos,
        world_linear_vel,
        world_angular_vel,
        joint_velocities,
        left_toe_vel,
        right_toe_vel,
        foot_contacts,
    ]
    frame: List[float] = []
    for (name, expected), seg in zip(_SEGMENTS, parts):
        seg = list(seg)
        if len(seg) != expected:
            raise ValueError(
                "segment %r has length %d, expected %d" % (name, len(seg), expected)
            )
        frame.extend(float(x) for x in seg)
    if len(frame) != FRAME_SIZE_59:
        raise ValueError("assembled frame is %d floats, expected 59" % len(frame))
    return frame


def new_episode(fps: int = 50, contacts_valid: bool = True) -> Dict[str, Any]:
    """Return an empty episode dict matching the upstream authoring schema.

    Adds one new top-level key beyond the upstream schema: ``FootContactValid``
    (D3) — an explicit marker so a downstream consumer knows whether the
    per-frame contacts are meaningful. Every other key is unchanged.
    """
    return {
        "LoopMode": "Wrap",
        "FPS": int(fps),
        "EnableCycleOffsetPosition": True,
        "EnableCycleOffsetRotation": False,
        "FootContactValid": bool(contacts_valid),  # D3 explicit marker
        "Joints": [],
        "Vel_x": [],
        "Vel_y": [],
        "Yaw": [],
        "Placo": [],
        "Frame_offset": [],
        "Frame_size": [],
        "Frames": [],
        "MotionWeight": 1,
    }


def write_source_json(episode: Dict[str, Any], path: str) -> str:
    """Write the 59-float authoring JSON (unchanged schema). Returns ``path``."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(episode, f, indent=4)
    return path


def export_and_compile(
    episode: Dict[str, Any],
    meta: Dict[str, Any],
    source_path: str,
    duckanim_path: str,
    frame_range: Sequence[int] = None,  # type: ignore[assignment]
) -> Dict[str, str]:
    """Write the 59-float JSON then compile a ``.duckanim`` from it.

    Reuses :func:`open_duck_anim.compiler.compile_file` — no compiler logic is
    duplicated. ``meta`` should come from
    :meth:`open_duck_anim_blender.metadata.ClipMetadata.to_compiler_meta`.

    Returns ``{"source_path", "duckanim_path", "source_sha256"}``.
    """
    write_source_json(episode, source_path)
    source_sha256 = compiler.compile_file(
        source_path, meta, duckanim_path, frame_range=frame_range
    )
    return {
        "source_path": source_path,
        "duckanim_path": duckanim_path,
        "source_sha256": source_sha256,
    }
