"""One-way compiler: 59-float authoring JSON → ``.duckanim`` (plan §4.1-C, §5).

This is a **projection**, never a round-trip: it drops root pose/quaternion, toe
positions, velocities and contacts, keeping only the head/antenna/show subset the
runtime needs (plan §5.1). There is deliberately **no reverse conversion**.

Determinism (plan §4.1-C, §5.2, Phase 1 acceptance): identical input ⇒
byte-identical output. Achieved by (1) no timestamps or environment data, (2)
``json.dumps(..., sort_keys=True, separators=(",", ":"))`` for stable key order
and separators, and (3) converting every numeric value to a Python ``float``
whose ``repr`` is the platform-independent shortest round-trip — so the same
double serialises to the same bytes everywhere.

Provenance (plan §5.2): the compiler stamps the sha256 of the **source JSON
content**, the source ``.blend`` name, the frame range, and the compiler version.

59-float frame layout (plan Appendix B): ``joint_positions`` occupy inclusive
bytes 7..22 (Python slice ``[7:23]``, 16 joints, radians); the antenna channels
are joint indices 9 (left) and 10 (right), i.e. frame ``[16]`` and ``[17]``.
"""

from typing import Any, Dict, List, Optional, Sequence
import hashlib
import json

import numpy as np

from . import clip as clip_mod
from .clip import AntennaCalibration, validate_clip_dict
from .joint_order import JOINT_ORDER_16, N_JOINTS_16
from .version import __version__

COMPILER_VERSION = "open_duck_anim %s" % __version__

# 59-float authoring frame layout (plan Appendix B).
FRAME_SIZE_59 = 59
JOINT_SLICE_59 = slice(7, 23)          # 16 joint positions, radians
LEFT_ANTENNA_FRAME_IDX = 7 + 9         # canonical left_antenna (index 9 of 16)
RIGHT_ANTENNA_FRAME_IDX = 7 + 10       # canonical right_antenna (index 10 of 16)

# Required clip-metadata keys (from the Blender clip-metadata panel, Phase 2).
_REQUIRED_META = (
    "name", "loop_mode", "blend_in_s", "blend_out_s",
    "show_blend_in_s", "show_blend_out_s", "layer_mask", "priority",
    "requires_mode", "antenna_calibration", "source_blend",
)


class CompileError(ValueError):
    """Raised when compilation input is malformed (fails loudly, plan §4.1-C)."""


def _canonical_json_bytes(obj: Dict[str, Any]) -> bytes:
    """Serialise deterministically to bytes (plan §4.1-C / §5.2)."""
    text = json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return text.encode("utf-8")


def _to_py_floats(arr: np.ndarray) -> List[float]:
    """Convert to a list of Python floats (deterministic shortest-repr)."""
    return [float(x) for x in np.asarray(arr, dtype=np.float64).tolist()]


def compile_to_dict(
    source_json_text: str,
    meta: Dict[str, Any],
    frame_range: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Compile source JSON text + clip metadata into a ``.duckanim`` dict.

    Args:
        source_json_text: the **raw text** of the 59-float reference JSON (the
            single source of truth). Hashed verbatim for provenance.
        meta: clip authoring metadata (see ``_REQUIRED_META``) plus optional
            ``eyes`` (per-frame list), ``events`` (list of ``{frame,type,value}``),
            ``fps`` (overrides source ``FPS``), and validation controls
            ``on_channel_violation`` / ``nominal_pose_16``.
        frame_range: optional ``[start, end]`` **1-based inclusive** range
            (Blender convention). Defaults to the full set of source frames.

    Returns a validated ``.duckanim`` dict. Use :func:`compile_to_json_bytes` for
    the deterministic on-disk form.
    """
    for k in _REQUIRED_META:
        if k not in meta:
            raise CompileError("meta missing required key: %r" % k)

    try:
        source = json.loads(source_json_text)
    except json.JSONDecodeError as exc:
        raise CompileError("source_json_text is not valid JSON: %s" % exc)

    if "Frames" not in source:
        raise CompileError("source JSON missing 'Frames' (plan Appendix B)")
    all_frames = source["Frames"]
    n_source = len(all_frames)
    if n_source == 0:
        raise CompileError("source JSON has zero frames")

    fps = int(meta.get("fps", source.get("FPS", 50)))
    if fps <= 0:
        raise CompileError("fps must be positive, got %r" % fps)

    # Resolve 1-based inclusive frame range → 0-based python indices.
    if frame_range is None:
        start1, end1 = 1, n_source
    else:
        if len(frame_range) != 2:
            raise CompileError("frame_range must be [start, end]")
        start1, end1 = int(frame_range[0]), int(frame_range[1])
    if not (1 <= start1 <= end1 <= n_source):
        raise CompileError(
            "frame_range [%d, %d] out of bounds [1, %d]" % (start1, end1, n_source)
        )
    selected = all_frames[start1 - 1:end1]
    frame_count = len(selected)

    # Extract the 16 joint positions per frame (radians).
    joints = np.empty((frame_count, N_JOINTS_16), dtype=np.float64)
    for i, fr in enumerate(selected):
        if len(fr) != FRAME_SIZE_59:
            raise CompileError(
                "frame %d has %d floats, expected %d (plan Appendix B)"
                % (start1 + i, len(fr), FRAME_SIZE_59)
            )
        joints[i, :] = fr[JOINT_SLICE_59]

    # Antenna radians → normalised [-1,1] per side (plan §5.2 / Appendix A).
    cal_in = meta["antenna_calibration"]
    for side in ("left", "right"):
        if side not in cal_in:
            raise CompileError("meta.antenna_calibration missing side %r" % side)
    cal_left = AntennaCalibration(
        sign=int(cal_in["left"]["sign"]),
        rad_min=float(cal_in["left"]["rad_min"]),
        rad_max=float(cal_in["left"]["rad_max"]),
    )
    cal_right = AntennaCalibration(
        sign=int(cal_in["right"]["sign"]),
        rad_min=float(cal_in["right"]["rad_min"]),
        rad_max=float(cal_in["right"]["rad_max"]),
    )
    left_rad = joints[:, 9]
    right_rad = joints[:, 10]
    antenna_left = cal_left.to_normalized(left_rad)
    antenna_right = cal_right.to_normalized(right_rad)

    duration_s = frame_count / fps

    eyes = list(meta.get("eyes", []))
    events = list(meta.get("events", []))

    source_sha256 = hashlib.sha256(source_json_text.encode("utf-8")).hexdigest()

    out: Dict[str, Any] = {
        "format": clip_mod.FORMAT_TAG,
        "version": clip_mod.FORMAT_VERSION,
        "name": str(meta["name"]),
        "fps": fps,
        "loop_mode": str(meta["loop_mode"]),
        "frame_count": frame_count,
        "duration_s": float(duration_s),
        "blend_in_s": float(meta["blend_in_s"]),
        "blend_out_s": float(meta["blend_out_s"]),
        "show_blend_in_s": float(meta["show_blend_in_s"]),
        "show_blend_out_s": float(meta["show_blend_out_s"]),
        "layer_mask": str(meta["layer_mask"]),
        "priority": int(meta["priority"]),
        "requires_mode": str(meta["requires_mode"]),
        "provenance": {
            "source_sha256": source_sha256,
            "source_blend": str(meta["source_blend"]),
            "source_frame_range": [start1, end1],
            "compiler_version": COMPILER_VERSION,
        },
        "joints": {
            "order": list(JOINT_ORDER_16),
            "frames": [_to_py_floats(row) for row in joints],
        },
        "show_functions": {
            "antenna_left": _to_py_floats(antenna_left),
            "antenna_right": _to_py_floats(antenna_right),
            "eyes": [int(e) for e in eyes],
            "events": [
                {"frame": int(ev["frame"]), "type": str(ev["type"]), "value": str(ev["value"])}
                for ev in events
            ],
        },
        "antenna_calibration": {
            "left": {"sign": cal_left.sign, "rad_min": cal_left.rad_min, "rad_max": cal_left.rad_max},
            "right": {"sign": cal_right.sign, "rad_min": cal_right.rad_min, "rad_max": cal_right.rad_max},
        },
    }

    # Run all validations; fail loudly with actionable messages (plan §4.1-C).
    validate_clip_dict(
        out,
        on_channel_violation=meta.get("on_channel_violation", "error"),
        nominal_pose_16=meta.get("nominal_pose_16", None),
    )
    return out


def compile_to_json_bytes(
    source_json_text: str,
    meta: Dict[str, Any],
    frame_range: Optional[Sequence[int]] = None,
) -> bytes:
    """Compile to the deterministic on-disk JSON bytes (plan §4.1-C)."""
    return _canonical_json_bytes(compile_to_dict(source_json_text, meta, frame_range))


def compile_file(
    source_path: str,
    meta: Dict[str, Any],
    out_path: str,
    frame_range: Optional[Sequence[int]] = None,
) -> str:
    """Compile a source JSON file to a ``.duckanim`` JSON file on disk.

    Returns the provenance ``source_sha256``.
    """
    with open(source_path, "r", encoding="utf-8") as f:
        source_json_text = f.read()
    data = compile_to_dict(source_json_text, meta, frame_range)
    with open(out_path, "wb") as f:
        f.write(_canonical_json_bytes(data))
    return data["provenance"]["source_sha256"]
