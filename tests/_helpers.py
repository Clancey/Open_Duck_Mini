"""Shared test helpers: build synthetic 59-float sources and compiled clips."""

import json
from typing import Dict, List, Optional

import numpy as np

from open_duck_anim import compiler, clip as clipmod

# Static neutral leg values (constant across frames → passes leg-neutral check).
LEG_VALS = {2: -0.63, 3: 1.368, 4: -0.784, 13: 0.635, 14: 1.379, 15: -0.796}

DEFAULT_CAL = {
    "left": {"sign": 1, "rad_min": -0.6, "rad_max": 0.6},
    "right": {"sign": -1, "rad_min": -0.6, "rad_max": 0.6},
}


def make_source_text(
    n_frames: int = 20,
    head_yaw_end: float = 0.5,
    antenna_end_rad: float = 0.6,
    fps: int = 50,
    move_legs: bool = False,
    move_head: bool = True,
    head_const: Optional[float] = None,
    antenna_const_rad: Optional[float] = None,
) -> str:
    """Return the raw JSON text of a synthetic 59-float reference clip.

    If ``head_const`` / ``antenna_const_rad`` are given, that channel is held
    constant across all frames (useful for isolating blend-weight behaviour).
    """
    frames: List[List[float]] = []
    for i in range(n_frames):
        fr = [0.0] * 59
        jp = [0.0] * 16
        for k, v in LEG_VALS.items():
            jp[k] = v
        frac = 0.0 if n_frames <= 1 else i / (n_frames - 1)
        if head_const is not None:
            jp[7] = head_const
        elif move_head:
            jp[7] = head_yaw_end * frac  # head_yaw = joint index 7
        if move_legs:
            jp[3] = LEG_VALS[3] + 0.4 * frac  # left_knee moves → illegal for head mask
        if antenna_const_rad is not None:
            jp[9] = antenna_const_rad
            jp[10] = antenna_const_rad
        else:
            jp[9] = antenna_end_rad * frac       # left antenna rad
            jp[10] = antenna_end_rad * frac      # right antenna rad
        fr[7:23] = jp
        frames.append(fr)
    return json.dumps({"FPS": fps, "Frames": frames})


def make_meta(
    name: str = "clip",
    loop_mode: str = "once",
    blend_in_s: float = 0.1,
    blend_out_s: float = 0.1,
    show_blend_in_s: float = 0.05,
    show_blend_out_s: float = 0.05,
    layer_mask: str = "head",
    priority: int = 5,
    requires_mode: str = "any",
    events: Optional[List[Dict]] = None,
    eyes: Optional[List[int]] = None,
) -> Dict:
    meta = {
        "name": name,
        "loop_mode": loop_mode,
        "blend_in_s": blend_in_s,
        "blend_out_s": blend_out_s,
        "show_blend_in_s": show_blend_in_s,
        "show_blend_out_s": show_blend_out_s,
        "layer_mask": layer_mask,
        "priority": priority,
        "requires_mode": requires_mode,
        "source_blend": "test.blend",
        "antenna_calibration": DEFAULT_CAL,
    }
    if events is not None:
        meta["events"] = events
    if eyes is not None:
        meta["eyes"] = eyes
    return meta


def make_clip(source_text: Optional[str] = None, **meta_kwargs):
    """Compile a clip from a synthetic source and default/overridden metadata."""
    if source_text is None:
        source_text = make_source_text()
    src_specific = {}
    for k in ("n_frames", "head_yaw_end", "antenna_end_rad", "fps", "move_legs",
              "move_head", "head_const", "antenna_const_rad"):
        if k in meta_kwargs:
            src_specific[k] = meta_kwargs.pop(k)
    if src_specific:
        source_text = make_source_text(**src_specific)
    meta = make_meta(**meta_kwargs)
    d = compiler.compile_to_dict(source_text, meta)
    return clipmod.clip_from_dict(d)
