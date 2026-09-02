"""Clip authoring metadata and the head-envelope author warning — bpy-free.

Two concerns, both kept off Blender so they are testable:

1. :class:`ClipMetadata` — the author-set fields exposed by the Blender panel
   (``name``, ``layer_mask``, ``blend_in_s`` / ``blend_out_s``, ``loop_mode``,
   ``requires_mode``, priority, show-blend times, antenna calibration, and the
   D3 ``contacts_valid`` toggle). :meth:`ClipMetadata.to_compiler_meta` produces
   exactly the ``meta`` dict :func:`open_duck_anim.compiler.compile_to_dict`
   expects, so the panel never has to know the compiler's key names.

2. :func:`head_envelope_warnings` — at authoring time, flag keyframes whose head
   channels exceed the *measured-safe* head range (:mod:`open_duck_anim.envelope`
   ``DEFLECTION_LIMITS``). Applying the "unauthorable" principle to the newest
   (D13) constraint: a **warning**, not a hard block — the safe envelope is a
   property of the *current* ONNX checkpoint and is expected to change after the
   Phase 5 retrain, so blocking authoring on it would be too aggressive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from open_duck_anim.clip import LAYER_MASKS, LOOP_MODES, REQUIRES_MODES
from open_duck_anim.envelope import DEFLECTION_LIMITS, HEAD_CHANNELS
from open_duck_anim.joint_order import (
    HEAD_PITCH_16,
    HEAD_ROLL_16,
    HEAD_YAW_16,
    NECK_PITCH_16,
)

# Default antenna calibration (plan §5.2). Signs: LEFT=+1, RIGHT=-1.
DEFAULT_ANTENNA_CALIBRATION: Dict[str, Dict[str, float]] = {
    "left": {"sign": 1, "rad_min": -0.6, "rad_max": 0.6},
    "right": {"sign": -1, "rad_min": -0.6, "rad_max": 0.6},
}

# Head channel -> its index in the 16-joint frame, in HEAD_CHANNELS order.
_HEAD_INDEX_16: Tuple[int, int, int, int] = (
    NECK_PITCH_16,
    HEAD_PITCH_16,
    HEAD_YAW_16,
    HEAD_ROLL_16,
)


class MetadataError(ValueError):
    """Raised when authoring metadata is invalid (fails loudly)."""


@dataclass
class ClipMetadata:
    """Author-set clip metadata (mirrors the Blender panel and §5 schema)."""

    name: str = "untitled_clip"
    layer_mask: str = "head"
    loop_mode: str = "wrap"
    requires_mode: str = "any"
    priority: int = 10
    blend_in_s: float = 0.35
    blend_out_s: float = 0.35
    show_blend_in_s: float = 0.1
    show_blend_out_s: float = 0.1
    source_blend: str = "open-duck-mini.blend"
    # D3: when False the recorder writes [0,0] contacts + FootContactValid=false.
    contacts_valid: bool = True
    antenna_calibration: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "left": dict(DEFAULT_ANTENNA_CALIBRATION["left"]),
            "right": dict(DEFAULT_ANTENNA_CALIBRATION["right"]),
        }
    )
    # Optional show tracks / discrete events (per §5.2).
    eyes: List[int] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)

    def validate(self) -> None:
        """Author-time sanity checks (the compiler re-validates on compile)."""
        if not self.name:
            raise MetadataError("clip name must be non-empty")
        if self.layer_mask not in LAYER_MASKS:
            raise MetadataError(
                "layer_mask must be one of %r, got %r" % (LAYER_MASKS, self.layer_mask)
            )
        if self.loop_mode not in LOOP_MODES:
            raise MetadataError(
                "loop_mode must be one of %r, got %r" % (LOOP_MODES, self.loop_mode)
            )
        if self.requires_mode not in REQUIRES_MODES:
            raise MetadataError(
                "requires_mode must be one of %r, got %r"
                % (REQUIRES_MODES, self.requires_mode)
            )
        for v, n in (
            (self.blend_in_s, "blend_in_s"),
            (self.blend_out_s, "blend_out_s"),
            (self.show_blend_in_s, "show_blend_in_s"),
            (self.show_blend_out_s, "show_blend_out_s"),
        ):
            if v < 0:
                raise MetadataError("%s must be >= 0, got %r" % (n, v))

    def to_compiler_meta(self) -> Dict[str, Any]:
        """Return the ``meta`` dict for :func:`open_duck_anim.compiler`."""
        self.validate()
        return {
            "name": self.name,
            "layer_mask": self.layer_mask,
            "loop_mode": self.loop_mode,
            "requires_mode": self.requires_mode,
            "priority": int(self.priority),
            "blend_in_s": float(self.blend_in_s),
            "blend_out_s": float(self.blend_out_s),
            "show_blend_in_s": float(self.show_blend_in_s),
            "show_blend_out_s": float(self.show_blend_out_s),
            "source_blend": self.source_blend,
            "antenna_calibration": self.antenna_calibration,
            "eyes": list(self.eyes),
            "events": list(self.events),
        }


@dataclass(frozen=True)
class EnvelopeWarning:
    """One keyframe/channel that exceeds the measured-safe head range."""

    frame_index: int
    channel: str
    value: float
    low: float
    high: float

    def message(self) -> str:
        return (
            "frame %d: head channel %r = %.4f rad exceeds measured-safe range "
            "[%.4f, %.4f] (envelope.DEFLECTION_LIMITS; advisory, re-derived after "
            "Phase 5 retrain)"
            % (self.frame_index, self.channel, self.value, self.low, self.high)
        )


def head_envelope_warnings(
    joints16_per_frame: List[List[float]],
    nominal_head: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> List[EnvelopeWarning]:
    """Warn on keyframes whose head deflection exceeds the safe envelope (D13).

    Args:
        joints16_per_frame: list of 16-float joint vectors (JOINT_ORDER_16).
        nominal_head: the clip's neutral head pose (deflection is measured from
            here); defaults to all-zero, matching ``transform.NOMINAL_HEAD_POSE``.

    Returns a list of :class:`EnvelopeWarning`. Empty means every keyframe is
    within the measured-safe head range. This is advisory only — the caller
    (panel/CLI) surfaces it as a warning, never a hard block.
    """
    warnings: List[EnvelopeWarning] = []
    for fi, frame in enumerate(joints16_per_frame):
        for ci, channel in enumerate(HEAD_CHANNELS):
            deflection = float(frame[_HEAD_INDEX_16[ci]]) - float(nominal_head[ci])
            low, high = DEFLECTION_LIMITS[channel]
            if deflection < low or deflection > high:
                warnings.append(
                    EnvelopeWarning(fi, channel, deflection, low, high)
                )
    return warnings
