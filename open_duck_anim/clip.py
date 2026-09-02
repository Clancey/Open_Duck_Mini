""".duckanim clip loading, validation and the :class:`DuckAnimClip` dataclass.

Implements the plan §5 clip-format spec: strict schema validation, the §6.2
capability-matrix channel-legality check, and the §5.2 antenna precedence rule
(the runtime reads antenna values **only** from ``show_functions``, never from
the 16-joint array).

Containers (plan §5.2): both JSON (nested lists) and ``npz`` (arrays) are
supported. An ``npz`` stores the scalar metadata as a JSON blob under
``meta_json`` plus the large tracks as arrays; :func:`load_clip` dispatches on
extension.

Design choices where the plan is silent:

* "Neutral" leg/head channels (the capability check) are validated as
  **constant across frames** within ``channel_motion_tol`` — i.e. the animation
  does not *move* a channel it is not allowed to move. This is the unambiguous
  safety property ("legs held"); an optional ``nominal_pose_16`` additionally
  checks closeness to a supplied hold pose.
* Legs are never animation-movable in any Phase-1 mode (plan §6.2: dock=held,
  stand/walk=policy-owned), so a ``legs``/``full_body`` mask is rejected.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import json
import os
import warnings

import numpy as np

from . import joint_order as jo
from .joint_order import JOINT_ORDER_16, HEAD_SLICE_16, LEG_INDICES_16, ANTENNA_INDICES_16

# --- Enums (as frozensets; plan §5.2) -----------------------------------------
LOOP_MODES = ("wrap", "once", "clamp")
LAYER_MASKS = ("head", "antennas", "legs", "full_body")
REQUIRES_MODES = ("dock", "stand", "walk", "any")

FORMAT_TAG = "duckanim"
FORMAT_VERSION = 1

# Default numeric tolerances (documented choices; plan is silent on exact eps).
DURATION_TOL = 1e-6      # duration_s == frame_count / fps
CHANNEL_MOTION_TOL = 1e-4  # a "neutral" channel must not vary more than this
NEUTRAL_POSE_TOL = 0.12    # closeness to nominal hold, if a hold is supplied


class ClipValidationError(ValueError):
    """Raised when a ``.duckanim`` clip violates the plan §5 / §6.2 rules."""


@dataclass(frozen=True)
class DiscreteEvent:
    """A discrete, fire-once show event (plan §5.2 ``show_functions.events``).

    Deliberately carries no numeric interface, so it can never be fed into a
    joint rate limiter (plan §6.4: events are "never rate-limited as joint
    angles"). See :mod:`open_duck_anim.limits`.
    """

    frame: int
    type: str
    value: str


@dataclass(frozen=True)
class AntennaCalibration:
    """Per-side radians→normalised calibration (plan §5.2 / Appendix A)."""

    sign: int
    rad_min: float
    rad_max: float

    def to_normalized(self, rad: np.ndarray) -> np.ndarray:
        """Map radians → normalised ``[-1,1]`` (plan Appendix A formula).

        ``norm = clamp(sign * 2*(rad - centre)/(rad_max - rad_min), -1, 1)`` with
        ``centre = (rad_max + rad_min)/2``.
        """
        rad = np.asarray(rad, dtype=np.float64)
        span = self.rad_max - self.rad_min
        if span <= 0:
            raise ClipValidationError("antenna calibration rad_max must exceed rad_min")
        centre = 0.5 * (self.rad_max + self.rad_min)
        norm = self.sign * 2.0 * (rad - centre) / span
        return np.clip(norm, -1.0, 1.0)


@dataclass
class ShowOutput:
    """Runtime show-function tracks (plan §6.4 ``ShowOutput``).

    ``antenna_l`` / ``antenna_r`` are normalised ``[-1,1]`` — the **only**
    authoritative antenna source at runtime (plan §5.2 precedence rule).
    """

    antenna_l: np.ndarray
    antenna_r: np.ndarray
    eyes: np.ndarray
    events: List[DiscreteEvent] = field(default_factory=list)


@dataclass
class DuckAnimClip:
    """A validated ``.duckanim`` clip (plan §5.2).

    Antenna precedence rule (plan §5.2): the runtime reads antenna values **only**
    from ``show_functions`` (via :meth:`runtime_antennas`). To make it structurally
    impossible to read antennas from the joint array at runtime, the antenna
    columns (indices 9, 10) of :attr:`joints` are **zeroed at load** — the
    radians live only in the authoring/compile path, never in the runtime clip.
    There is intentionally no accessor that returns antenna values from
    ``joints``.
    """

    name: str
    fps: int
    loop_mode: str
    frame_count: int
    duration_s: float
    blend_in_s: float
    blend_out_s: float
    show_blend_in_s: float
    show_blend_out_s: float
    layer_mask: str
    priority: int
    requires_mode: str
    provenance: Dict[str, Any]
    joints: np.ndarray              # (frame_count, 16), radians, JOINT_ORDER_16
    show: ShowOutput
    antenna_calibration: Dict[str, AntennaCalibration]
    version: int = FORMAT_VERSION

    # --- runtime accessors ----------------------------------------------------
    @property
    def n_frames(self) -> int:
        return self.frame_count

    def head_block(self, frame_index: int) -> np.ndarray:
        """Return the 4 head channels (neck/head pitch/yaw/roll) for a frame."""
        return self.joints[frame_index, HEAD_SLICE_16].copy()

    def runtime_antennas(self, frame_index: int) -> Tuple[float, float]:
        """Return ``(antenna_left, antenna_right)`` normalised for a frame.

        This reads **only** from ``show_functions`` (plan §5.2 precedence rule).
        There is deliberately no method to read antennas from the joint array.
        """
        return (
            float(self.show.antenna_l[frame_index]),
            float(self.show.antenna_r[frame_index]),
        )


# --- validation helpers -------------------------------------------------------
def _require(d: Dict[str, Any], key: str) -> Any:
    if key not in d:
        raise ClipValidationError("missing required field: %r" % key)
    return d[key]


def _movable_blocks(layer_mask: str, requires_mode: str) -> set:
    """Blocks the animation is *permitted* to move (plan §6.2 capability matrix).

    Head is command/direct in every mode → always movable. Legs are movable in
    **exactly one** case: ``requires_mode == "dock"``. When docked/cradled the
    legs are not load-bearing, no policy is running (``DOCK_DEMO`` bypasses the
    policy), and there is no balance constraint, so animated leg motion is safe
    (plan §4.3, §6.2). In every other mode the legs are either held
    load-relieving (dock is the only place they *may* move, and standing/walking
    is not dock) or owned by the RL policy (``stand``/``walk``) — animating them
    there reintroduces the balance problem the whole architecture avoids, so
    legs stay immovable and any full-body clip is rejected.

    ``requires_mode == "any"`` must satisfy all three modes at once, so it can
    never move the legs (stand/walk forbid it). This is what makes a full-body
    clip authorable *only* as ``requires_mode="dock"``.
    """
    if requires_mode == "dock":
        return {"head", "legs"}
    return {"head"}


def _declared_blocks(layer_mask: str) -> set:
    return {
        "head": {"head"},
        "antennas": set(),
        "legs": {"legs"},
        "full_body": {"head", "legs"},
    }[layer_mask]


def _channel_is_neutral(
    frames: np.ndarray,
    indices: Sequence[int],
    nominal: Optional[np.ndarray],
    motion_tol: float,
    neutral_tol: float,
) -> Tuple[bool, str]:
    """Check a channel block is neutral (constant, and near nominal if given)."""
    block = frames[:, list(indices)]
    motion = float(np.max(np.ptp(block, axis=0))) if block.size else 0.0
    if motion > motion_tol:
        return False, "channel moves by %.4g rad (> %.4g)" % (motion, motion_tol)
    if nominal is not None:
        ref = np.asarray(nominal, dtype=np.float64)[list(indices)]
        dev = float(np.max(np.abs(block - ref))) if block.size else 0.0
        if dev > neutral_tol:
            return False, "channel deviates %.4g rad from nominal hold (> %.4g)" % (dev, neutral_tol)
    return True, ""


def validate_clip_dict(
    d: Dict[str, Any],
    on_channel_violation: str = "error",
    nominal_pose_16: Optional[np.ndarray] = None,
    channel_motion_tol: float = CHANNEL_MOTION_TOL,
    neutral_pose_tol: float = NEUTRAL_POSE_TOL,
) -> None:
    """Validate a parsed clip dict against the plan §5 / §6.2 rules.

    Raises :class:`ClipValidationError` on any violation (with an actionable
    message). ``on_channel_violation`` may be ``"error"`` (default) or ``"warn"``
    to downgrade channel-legality failures to warnings (plan §5.2: "rejected at
    compile time (or warned loudly)").
    """
    if on_channel_violation not in ("error", "warn"):
        raise ValueError("on_channel_violation must be 'error' or 'warn'")

    # --- format / version ---
    fmt = _require(d, "format")
    if fmt != FORMAT_TAG:
        raise ClipValidationError("format must be %r, got %r" % (FORMAT_TAG, fmt))
    ver = _require(d, "version")
    if ver != FORMAT_VERSION:
        raise ClipValidationError("version must be %d, got %r" % (FORMAT_VERSION, ver))

    name = _require(d, "name")
    if not isinstance(name, str) or not name:
        raise ClipValidationError("name must be a non-empty string")

    fps = _require(d, "fps")
    if not isinstance(fps, (int,)) or fps <= 0:
        raise ClipValidationError("fps must be a positive integer, got %r" % (fps,))

    frame_count = _require(d, "frame_count")
    if not isinstance(frame_count, int) or frame_count <= 0:
        raise ClipValidationError("frame_count must be a positive integer, got %r" % (frame_count,))

    duration_s = float(_require(d, "duration_s"))
    expected_duration = frame_count / fps
    if abs(duration_s - expected_duration) > DURATION_TOL:
        raise ClipValidationError(
            "duration_s (%r) != frame_count/fps (%r)" % (duration_s, expected_duration)
        )

    # --- enums ---
    loop_mode = _require(d, "loop_mode")
    if loop_mode not in LOOP_MODES:
        raise ClipValidationError("loop_mode must be one of %r, got %r" % (LOOP_MODES, loop_mode))
    layer_mask = _require(d, "layer_mask")
    if layer_mask not in LAYER_MASKS:
        raise ClipValidationError("layer_mask must be one of %r, got %r" % (LAYER_MASKS, layer_mask))
    requires_mode = _require(d, "requires_mode")
    if requires_mode not in REQUIRES_MODES:
        raise ClipValidationError(
            "requires_mode must be one of %r, got %r" % (REQUIRES_MODES, requires_mode)
        )

    # --- blend times ---
    blend_in_s = float(_require(d, "blend_in_s"))
    blend_out_s = float(_require(d, "blend_out_s"))
    show_blend_in_s = float(_require(d, "show_blend_in_s"))
    show_blend_out_s = float(_require(d, "show_blend_out_s"))
    for label, v in (
        ("blend_in_s", blend_in_s), ("blend_out_s", blend_out_s),
        ("show_blend_in_s", show_blend_in_s), ("show_blend_out_s", show_blend_out_s),
    ):
        if v < 0:
            raise ClipValidationError("%s must be >= 0, got %r" % (label, v))
    if blend_in_s + blend_out_s > duration_s + DURATION_TOL:
        raise ClipValidationError(
            "blend_in_s + blend_out_s (%r) must be <= duration_s (%r)"
            % (blend_in_s + blend_out_s, duration_s)
        )
    if show_blend_in_s + show_blend_out_s > duration_s + DURATION_TOL:
        raise ClipValidationError(
            "show_blend_in_s + show_blend_out_s (%r) must be <= duration_s (%r)"
            % (show_blend_in_s + show_blend_out_s, duration_s)
        )

    if not isinstance(_require(d, "priority"), int):
        raise ClipValidationError("priority must be an int")

    # --- provenance ---
    prov = _require(d, "provenance")
    for pk in ("source_sha256", "source_blend", "source_frame_range", "compiler_version"):
        if pk not in prov:
            raise ClipValidationError("provenance missing %r" % pk)

    # --- joints ---
    joints = _require(d, "joints")
    order = _require(joints, "order")
    if list(order) != JOINT_ORDER_16:
        raise ClipValidationError(
            "joints.order must equal canonical JOINT_ORDER_16 (plan Appendix A)"
        )
    frames = np.asarray(_require(joints, "frames"), dtype=np.float64)
    if frames.ndim != 2 or frames.shape[1] != jo.N_JOINTS_16:
        raise ClipValidationError(
            "joints.frames must be (frame_count, 16), got shape %r" % (frames.shape,)
        )
    if frames.shape[0] != frame_count:
        raise ClipValidationError(
            "joints.frames has %d rows but frame_count is %d" % (frames.shape[0], frame_count)
        )
    if not np.all(np.isfinite(frames)):
        raise ClipValidationError(
            "joints.frames contains non-finite values (NaN/Inf); a non-finite "
            "joint angle would propagate through the blend to an actuator "
            "command (plan §6.5 safety)"
        )

    # --- show_functions track lengths ---
    show = _require(d, "show_functions")
    a_left = np.asarray(_require(show, "antenna_left"), dtype=np.float64)
    a_right = np.asarray(_require(show, "antenna_right"), dtype=np.float64)
    if a_left.shape != (frame_count,):
        raise ClipValidationError(
            "show_functions.antenna_left length %r != frame_count %d" % (a_left.shape, frame_count)
        )
    if a_right.shape != (frame_count,):
        raise ClipValidationError(
            "show_functions.antenna_right length %r != frame_count %d" % (a_right.shape, frame_count)
        )
    if not (np.all(np.isfinite(a_left)) and np.all(np.isfinite(a_right))):
        raise ClipValidationError(
            "antenna tracks contain non-finite values (NaN/Inf); NaN silently "
            "passes range checks (NaN comparisons are always false) and would "
            "reach the antenna servo command (plan §6.5 safety)"
        )
    if np.any(np.abs(a_left) > 1.0 + 1e-9) or np.any(np.abs(a_right) > 1.0 + 1e-9):
        raise ClipValidationError("antenna tracks must be normalised within [-1, 1]")
    eyes = show.get("eyes", [])
    eyes_arr = np.asarray(eyes)
    if eyes_arr.size not in (0, frame_count):
        raise ClipValidationError(
            "show_functions.eyes length %d must be 0 or frame_count %d" % (eyes_arr.size, frame_count)
        )
    events = show.get("events", [])
    for ev in events:
        ef = ev.get("frame")
        if not isinstance(ef, int) or ef < 0 or ef >= frame_count:
            raise ClipValidationError(
                "event frame %r out of range [0, %d)" % (ef, frame_count)
            )
        if "type" not in ev or "value" not in ev:
            raise ClipValidationError("event missing 'type' or 'value': %r" % (ev,))

    # --- antenna_calibration ---
    cal = _require(d, "antenna_calibration")
    # Signs are HARDWARE CONSTANTS from antennas.py (plan Appendix A): LEFT=+1,
    # RIGHT=-1. Pin them exactly. A swapped sign compiles and loads cleanly but
    # inverts an antenna on hardware; this validator is the layer that must catch
    # the upstream Blender L/R swap defect (plan Appendix A, bug D2).
    _REQUIRED_SIGN = {"left": 1, "right": -1}
    for side in ("left", "right"):
        if side not in cal:
            raise ClipValidationError("antenna_calibration missing side %r" % side)
        for ck in ("sign", "rad_min", "rad_max"):
            if ck not in cal[side]:
                raise ClipValidationError("antenna_calibration.%s missing %r" % (side, ck))
        if cal[side]["sign"] != _REQUIRED_SIGN[side]:
            raise ClipValidationError(
                "antenna_calibration.%s.sign must be %+d (fixed hardware constant "
                "LEFT=+1/RIGHT=-1, plan Appendix A); got %r"
                % (side, _REQUIRED_SIGN[side], cal[side]["sign"])
            )

    # --- channel legality (plan §6.2 capability matrix) ---
    declared = _declared_blocks(layer_mask)
    movable = _movable_blocks(layer_mask, requires_mode)
    violations: List[str] = []
    if not declared.issubset(movable):
        illegal = declared - movable
        if "legs" in illegal:
            violations.append(
                "layer_mask %r moves the legs but requires_mode is %r; full-body "
                "(leg) animation is permitted ONLY with requires_mode=\"dock\". "
                "On the dock the legs are not load-bearing and no policy runs, so "
                "animating them is safe; in any/stand/walk the legs are held or "
                "owned by the RL policy and animated leg motion would reintroduce "
                "the balance failure the architecture avoids (plan §6.2). Re-author "
                "this clip as requires_mode=\"dock\", or use layer_mask=\"head\"."
                % (layer_mask, requires_mode)
            )
        else:
            violations.append(
                "layer_mask %r declares motion on %s but requires_mode %r does not "
                "permit it (plan §6.2: legs are held in dock and policy-owned in "
                "stand/walk)" % (layer_mask, sorted(illegal), requires_mode)
            )

    # Data check: any block NOT permitted-and-declared must be neutral in-data.
    allowed = declared & movable
    if "head" not in allowed:
        ok, why = _channel_is_neutral(
            frames, range(HEAD_SLICE_16.start, HEAD_SLICE_16.stop),
            nominal_pose_16, channel_motion_tol, neutral_pose_tol,
        )
        if not ok:
            violations.append("head block must be neutral for layer_mask %r: %s" % (layer_mask, why))
    if "legs" not in allowed:
        ok, why = _channel_is_neutral(
            frames, LEG_INDICES_16, nominal_pose_16, channel_motion_tol, neutral_pose_tol,
        )
        if not ok:
            violations.append("leg block must be neutral for layer_mask %r: %s" % (layer_mask, why))

    if violations:
        msg = "channel-legality violation(s): " + "; ".join(violations)
        if on_channel_violation == "error":
            raise ClipValidationError(msg)
        warnings.warn(msg, stacklevel=2)


def clip_from_dict(
    d: Dict[str, Any],
    validate: bool = True,
    **validate_kwargs: Any,
) -> DuckAnimClip:
    """Build a :class:`DuckAnimClip` from a parsed dict, validating by default."""
    if validate:
        validate_clip_dict(d, **validate_kwargs)

    frames = np.ascontiguousarray(np.asarray(d["joints"]["frames"], dtype=np.float64))
    # Antenna precedence (plan §5.2): zero the antenna joint columns (9, 10) in
    # the runtime clip so antenna radians cannot be read from the joint array.
    # The authoritative normalised values live only in ``show_functions``.
    frames[:, list(ANTENNA_INDICES_16)] = 0.0
    show_d = d["show_functions"]
    frame_count = int(d["frame_count"])
    eyes = np.asarray(show_d.get("eyes", []), dtype=np.int64)
    events = [
        DiscreteEvent(frame=int(ev["frame"]), type=str(ev["type"]), value=str(ev["value"]))
        for ev in show_d.get("events", [])
    ]
    events.sort(key=lambda e: e.frame)
    show = ShowOutput(
        antenna_l=np.ascontiguousarray(np.asarray(show_d["antenna_left"], dtype=np.float64)),
        antenna_r=np.ascontiguousarray(np.asarray(show_d["antenna_right"], dtype=np.float64)),
        eyes=eyes,
        events=events,
    )
    cal = {
        side: AntennaCalibration(
            sign=int(d["antenna_calibration"][side]["sign"]),
            rad_min=float(d["antenna_calibration"][side]["rad_min"]),
            rad_max=float(d["antenna_calibration"][side]["rad_max"]),
        )
        for side in ("left", "right")
    }
    return DuckAnimClip(
        name=str(d["name"]),
        fps=int(d["fps"]),
        loop_mode=str(d["loop_mode"]),
        frame_count=frame_count,
        duration_s=float(d["duration_s"]),
        blend_in_s=float(d["blend_in_s"]),
        blend_out_s=float(d["blend_out_s"]),
        show_blend_in_s=float(d["show_blend_in_s"]),
        show_blend_out_s=float(d["show_blend_out_s"]),
        layer_mask=str(d["layer_mask"]),
        priority=int(d["priority"]),
        requires_mode=str(d["requires_mode"]),
        provenance=dict(d["provenance"]),
        joints=frames,
        show=show,
        antenna_calibration=cal,
        version=int(d.get("version", FORMAT_VERSION)),
    )


# --- containers ---------------------------------------------------------------
def _reject_json_constant(token: str):
    """``json.load`` ``parse_constant`` hook: reject bare NaN/Infinity literals.

    Python's ``json`` accepts ``NaN``/``Infinity``/``-Infinity`` by default,
    which would smuggle non-finite values past the schema (plan §6.5 safety).
    """
    raise ClipValidationError(
        "clip JSON contains a non-finite literal %r (NaN/Infinity are rejected)" % token
    )


def load_clip_json(path: str, **kwargs: Any) -> DuckAnimClip:
    """Load a JSON ``.duckanim`` clip (plan §5.2)."""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f, parse_constant=_reject_json_constant)
    return clip_from_dict(d, **kwargs)


def save_clip_npz(path: str, d: Dict[str, Any]) -> None:
    """Write a clip dict to the canonical ``npz`` container (plan §5.2).

    Large tracks (``joints.frames``, antenna arrays, ``eyes``) are stored as
    arrays; everything else — including the *rest* of ``show_functions``
    (``events``, show metadata) — is preserved verbatim in the ``meta_json``
    scalar so no structure is silently dropped on round-trip. This is the exact
    inverse of :func:`load_clip_npz`.
    """
    show = dict(d.get("show_functions", {}))
    a_left = np.asarray(show.pop("antenna_left"), dtype=np.float64)
    a_right = np.asarray(show.pop("antenna_right"), dtype=np.float64)
    eyes = show.pop("eyes", None)
    meta = {k: v for k, v in d.items() if k not in ("joints",)}
    # keep the remaining show_functions structure (events, etc.) in meta_json
    meta["show_functions"] = show
    arrays = {
        "meta_json": json.dumps(meta, sort_keys=True),
        "joints_frames": np.asarray(d["joints"]["frames"], dtype=np.float64),
        "antenna_left": a_left,
        "antenna_right": a_right,
    }
    if eyes is not None:
        arrays["eyes"] = np.asarray(eyes, dtype=np.float64)
    np.savez(path, **arrays)


def load_clip_npz(path: str, **kwargs: Any) -> DuckAnimClip:
    """Load an ``npz`` ``.duckanim`` clip (plan §5.2 alternative container).

    Layout: scalar metadata as a JSON string under ``meta_json``; large tracks
    as arrays ``joints_frames``, ``antenna_left``, ``antenna_right``, ``eyes``.
    """
    with np.load(path, allow_pickle=False) as npz:
        meta = json.loads(str(npz["meta_json"]), parse_constant=_reject_json_constant)
        d = dict(meta)
        d.setdefault("joints", {})
        d["joints"] = {"order": list(JOINT_ORDER_16), "frames": npz["joints_frames"]}
        show = dict(d.get("show_functions", {}))
        show["antenna_left"] = npz["antenna_left"]
        show["antenna_right"] = npz["antenna_right"]
        if "eyes" in npz.files:
            show["eyes"] = npz["eyes"]
        d["show_functions"] = show
    return clip_from_dict(d, **kwargs)


def load_clip(path: str, **kwargs: Any) -> DuckAnimClip:
    """Load a ``.duckanim`` clip, dispatching on file extension (JSON or npz)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        return load_clip_npz(path, **kwargs)
    return load_clip_json(path, **kwargs)
