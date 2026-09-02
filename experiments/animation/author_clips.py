#!/usr/bin/env python3
"""Author the Open Duck Mini v2 expressive head-animation library.

This is the *reusable* authoring tool for the ``.duckanim`` clip library. All the
creative content lives in :func:`build_specs` near the top of the file as a list
of :class:`ClipSpec` objects — a non-expert can copy a spec, change amplitudes /
timings, and re-run to add a clip.

Two backends produce **format-identical** output because both feed the same
59-float source frames into the one shared :mod:`open_duck_anim.compiler`:

* ``--backend procedural`` (default, no Blender needed): the curve engine is
  evaluated straight into 59-float frames.
* ``--backend blender`` (run under ``Blender --background --python``): the same
  per-frame head/antenna Euler values are keyframed onto the real 49-bone rig,
  then read back through the deterministic ``DataRecorder.frame_set`` recorder
  and the calibrated bone->joint ``transform_table``.

The only floats the compiler keeps are the 16 joint angles (plan §5.1), and both
backends compute those from the identical per-frame bone Euler map via the same
``transform_table``, so for a given clip the two backends emit **byte-identical**
``.duckanim`` (proven by ``--verify-identical``). The Blender path additionally
exercises the real rig + recorder end to end.

SAFETY (measured, not optional). Every clip is head-masked (legs held), and every
frame's head offset is checked against the ×0.5 hardware-derated safety envelope
(``open_duck_anim.envelope``): per-channel deflection box + combined L2 budget +
slew. The script refuses to write a clip whose authored motion the envelope would
clamp (``max clamp delta`` must be ~0) — a clip is meant to be authored *inside*
the envelope, never shipped clamped. Pass ``--allow-clamp`` only to inspect.

Usage::

    # procedural (fast, no Blender):
    python experiments/animation/author_clips.py

    # through the real Blender rig (headless):
    /Applications/Blender.app/Contents/MacOS/Blender --background \
        "$BLEND/open-duck-mini.blend" --python experiments/animation/author_clips.py -- \
        --backend blender --only idle_scan curious_tilt

    # prove the two paths agree byte-for-byte for one clip:
    python experiments/animation/author_clips.py --verify-identical --only curious_tilt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --- make the repo importable whether run by python or by Blender -------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_MAIN_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_MAIN_REPO, os.path.join(_MAIN_REPO, "blender")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from open_duck_anim import compiler  # noqa: E402
from open_duck_anim.envelope import DEFAULT_ENVELOPE, HARDWARE_DERATING  # noqa: E402
from open_duck_anim.joint_order import JOINT_ORDER_16  # noqa: E402
from open_duck_anim_blender import export as export_mod  # noqa: E402
from open_duck_anim_blender.transform_table import (  # noqa: E402
    REQUIRED_BONES,
    TRANSFORM_BY_JOINT,
    joints_from_bone_eulers,
)

# Output locations. The .duckanim files live in the repo; heavy artefacts do not.
REPO_CLIPS_DIR = os.path.join(_HERE, "clips")
_SESSION = ("/Users/clancey/.copilot/session-state/"
            "9d7d4839-8a6c-44d0-8b98-328aebd93579/files/clips")
SOURCE_JSON_DIR = os.path.join(_SESSION, "source_json")
BLENDER_OUT_DIR = os.path.join(_SESSION, "blender_out")

FPS = 50
SOURCE_BLEND = "open-duck-mini.blend"

# Head channel order everywhere (plan §3.3 / §6.3).
HEAD_CHANNELS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")

# Antenna calibration: rad_min/rad_max = -1..1 so an authored antenna value maps
# 1:1 to the normalised [-1,1] runtime track (left sign +1, right sign -1, fixed
# hardware constants). Author antenna tracks directly in normalised units.
ANTENNA_CAL = {
    "left": {"sign": 1, "rad_min": -1.0, "rad_max": 1.0},
    "right": {"sign": -1, "rad_min": -1.0, "rad_max": 1.0},
}

# =============================================================================
# Curve engine — small, composable, C2-smooth. Everything a spec needs.
# =============================================================================


def smootherstep(x: np.ndarray) -> np.ndarray:
    """C2-continuous ease 6x^5-15x^4+10x^3 on [0,1] (zero vel+accel at ends).

    Preferred over a linear ramp so the soft head servo (kp=8) is never asked
    for an instantaneous velocity or acceleration change — reads smooth and
    tracks well under lag.
    """
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


_EASINGS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "smooth": smootherstep,
    "linear": lambda x: np.clip(x, 0.0, 1.0),
    # ease_out == fast start, slow stop (decelerate into a hold).
    "ease_out": lambda x: 1.0 - (1.0 - np.clip(x, 0.0, 1.0)) ** 2,
    # ease_in == slow start, fast finish (accelerate out of a hold).
    "ease_in": lambda x: np.clip(x, 0.0, 1.0) ** 2,
    "hold": lambda x: np.zeros_like(x),  # stay at the previous keyframe value
}


class Track:
    """A callable ``f(t_seconds_array) -> values_array`` with + and * algebra.

    Tracks compose so a spec can read like ``keys([...]) + sine(...) + const(c)``.
    """

    def __init__(self, fn: Callable[[np.ndarray], np.ndarray]):
        self._fn = fn

    def __call__(self, t: np.ndarray) -> np.ndarray:
        return np.asarray(self._fn(np.asarray(t, dtype=np.float64)), dtype=np.float64)

    def __add__(self, other: "TrackLike") -> "Track":
        o = as_track(other)
        return Track(lambda t: self(t) + o(t))

    __radd__ = __add__

    def __mul__(self, k: float) -> "Track":
        return Track(lambda t: self(t) * float(k))

    __rmul__ = __mul__


TrackLike = "Track | float | int"


def as_track(x: "TrackLike") -> Track:
    if isinstance(x, Track):
        return x
    val = float(x)
    return Track(lambda t: np.full_like(np.asarray(t, dtype=np.float64), val))


def const(v: float) -> Track:
    return as_track(v)


ZERO = const(0.0)


def keys(points: Sequence[Tuple], loop: bool = False, duration: Optional[float] = None) -> Track:
    """Piecewise keyframe track.

    ``points`` = ``[(t_sec, value[, easing]), ...]`` sorted by time. ``easing``
    (default ``"smooth"``) selects how the segment ENDING at this key eases from
    the previous key. ``"hold"`` keeps the previous value until this key's time
    then jumps — use it to build flat holds between two equal-value keys.

    If ``loop`` is True the value is made seamless by forcing the last value to
    equal the first (so a ``wrap`` clip does not pop at the loop seam); pass
    ``duration`` to anchor the final key.
    """
    pts = [(float(p[0]), float(p[1]), (p[2] if len(p) > 2 else "smooth")) for p in points]
    pts.sort(key=lambda p: p[0])
    if loop:
        if duration is not None and pts[-1][0] < duration - 1e-9:
            pts.append((float(duration), pts[0][1], "smooth"))
        # force seam continuity of value
        pts[-1] = (pts[-1][0], pts[0][1], pts[-1][2])
    ts = np.array([p[0] for p in pts])
    vs = np.array([p[1] for p in pts])
    es = [p[2] for p in pts]

    def fn(t: np.ndarray) -> np.ndarray:
        out = np.empty_like(t)
        for i, tv in enumerate(t):
            if tv <= ts[0]:
                out[i] = vs[0]
                continue
            if tv >= ts[-1]:
                out[i] = vs[-1]
                continue
            k = int(np.searchsorted(ts, tv, side="right"))  # segment [k-1, k]
            t0, t1 = ts[k - 1], ts[k]
            v0, v1 = vs[k - 1], vs[k]
            frac = 0.0 if t1 <= t0 else (tv - t0) / (t1 - t0)
            ease = _EASINGS[es[k]]
            w = float(ease(np.array([frac]))[0])
            out[i] = v0 + (v1 - v0) * w
        return out

    return Track(fn)


def sine(freq_hz: float, amp: float, phase: float = 0.0) -> Track:
    """Continuous sine ``amp*sin(2*pi*freq*t + phase)``. Use integer cycles over
    the loop duration for a seamless ``wrap`` underlay."""
    w = 2.0 * np.pi * float(freq_hz)
    return Track(lambda t: float(amp) * np.sin(w * t + float(phase)))


def drift(*comps: Tuple[float, float, float]) -> Track:
    """Sum of detuned low-amplitude sines ``(freq_hz, amp, phase)`` — a living,
    non-periodic-feeling wander. For a seamless loop use frequencies that are
    integer multiples of ``1/duration``."""
    tracks = [sine(f, a, p) for (f, a, p) in comps]
    out = tracks[0]
    for tr in tracks[1:]:
        out = out + tr
    return out


def pulse(t_center: float, width: float, amp: float, ease: str = "smooth") -> Track:
    """A single smooth there-and-back bump centred at ``t_center`` of half-width
    ``width`` (a raised cosine window). Good for flicks, bobs, double-takes."""
    def fn(t: np.ndarray) -> np.ndarray:
        x = (t - t_center) / width
        w = np.where(np.abs(x) < 1.0, 0.5 * (1.0 + np.cos(np.pi * x)), 0.0)
        if ease == "smooth":
            w = smootherstep(w)
        return amp * w
    return Track(fn)


# =============================================================================
# Clip specification
# =============================================================================


@dataclass
class ClipSpec:
    name: str
    duration_s: float
    loop_mode: str                 # "wrap" | "once" | "clamp"
    requires_mode: str             # "dock" | "stand" | "walk" | "any"
    priority: int
    doc: str = ""                  # one-line catalogue description
    # Designated authoring path for shipping: "parametric" (procedural curve) or
    # "blender" (keyed on the real rig, recorded via frame_set). Both backends
    # produce numerically-equivalent output; this records provenance intent.
    authoring_path: str = "parametric"
    blend_in_s: float = 0.35       # T_alpha body default
    blend_out_s: float = 0.35
    show_blend_in_s: float = 0.10  # T_beta show default
    show_blend_out_s: float = 0.10
    # Head channel tracks (absolute authored head angle, rad). Default zero.
    neck_pitch: TrackLike = 0.0
    head_pitch: TrackLike = 0.0
    head_yaw: TrackLike = 0.0
    head_roll: TrackLike = 0.0
    # Antenna tracks in normalised [-1,1] units (left/right independent). Default
    # is rest (0.0): antennas are quiet by default and only move where the motion
    # carries meaning — real-hardware feedback was that they were audibly noisy.
    antenna_l: TrackLike = 0.0
    antenna_r: TrackLike = 0.0
    # Discrete show events: (type, value, t_sec). Eye blinks: ("eye","blink",t).
    events: Sequence[Tuple[str, str, float]] = field(default_factory=tuple)
    # Optional eye-open/close track as (t_sec, open_bool). Sparse; expands to a
    # per-frame 0/1 track. Empty -> eyes always open (emit nothing).
    eyes: Sequence[Tuple[float, int]] = field(default_factory=tuple)

    @property
    def frame_count(self) -> int:
        return int(round(self.duration_s * FPS))

    def head_tracks(self) -> Dict[str, Track]:
        return {
            "neck_pitch": as_track(self.neck_pitch),
            "head_pitch": as_track(self.head_pitch),
            "head_yaw": as_track(self.head_yaw),
            "head_roll": as_track(self.head_roll),
        }


# =============================================================================
# THE LIBRARY.  Edit here to tune or add clips.
# =============================================================================
#
# Sign conventions (approximate, chosen for legibility):
#   neck_pitch  + = lower head / lean forward,  - = lift head / lean back
#   head_pitch  + = chin down (nod down),       - = chin up (nod up)
#   head_yaw    + = turn one way,               - = turn the other
#   head_roll   + = tilt one way,               - = tilt the other
#   antenna_*   + = perked up,                  - = laid back
#
# Amplitudes are deliberately kept inside the ×0.5 derated envelope so nothing is
# clamped at runtime (the script asserts this). Derated single-axis ceilings
# (after the combined L2 budget) are roughly: neck ±0.12, head_pitch ±0.27,
# head_yaw ±0.52, head_roll ±0.17 rad. Multi-axis motion shares that budget.


def build_specs() -> List[ClipSpec]:
    # --- Antenna philosophy (explicit owner decision, measured HW noise) -----
    # The antennas are open-loop 9g-class hobby servos on GPIO (D13 left, D12
    # right). On the physical robot the owner reported they are audibly noisy:
    # PWM hobby servos buzz and chatter in proportion to how OFTEN they are
    # driven, not merely how far they travel. A dock idle loop may run for many
    # minutes continuously on a desk, so *any* antenna motion inside a loop is
    # effectively continuous buzz.
    #
    # OWNER DECISION (watching the robot): "for the idle animations, I don't want
    # to use the antennas. They are very noisy." Taken literally: looping / idle
    # clips move the antennas ZERO — the antenna tracks are a flat constant at the
    # neutral rest value for the whole clip, so the runtime issues no changing
    # antenna command and the servos are never asked to move. An earlier pass only
    # REDUCED idle antenna motion; that was not enough, because any motion in a
    # minutes-long loop still buzzes.
    #
    # Antenna motion is RESERVED for the brief triggered reactions (once clips) —
    # a startle snap, a happy flick, a sad fold — where a short CRISP gesture
    # carries meaning and is over in a moment, not sustained noise. Do NOT
    # "helpfully" add antenna motion back into any looping / idle clip: the head
    # is what makes the duck feel alive; the antennas are momentary punctuation
    # only. The library guard test (tests/test_clip_library.py) enforces zero
    # antenna motion in every loop_mode=="wrap" / background-layer clip. See the
    # clips README for the full rationale.

    specs: List[ClipSpec] = []

    # ---- A. Idle / "alive" loops (wrap) -------------------------------------
    # Varied loop lengths (6/8/11 s) so overlapping idle layers never sync up.

    # 1) Breathing: slow neck bob + a gentle counter head_pitch + antenna drift.
    d = 6.0
    f = 1.0 / d
    specs.append(ClipSpec(
        name="idle_breathe", duration_s=d, loop_mode="wrap", requires_mode="any",
        priority=0, blend_in_s=0.0, blend_out_s=0.0, show_blend_in_s=0.0,
        show_blend_out_s=0.0,
        doc="Slow breathing-like neck bob; the default background 'alive' loop.",
        neck_pitch=sine(f, 0.045, 0.0) + sine(2 * f, 0.012, 0.7),
        head_pitch=sine(f, 0.018, np.pi),          # counter-move, follow-through
        head_roll=sine(f, 0.010, 0.9),             # micro weight shift
        # Antennas rest fully in the default dock background loop — the quietest
        # option for a loop that may run for minutes (hardware-noise feedback).
        antenna_l=ZERO,
        antenna_r=ZERO,
    ))

    # 2) Slow scan: occasional slow left-right look with holds, breathing under.
    d = 11.0
    f = 1.0 / d
    specs.append(ClipSpec(
        name="idle_scan", duration_s=d, loop_mode="wrap", requires_mode="any",
        priority=0, blend_in_s=0.0, blend_out_s=0.0, show_blend_in_s=0.0,
        show_blend_out_s=0.0,
        doc="Occasional slow head scan with holds over a breathing underlay.",
        neck_pitch=sine(f, 0.035, 0.0),
        head_yaw=keys([(0.0, 0.0), (2.5, 0.34, "ease_out"), (4.0, 0.34, "hold"),
                       (6.5, -0.30, "smooth"), (8.0, -0.30, "hold"),
                       (11.0, 0.0, "smooth")], loop=True, duration=d),
        head_roll=sine(2 * f, 0.020, 0.4),         # slight roll trailing the yaw
        # Owner decision: idle/looping clips never move the antennas (measured
        # hardware noise). Flat at rest for the whole loop — zero antenna command.
        antenna_l=ZERO,
        antenna_r=ZERO,
    ))

    # 3) Micro look-around: non-periodic-feeling wander (detuned sines) + shifts.
    d = 8.0
    f = 1.0 / d
    specs.append(ClipSpec(
        name="idle_lookaround", duration_s=d, loop_mode="wrap", requires_mode="any",
        priority=0, blend_in_s=0.0, blend_out_s=0.0, show_blend_in_s=0.0,
        show_blend_out_s=0.0,
        doc="Restless micro weight-shifts and gaze wander; never quite repeats.",
        neck_pitch=drift((f, 0.030, 0.0), (2 * f, 0.015, 2.0)),
        head_yaw=drift((f, 0.12, 0.3), (2 * f, 0.07, 1.7), (3 * f, 0.04, 0.9)),
        head_roll=drift((f, 0.05, 1.2), (2 * f, 0.03, 0.1)),
        head_pitch=drift((2 * f, 0.02, 0.5)),
        # Owner decision: idle/looping clips never move the antennas (measured
        # hardware noise). Flat at rest for the whole loop — zero antenna command.
        antenna_l=ZERO,
        antenna_r=ZERO,
    ))

    # ---- B. Curiosity / attention (once) ------------------------------------

    # 4) Curious head tilt with a hold + a single blink partway in.
    specs.append(ClipSpec(
        name="curious_tilt", authoring_path="blender", duration_s=2.6, loop_mode="once", requires_mode="any",
        priority=10, blend_in_s=0.25, blend_out_s=0.35,
        doc="Inquisitive head-roll tilt held briefly, with a blink.",
        head_roll=keys([(0.0, 0.0), (0.7, 0.16, "ease_out"), (1.8, 0.16, "hold"),
                        (2.6, 0.0, "smooth")]),
        head_yaw=keys([(0.0, 0.0), (0.7, 0.07, "ease_out"), (1.8, 0.07, "hold"),
                       (2.6, 0.0, "smooth")]),
        neck_pitch=pulse(0.8, 0.9, -0.03),          # tiny perk into the tilt
        antenna_l=keys([(0.0, 0.0), (0.7, 0.18, "ease_out"), (2.6, 0.0)]),
        antenna_r=keys([(0.0, 0.0), (0.8, 0.14, "ease_out"), (2.6, 0.0)]),
        events=(("eye", "blink", 0.85),),
    ))

    # 5) Look-toward: turn to one side with a slight downward interest, hold.
    specs.append(ClipSpec(
        name="look_toward", authoring_path="blender", duration_s=2.2, loop_mode="once", requires_mode="any",
        priority=10, blend_in_s=0.3, blend_out_s=0.4,
        doc="Directed look toward a point of interest, held, then released.",
        head_yaw=keys([(0.0, 0.0), (0.7, 0.42, "ease_out"), (1.5, 0.42, "hold"),
                       (2.2, 0.0, "smooth")]),
        neck_pitch=keys([(0.0, 0.0), (0.8, 0.04, "ease_out"), (1.5, 0.04, "hold"),
                         (2.2, 0.0)]),
        head_pitch=keys([(0.0, 0.0), (0.8, 0.03, "ease_out"), (2.2, 0.0)]),
        antenna_l=keys([(0.0, 0.0), (0.7, 0.15, "ease_out"), (2.2, 0.05)]),
        antenna_r=keys([(0.0, 0.0), (0.7, 0.11, "ease_out"), (2.2, 0.05)]),
    ))

    # 6) Double-take: glance one way, snap back the other, settle. Blink on snap.
    specs.append(ClipSpec(
        name="double_take", authoring_path="blender", duration_s=2.4, loop_mode="once", requires_mode="stand",
        priority=12, blend_in_s=0.15, blend_out_s=0.35,
        doc="Glance away then a quick snap-back double-take. Stand only (snappy).",
        head_yaw=keys([(0.0, 0.0), (0.4, 0.18, "ease_out"), (0.8, 0.18, "hold"),
                       (1.15, -0.38, "ease_out"), (1.7, -0.34, "hold"),
                       (2.4, 0.0, "smooth")]),
        head_roll=pulse(1.2, 0.4, -0.06),           # counter-roll on the snap
        neck_pitch=pulse(1.2, 0.5, -0.03),
        antenna_l=keys([(0.0, 0.0), (1.15, 0.22, "ease_out"), (2.4, 0.0)]),
        antenna_r=keys([(0.0, 0.0), (1.15, 0.19, "ease_out"), (2.4, 0.0)]),
        events=(("eye", "blink", 1.15),),
    ))

    # 7) Perk-up / alert: head lifts, antennas snap up, brief scan, hold.
    specs.append(ClipSpec(
        name="perk_up", authoring_path="blender", duration_s=1.8, loop_mode="once", requires_mode="stand",
        priority=15, blend_in_s=0.12, blend_out_s=0.3,
        doc="Sudden alert perk-up: head lifts, antennas raise, brief scan.",
        neck_pitch=keys([(0.0, 0.0), (0.35, -0.09, "ease_out"), (1.2, -0.08, "hold"),
                         (1.8, 0.0, "smooth")]),
        head_pitch=keys([(0.0, 0.0), (0.35, -0.12, "ease_out"), (1.2, -0.10, "hold"),
                         (1.8, 0.0)]),
        head_yaw=pulse(1.0, 0.5, 0.12),             # a quick look-around at the top
        # Perk gesture preserved (crisp raise + hold) but the excursion halved —
        # "ears up" still reads without driving the servo to a loud extreme.
        antenna_l=keys([(0.0, 0.0), (0.3, 0.4, "ease_out"), (1.3, 0.35, "hold"),
                        (1.8, 0.05)]),
        antenna_r=keys([(0.0, 0.0), (0.3, 0.4, "ease_out"), (1.3, 0.35, "hold"),
                        (1.8, 0.05)]),
        events=(("eye", "wide", 0.3),),
    ))

    # 8) Slow deliberate scan left->right, once (curiosity survey).
    specs.append(ClipSpec(
        name="scan_curious", authoring_path="blender", duration_s=4.0, loop_mode="once", requires_mode="any",
        priority=10, blend_in_s=0.3, blend_out_s=0.4,
        doc="Deliberate slow survey scan from one side to the other and back.",
        head_yaw=keys([(0.0, 0.0), (1.4, -0.40, "ease_out"), (2.0, -0.40, "hold"),
                       (3.4, 0.40, "smooth"), (4.0, 0.0, "ease_in")]),
        neck_pitch=sine(0.25, 0.03, 0.0),
        head_roll=keys([(0.0, 0.0), (1.4, -0.06), (3.4, 0.06), (4.0, 0.0)]),
        antenna_l=keys([(0.0, 0.03), (2.0, 0.12), (4.0, 0.03)]),
        antenna_r=keys([(0.0, 0.03), (2.0, 0.10), (4.0, 0.03)]),
    ))

    # ---- C. Expressive reactions (once) -------------------------------------

    # 9) Yes-nod: two chin-down nods with follow-through, small neck bob.
    specs.append(ClipSpec(
        name="nod_yes", authoring_path="blender", duration_s=2.2, loop_mode="once", requires_mode="any",
        priority=20, blend_in_s=0.15, blend_out_s=0.25,
        doc="Affirmative double nod (yes). Small enough to use in any mode.",
        head_pitch=(pulse(0.55, 0.4, 0.20) + pulse(1.25, 0.4, 0.17)),
        neck_pitch=(pulse(0.55, 0.45, 0.05) + pulse(1.25, 0.45, 0.04)),
        antenna_l=pulse(0.55, 0.7, 0.10),
        antenna_r=pulse(0.55, 0.7, 0.08),
    ))

    # 10) No-shake: head yaw oscillation, antennas trailing (follow-through).
    specs.append(ClipSpec(
        name="shake_no", authoring_path="blender", duration_s=2.2, loop_mode="once", requires_mode="stand",
        priority=20, blend_in_s=0.15, blend_out_s=0.3,
        doc="Negative head shake (no). Stand only — yaw amplitude reads big.",
        head_yaw=(pulse(0.55, 0.35, 0.34) + pulse(1.1, 0.35, -0.34)
                  + pulse(1.6, 0.32, 0.22)),
        head_roll=sine(1.4, 0.03, 0.0),
        antenna_l=(pulse(0.7, 0.5, 0.12) + pulse(1.25, 0.5, -0.08)),  # trailing
        antenna_r=(pulse(0.7, 0.5, 0.10) + pulse(1.25, 0.5, -0.06)),
    ))

    # 11) Happy bounce: neck bob up + a bright antenna flick (event on the flick).
    specs.append(ClipSpec(
        name="happy_bounce", authoring_path="blender", duration_s=2.0, loop_mode="once", requires_mode="stand",
        priority=18, blend_in_s=0.12, blend_out_s=0.3,
        doc="Delighted bob: head bounces up twice with a bright antenna flick.",
        neck_pitch=(pulse(0.45, 0.35, -0.10) + pulse(1.0, 0.35, -0.08)),
        head_pitch=(pulse(0.45, 0.35, -0.10) + pulse(1.0, 0.35, -0.08)),
        head_roll=pulse(1.3, 0.5, 0.08),
        # The flick is this clip's read, so it is preserved — but as a small CRISP
        # flick (excursion halved, and the width eased out just enough to stay
        # within the lowered antenna slew cap). A big fast antenna sweep is what
        # buzzed on hardware; a quick low flick still says "delight".
        antenna_l=(pulse(0.45, 0.34, 0.45) + pulse(1.0, 0.34, 0.35)),
        antenna_r=(pulse(0.45, 0.34, 0.45) + pulse(1.0, 0.34, 0.35)),
        events=(("antenna", "flick", 0.45), ("eye", "happy", 0.45)),
    ))

    # 12) Sad droop: head sinks forward and down, antennas fold back, slow settle.
    specs.append(ClipSpec(
        name="sad_droop", authoring_path="blender", duration_s=3.2, loop_mode="once", requires_mode="stand",
        priority=16, blend_in_s=0.4, blend_out_s=0.5,
        doc="Dejected droop: head sinks, antennas fold back, slow sighing settle.",
        neck_pitch=keys([(0.0, 0.0), (1.1, 0.09, "ease_out"), (2.2, 0.09, "hold"),
                         (3.2, 0.0, "smooth")]),
        head_pitch=keys([(0.0, 0.0), (1.1, 0.14, "ease_out"), (2.2, 0.13, "hold"),
                         (3.2, 0.0, "smooth")]),
        head_roll=sine(0.3, 0.03, 0.0),
        # Fold-back preserved but shallower: a moderate fold still reads dejected
        # while cutting the large slow sweep that made noise on hardware.
        antenna_l=keys([(0.0, 0.0), (1.1, -0.30, "ease_out"), (2.4, -0.28, "hold"),
                        (3.2, 0.0)]),
        antenna_r=keys([(0.0, 0.0), (1.1, -0.30, "ease_out"), (2.4, -0.28, "hold"),
                        (3.2, 0.0)]),
        eyes=((0.0, 1), (1.0, 0), (1.35, 1)),       # a slow blink as it sinks
    ))

    # 13) Startle: fast recoil (head snaps up/back), then a wary settle. Fast in.
    specs.append(ClipSpec(
        name="startle", authoring_path="blender", duration_s=1.6, loop_mode="once", requires_mode="stand",
        priority=30, blend_in_s=0.05, blend_out_s=0.35,
        doc="Startled recoil: fast head snap back then a wary settle. Highest prio.",
        neck_pitch=keys([(0.0, 0.0), (0.18, -0.09, "ease_out"), (0.5, -0.05, "smooth"),
                         (1.6, 0.0, "smooth")]),
        head_pitch=keys([(0.0, 0.0), (0.18, -0.13, "ease_out"), (0.5, -0.05, "smooth"),
                         (1.6, 0.0)]),
        head_roll=pulse(0.6, 0.35, -0.09),          # flinch to one side
        head_yaw=pulse(0.85, 0.4, 0.12),            # dart a glance
        # The snap is the read (fast + crisp = alarm), so it is preserved — but
        # the excursion is halved AND the front-loaded velocity is gentled with a
        # smooth ease so it stays within the (lowered) antenna slew cap: the
        # antenna no longer slams, which is what buzzed on hardware.
        antenna_l=keys([(0.0, 0.0), (0.25, 0.5, "smooth"), (0.9, 0.2, "smooth"),
                        (1.6, 0.05)]),
        antenna_r=keys([(0.0, 0.0), (0.25, 0.5, "smooth"), (0.9, 0.2, "smooth"),
                        (1.6, 0.05)]),
        events=(("eye", "wide", 0.1),),
    ))

    # ---- D. Walk-compatible variants (small amplitude, requires_mode=walk) ---

    # 14) Look-around while walking: gentle yaw wander, small roll, seamless loop.
    d = 7.0
    f = 1.0 / d
    specs.append(ClipSpec(
        name="walk_look_around", duration_s=d, loop_mode="wrap", requires_mode="walk",
        priority=5, blend_in_s=0.35, blend_out_s=0.35, show_blend_in_s=0.1,
        show_blend_out_s=0.1,
        doc="Gentle gaze wander to overlay while walking. Small, legible, seamless.",
        head_yaw=drift((f, 0.28, 0.2), (2 * f, 0.10, 1.4)),
        head_roll=drift((f, 0.05, 0.8)),
        neck_pitch=sine(f, 0.03, 0.0),
        # Owner decision: idle/looping/background layers never move the antennas
        # (measured hardware noise). This wrap loop can overlay a long walk, so
        # it stays flat at rest — the head carries the "looking around" read.
        antenna_l=ZERO,
        antenna_r=ZERO,
    ))

    # 15) Alert while walking: a contained perk + a small look, once. Authored
    # with a clear yaw sweep and a short hold (rather than a long dwell) so the
    # motion reads cleanly above the walking gait's head disturbance.
    specs.append(ClipSpec(
        name="walk_alert", authoring_path="blender", duration_s=2.0, loop_mode="once", requires_mode="walk",
        priority=15, blend_in_s=0.2, blend_out_s=0.35,
        doc="Contained 'something caught my eye' alert usable mid-stride.",
        neck_pitch=keys([(0.0, 0.0), (0.4, -0.07, "ease_out"), (1.0, -0.06, "hold"),
                         (2.0, 0.0, "smooth")]),
        head_pitch=keys([(0.0, 0.0), (0.45, -0.05, "ease_out"), (1.0, -0.045, "hold"),
                         (2.0, 0.0, "smooth")]),
        head_yaw=keys([(0.0, 0.0), (0.5, 0.34, "ease_out"), (0.95, 0.32, "hold"),
                       (2.0, 0.0, "smooth")]),
        antenna_l=keys([(0.0, 0.0), (0.35, 0.25, "ease_out"), (1.4, 0.2, "hold"),
                        (2.0, 0.05)]),
        antenna_r=keys([(0.0, 0.0), (0.35, 0.25, "ease_out"), (1.4, 0.2, "hold"),
                        (2.0, 0.05)]),
        events=(("eye", "blink", 0.4),),
    ))

    return specs


# =============================================================================
# Evaluation: spec -> per-frame bone Euler map -> 59-float source frames
# =============================================================================


def _eval_head_and_antennas(spec: ClipSpec):
    """Return ``(head4 [N,4], antennas [N,2])`` sampled at frame centres.

    Frames are sampled at ``t = i/FPS`` for ``i in [0, N)`` (the recorder samples
    frame ``i`` at scene frame ``first+i``; time base identical in both backends).
    """
    n = spec.frame_count
    t = np.arange(n, dtype=np.float64) / FPS
    tracks = spec.head_tracks()
    head4 = np.stack([tracks[c](t) for c in HEAD_CHANNELS], axis=1)
    a_l = as_track(spec.antenna_l)(t)
    a_r = as_track(spec.antenna_r)(t)
    antennas = np.stack([a_l, a_r], axis=1)
    return head4, antennas


def _bone_euler_map_for_frame(head4_row: np.ndarray, antenna_row: np.ndarray) -> Dict[str, Tuple[float, float, float]]:
    """Build the ``{bone: (rx,ry,rz)}`` map that reproduces this frame's joints.

    Head + antenna joints are placed on each bone's declared axis via the inverse
    of the calibrated transform; every OTHER (leg) bone stays at rest Euler
    ``(0,0,0)`` so the legs are held (the clip is head-masked). This is the exact
    map both backends use, guaranteeing identical joint output.
    """
    euler = {b: [0.0, 0.0, 0.0] for b in REQUIRED_BONES}
    # antenna joint values are the raw radians the compiler will calibrate; here
    # authored antenna == desired normalised value, and calibration rad_min/max
    # = -1..1 makes the raw joint angle equal that same number (left) / its
    # negative is handled by the RIGHT sign downstream — so store the value such
    # that compiler output == authored. left: joint = value; right: joint = -value.
    joint_targets = {
        "neck_pitch": head4_row[0],
        "head_pitch": head4_row[1],
        "head_yaw": head4_row[2],
        "head_roll": head4_row[3],
        "left_antenna": antenna_row[0],
        "right_antenna": -antenna_row[1],
    }
    for joint_name, value in joint_targets.items():
        tr = TRANSFORM_BY_JOINT[joint_name]
        euler[tr.bone][tr.axis] = tr.inverse(float(value))
    return {b: tuple(e) for b, e in euler.items()}


def build_episode_procedural(spec: ClipSpec) -> Dict:
    """Evaluate the spec straight into a 59-float episode (no Blender)."""
    head4, antennas = _eval_head_and_antennas(spec)
    n = spec.frame_count
    episode = export_mod.new_episode(fps=FPS, contacts_valid=False)
    zeros3, zeros16, zeros2 = [0.0] * 3, [0.0] * 16, [0.0, 0.0]
    ident_quat = [0.0, 0.0, 0.0, 1.0]
    for i in range(n):
        euler_map = _bone_euler_map_for_frame(head4[i], antennas[i])
        joints16 = joints_from_bone_eulers(euler_map)
        frame = export_mod.assemble_frame(
            root_position=zeros3, root_quaternion=ident_quat,
            joint_positions=joints16, left_toe_pos=zeros3, right_toe_pos=zeros3,
            world_linear_vel=zeros3, world_angular_vel=zeros3,
            joint_velocities=zeros16, left_toe_vel=zeros3, right_toe_vel=zeros3,
            foot_contacts=zeros2,
        )
        episode["Frames"].append(frame)
    return episode


def spec_to_meta(spec: ClipSpec) -> Dict:
    """Compiler metadata for a spec (mirrors ClipMetadata.to_compiler_meta)."""
    n = spec.frame_count
    # expand sparse eye keyframes into a per-frame 0/1 track (step-held).
    eyes: List[int] = []
    if spec.eyes:
        pts = sorted(spec.eyes, key=lambda p: p[0])
        for i in range(n):
            ti = i / FPS
            val = pts[0][1]
            for (tk, vk) in pts:
                if ti + 1e-9 >= tk:
                    val = int(vk)
            eyes.append(int(val))
    events = [
        {"frame": int(round(t * FPS)), "type": str(typ), "value": str(val)}
        for (typ, val, t) in spec.events
    ]
    for ev in events:
        ev["frame"] = max(0, min(n - 1, ev["frame"]))
    return {
        "name": spec.name,
        "loop_mode": spec.loop_mode,
        "requires_mode": spec.requires_mode,
        "priority": int(spec.priority),
        "layer_mask": "head",
        "blend_in_s": float(spec.blend_in_s),
        "blend_out_s": float(spec.blend_out_s),
        "show_blend_in_s": float(spec.show_blend_in_s),
        "show_blend_out_s": float(spec.show_blend_out_s),
        "source_blend": SOURCE_BLEND,
        "antenna_calibration": ANTENNA_CAL,
        "fps": FPS,
        "eyes": eyes,
        "events": events,
    }


# =============================================================================
# Derated-envelope self-check (design-time safety gate)
# =============================================================================


def envelope_report(spec: ClipSpec, derating: float = HARDWARE_DERATING) -> Dict:
    """Run the authored head offsets through the derated envelope frame by frame
    (with the slew guard, as the runtime does) and report how close the motion
    sits to the limits and whether the envelope would clamp it."""
    head4, _ = _eval_head_and_antennas(spec)
    env = DEFAULT_ENVELOPE.derated(derating)
    L = env._L
    prev = None
    max_defl = np.zeros(4)
    max_l2 = 0.0
    max_clamp_delta = 0.0
    for row in head4:
        clamped = env.clamp(row, prev_command_head=prev, dt=1.0 / FPS)
        max_clamp_delta = max(max_clamp_delta, float(np.max(np.abs(clamped - row))))
        max_defl = np.maximum(max_defl, np.abs(row))
        max_l2 = max(max_l2, float(np.sqrt(np.sum((row / L) ** 2))))
        prev = clamped
    return {
        "max_abs_deflection": {c: float(max_defl[i]) for i, c in enumerate(HEAD_CHANNELS)},
        "max_l2_norm": max_l2,
        "l2_budget": env.l2_budget,
        "max_clamp_delta_rad": max_clamp_delta,
        "derating": derating,
    }


# =============================================================================
# Backends
# =============================================================================


def compile_spec(episode: Dict, spec: ClipSpec, out_path: str, source_path: str) -> str:
    """Write the 59-float source JSON and compile a validated .duckanim."""
    meta = spec_to_meta(spec)
    res = export_mod.export_and_compile(episode, meta, source_path, out_path)
    return res["source_sha256"]


def run_procedural(specs: List[ClipSpec], out_dir: str, allow_clamp: bool) -> List[Dict]:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(SOURCE_JSON_DIR, exist_ok=True)
    results = []
    for spec in specs:
        rep = envelope_report(spec)
        clamp = rep["max_clamp_delta_rad"]
        status = "ok"
        if clamp > 1e-4:
            status = "CLAMPED"
            if not allow_clamp:
                raise SystemExit(
                    "clip %r would be clamped by the ×%.2f derated envelope "
                    "(max clamp delta %.4f rad, max ||c/L||=%.3f > budget %.2f). "
                    "Reduce amplitudes; do not ship clamped."
                    % (spec.name, rep["derating"], clamp, rep["max_l2_norm"],
                       rep["l2_budget"]))
        episode = build_episode_procedural(spec)
        out_path = os.path.join(out_dir, spec.name + ".duckanim")
        source_path = os.path.join(SOURCE_JSON_DIR, spec.name + ".source.json")
        sha = compile_spec(episode, spec, out_path, source_path)
        results.append({"name": spec.name, "path": out_path, "sha256": sha,
                        "status": status, **rep})
        print("[procedural] %-18s frames=%-4d ||c/L||max=%.3f clampΔ=%.4g  %s"
              % (spec.name, spec.frame_count, rep["max_l2_norm"], clamp, status))
    return results


# ---- Blender backend (only importable/runnable inside Blender) ---------------

# Default rig location (git-lfs real file, verified on Blender 5.2.1).
DEFAULT_BLEND = ("/Users/clancey/.copilot/session-state/"
                 "9d7d4839-8a6c-44d0-8b98-328aebd93579/files/upstream/"
                 "Open_Duck_Blender/open-duck-mini.blend")

# The six bones a head-masked clip animates, with the Euler axis each uses.
_ANIMATED_JOINTS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll",
                    "left_antenna", "right_antenna")


def run_blender(specs: List[ClipSpec], out_dir: str, allow_clamp: bool,
                blend_path: str = DEFAULT_BLEND) -> List[Dict]:
    """Drive the real rig headless: keyframe head/antenna bones, record via the
    deterministic frame_set recorder, compile. Legs are held at rest (the clip is
    head-masked). Mirrors the proven Phase-2 verification path (verify_real.py)."""
    import bpy  # type: ignore
    from open_duck_anim_blender import recorder as rec_mod

    if blend_path and os.path.exists(blend_path):
        bpy.ops.wm.open_mainfile(filepath=blend_path)

    # find the armature that carries our bones.
    armature_name = None
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE" and all(b in obj.pose.bones for b in REQUIRED_BONES):
            armature_name = obj.name
            break
    if armature_name is None:
        raise SystemExit("no armature with the required bones found in the .blend")
    obj = bpy.data.objects[armature_name]
    scene = bpy.context.scene
    print("[blender] armature:", armature_name, "| bones:", len(obj.pose.bones))

    # bone + axis for each animated joint (from the calibrated transform table).
    anim = [(jn, TRANSFORM_BY_JOINT[jn].bone, TRANSFORM_BY_JOINT[jn].axis)
            for jn in _ANIMATED_JOINTS]

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(SOURCE_JSON_DIR, exist_ok=True)
    results = []
    for spec in specs:
        rep = envelope_report(spec)
        if rep["max_clamp_delta_rad"] > 1e-4 and not allow_clamp:
            raise SystemExit("clip %r would be clamped; fix amplitudes" % spec.name)
        head4, antennas = _eval_head_and_antennas(spec)
        n = spec.frame_count

        # Hold legs at rest: drop any action, set every required bone to XYZ euler
        # (0,0,0). Legs then stay constant across all frames (held).
        if obj.animation_data is not None:
            obj.animation_data.action = None
        for b in REQUIRED_BONES:
            pb = obj.pose.bones[b]
            pb.rotation_mode = "XYZ"
            pb.rotation_euler = (0.0, 0.0, 0.0)

        scene.frame_start = 1
        scene.frame_end = n
        for i in range(n):
            f = i + 1
            targets = {
                "neck_pitch": head4[i, 0], "head_pitch": head4[i, 1],
                "head_yaw": head4[i, 2], "head_roll": head4[i, 3],
                "left_antenna": antennas[i, 0], "right_antenna": -antennas[i, 1],
            }
            for jn, bone, axis in anim:
                pb = obj.pose.bones[bone]
                pb.rotation_euler[axis] = TRANSFORM_BY_JOINT[jn].inverse(float(targets[jn]))
                pb.keyframe_insert("rotation_euler", index=axis, frame=f)

        recorder = rec_mod.DataRecorder(armature_name=armature_name, fps=FPS,
                                        contacts_valid=False)
        episode = recorder.record()

        out_path = os.path.join(out_dir, spec.name + ".duckanim")
        source_path = os.path.join(SOURCE_JSON_DIR, spec.name + ".blender.source.json")
        sha = compile_spec(episode, spec, out_path, source_path)
        results.append({"name": spec.name, "path": out_path, "sha256": sha, **rep})
        print("[blender] %-18s frames=%-4d recorded+compiled -> %s"
              % (spec.name, n, os.path.basename(out_path)))
    return results


# =============================================================================
# CLI
# =============================================================================


def _clips_equivalent(da: Dict, db: Dict, tol: float = 1e-5):
    """True if two clip dicts are format-identical and numerically equivalent.

    All non-track fields (metadata, provenance except source hash, joints.order,
    show events/eyes, antenna calibration) must match exactly; the joint frames
    and antenna tracks must match within ``tol`` (Blender float32 vs float64)."""
    scalar_keys = ("format", "version", "name", "fps", "loop_mode", "frame_count",
                   "duration_s", "blend_in_s", "blend_out_s", "show_blend_in_s",
                   "show_blend_out_s", "layer_mask", "priority", "requires_mode",
                   "antenna_calibration")
    for k in scalar_keys:
        if da.get(k) != db.get(k):
            return False, "field %r differs (%r vs %r)" % (k, da.get(k), db.get(k))
    if da["joints"]["order"] != db["joints"]["order"]:
        return False, "joints.order differs"
    ja = np.asarray(da["joints"]["frames"]); jb = np.asarray(db["joints"]["frames"])
    if ja.shape != jb.shape:
        return False, "joints.frames shape %r vs %r" % (ja.shape, jb.shape)
    dmax = float(np.max(np.abs(ja - jb))) if ja.size else 0.0
    if dmax > tol:
        return False, "joints differ by %.2e > %.0e" % (dmax, tol)
    for key in ("antenna_left", "antenna_right"):
        aa = np.asarray(da["show_functions"][key]); bb = np.asarray(db["show_functions"][key])
        if aa.shape != bb.shape or (aa.size and float(np.max(np.abs(aa - bb))) > tol):
            return False, "%s differs" % key
    if da["show_functions"].get("events") != db["show_functions"].get("events"):
        return False, "events differ"
    if list(da["show_functions"].get("eyes", [])) != list(db["show_functions"].get("eyes", [])):
        return False, "eyes differ"
    return True, "max joint delta %.2e" % dmax


def _parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["procedural", "blender"], default="procedural")
    ap.add_argument("--out-dir", default=REPO_CLIPS_DIR,
                    help="where to write .duckanim (default: repo clips dir)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="only build these clip names")
    ap.add_argument("--path", choices=["blender", "parametric"], default=None,
                    help="only build clips whose designated authoring path matches")
    ap.add_argument("--blend", default=DEFAULT_BLEND,
                    help="path to the .blend rig (blender backend only)")
    ap.add_argument("--allow-clamp", action="store_true",
                    help="do not fail if a clip exceeds the derated envelope")
    ap.add_argument("--verify-identical", action="store_true",
                    help="build procedural into a scratch dir and compare "
                         "structurally+numerically with the clips in --out-dir")
    ap.add_argument("--list", action="store_true", help="list clips and exit")
    return ap.parse_args(argv)


def _argv_after_dashdash() -> List[str]:
    # Under Blender, real args come after a literal '--'.
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1:]
    # Plain python: drop the script name.
    return sys.argv[1:]


def main() -> int:
    args = _parse_args(_argv_after_dashdash())
    specs = build_specs()
    if args.only:
        specs = [s for s in specs if s.name in set(args.only)]
        if not specs:
            raise SystemExit("no clips matched --only %r" % (args.only,))
    if args.path:
        specs = [s for s in specs if s.authoring_path == args.path]
        if not specs:
            raise SystemExit("no clips matched --path %r" % (args.path,))
    if args.list:
        for s in specs:
            print("%-18s %5.1fs %-5s %-6s prio=%-3d path=%-10s  %s"
                  % (s.name, s.duration_s, s.loop_mode, s.requires_mode,
                     s.priority, s.authoring_path, s.doc))
        return 0

    if args.verify_identical:
        # Build procedural into a scratch dir and compare to the clips in
        # out_dir (typically Blender-recorded) STRUCTURALLY + NUMERICALLY. The two
        # backends cannot be byte-identical because Blender stores rotation_euler
        # as float32, so head/antenna joints differ by ~1e-7; everything else
        # (schema, metadata, leg values, frame counts) must match exactly.
        scratch = os.path.join(_SESSION, "verify_scratch")
        run_procedural(specs, scratch, allow_clamp=True)
        mismatches = 0
        for s in specs:
            a = os.path.join(scratch, s.name + ".duckanim")
            b = os.path.join(args.out_dir, s.name + ".duckanim")
            if not os.path.exists(b):
                print("  MISSING in out-dir:", s.name); mismatches += 1; continue
            da = json.load(open(a)); db = json.load(open(b))
            ok, why = _clips_equivalent(da, db)
            print("  %-18s %s%s" % (s.name, "equivalent" if ok else "DIFFERS",
                                    "" if ok else " (%s)" % why))
            if not ok:
                mismatches += 1
        print("verify-identical: %d mismatches" % mismatches)
        return 1 if mismatches else 0

    if args.backend == "blender":
        run_blender(specs, args.out_dir, args.allow_clamp, blend_path=args.blend)
    else:
        run_procedural(specs, args.out_dir, args.allow_clamp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
