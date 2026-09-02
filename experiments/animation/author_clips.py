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

    # ---- A2. Emotional mood idle loops (wrap, priority 0) -------------------
    # Moods the duck can *sit in* — so it can BE happy/sad/sleepy for a while,
    # not just DO a happy thing. Each reads distinctly from POSTURE + RHYTHM
    # alone. The single most important authoring insight (measured): neck_pitch
    # is effectively maxed (only ~0.24 rad derated ptp) and is the binding
    # constraint, so emotion is built from head_roll (tilt), head_pitch
    # (carriage), timing, and the eyes — NEVER from neck_pitch. Head TILT (roll)
    # is the classic emotional channel and is ~3x unused; a slight PERSISTENT
    # roll bias is the cheapest, strongest "this duck feels something" signal.
    #
    # Antennas are held flat at rest (ZERO) in every one of these loops — owner
    # decision, measured hardware noise; enforced by the library guard test.
    # Blink CADENCE is an emotional channel and is free of that constraint, so
    # each mood carries its feeling partly in the eyes (per-frame `eyes` track):
    # happy=frequent quick blinks, sad=one slow heavy blink, sleepy=long droopy
    # closes, alert=rare, grumpy=terse. Durations are mutually co-prime-ish
    # (6.5/7.5/8.5/9.5/12.0) and distinct from the neutral idles (6/8/11) so no
    # two background layers ever sync up. Moods carry a gentle 0.5 s body blend
    # (unlike the neutral idles' 0.0) so swapping mood eases rather than snaps.
    #
    # requires_mode is "stand" (i.e. standing OR docked, not walking), NOT "any":
    # a mood is an ambient emotional STATE you sit in at rest. The neutral idles
    # stay "any" as the always-on baseline; while walking the gait swamps the
    # subtle mood motion (the phase-4 head-follow check confirmed the two subtlest
    # moods do not read over a gait) and walk_look_around (priority 5) overlays
    # instead. Marking moods "stand" is honest about where they actually read.

    # M1) Content / happy: higher head carriage (chin up), light quick rhythm,
    # small bright wander, a slight perky persistent tilt, frequent quick blinks.
    d = 6.5
    f = 1.0 / d
    specs.append(ClipSpec(
        name="mood_content", duration_s=d, loop_mode="wrap", requires_mode="stand",
        priority=0, blend_in_s=0.5, blend_out_s=0.5, show_blend_in_s=0.1,
        show_blend_out_s=0.1,
        doc="Content/happy mood loop: high bright carriage, light quick rhythm, slow content blinks.",
        neck_pitch=sine(3 * f, 0.024, 0.0),               # light quick breath
        head_pitch=const(-0.10) + sine(3 * f, 0.018, np.pi),  # chin-up carriage
        head_yaw=drift((f, 0.085, 0.3), (2 * f, 0.05, 1.7), (3 * f, 0.028, 0.9)),
        head_roll=const(0.03) + drift((f, 0.045, 1.2), (2 * f, 0.024, 0.1)),
        antenna_l=ZERO, antenna_r=ZERO,
        # Contentment reads as slow, relaxed lid closes (a real heavy blink on
        # the LEDs via the slow_blink cue), not the crisp idle flick.
        events=(("eye", "slow_blink", 1.6), ("eye", "slow_blink", 4.4)),
    ))

    # M2) Sad / dejected: low carriage (chin down), slow, a slight PERSISTENT
    # roll-tilt, long pauses, little looking around, one slow heavy blink.
    d = 9.5
    f = 1.0 / d
    specs.append(ClipSpec(
        name="mood_sad", duration_s=d, loop_mode="wrap", requires_mode="stand",
        priority=0, blend_in_s=0.6, blend_out_s=0.7, show_blend_in_s=0.1,
        show_blend_out_s=0.1,
        doc="Sad/dejected mood loop: low slow carriage, persistent tilt, heavy blink.",
        neck_pitch=const(0.035) + sine(f, 0.012, 0.0),    # tiny sink (neck maxed)
        head_pitch=const(0.13) + sine(f, 0.018, np.pi),   # sunk carriage + sigh
        head_roll=const(0.07) + sine(f, 0.018, 0.5),      # persistent lean
        head_yaw=drift((f, 0.05, 0.2), (2 * f, 0.02, 1.0)),  # barely looks around
        antenna_l=ZERO, antenna_r=ZERO,
        # One slow heavy blink — a genuine long lid close on the LEDs via the
        # slow_blink cue (the per-frame track only ever rendered a crisp flick).
        events=(("eye", "slow_blink", 4.0),),
    ))

    # M3) Sleepy / drowsy: very slow drift, head gradually settling then rousing
    # slightly, a slow side loll, long droopy eye closes.
    d = 12.0
    f = 1.0 / d
    specs.append(ClipSpec(
        name="mood_sleepy", duration_s=d, loop_mode="wrap", requires_mode="stand",
        priority=0, blend_in_s=0.6, blend_out_s=0.7, show_blend_in_s=0.1,
        show_blend_out_s=0.1,
        doc="Sleepy/drowsy mood loop: head settles then rouses, long droopy blinks.",
        neck_pitch=const(0.03) + sine(f, 0.014, 0.0),
        head_pitch=keys([(0.0, 0.05), (5.0, 0.15, "ease_out"), (7.5, 0.14, "hold"),
                         (9.5, 0.07, "smooth")], loop=True, duration=d),  # settle->rouse
        head_roll=const(0.03) + sine(f, 0.05, 0.3),       # slow heavy loll
        head_yaw=sine(f, 0.04, 1.0),                      # almost still
        antenna_l=ZERO, antenna_r=ZERO,
        # Long droopy closes rendered as real heavy blinks via the slow_blink cue.
        events=(("eye", "slow_blink", 2.2), ("eye", "slow_blink", 7.5)),
    ))

    # M4) Alert / attentive: upright, still, small SHARP scans with long stillness
    # between (energy in sharpness, not amplitude), rare blink.
    d = 7.5
    f = 1.0 / d
    specs.append(ClipSpec(
        name="mood_alert", duration_s=d, loop_mode="wrap", requires_mode="stand",
        priority=0, blend_in_s=0.3, blend_out_s=0.4, show_blend_in_s=0.1,
        show_blend_out_s=0.1,
        doc="Alert/attentive mood loop: upright and still with small sharp scans.",
        neck_pitch=sine(2 * f, 0.010, 0.0),
        head_pitch=const(-0.06) + sine(2 * f, 0.010, np.pi),  # attentive lift
        head_yaw=keys([(0.0, 0.0), (1.2, 0.16, "ease_out"), (1.6, 0.16, "hold"),
                       (2.0, 0.0, "ease_in"), (4.6, 0.0, "hold"),
                       (5.2, -0.14, "ease_out"), (5.6, -0.14, "hold"),
                       (6.1, 0.0, "ease_in")], loop=True, duration=d),  # sharp scans
        head_roll=sine(2 * f, 0.014, 0.7),
        antenna_l=ZERO, antenna_r=ZERO,
        eyes=((0.0, 1), (3.6, 0), (3.68, 1)),             # one crisp blink, wide otherwise
    ))

    # M5) Grumpy / annoyed: a persistent COCKED tilt (opposite sense to sad),
    # slightly lowered carriage, terse sharp dismissive turn-aways, terse blinks.
    d = 8.5
    f = 1.0 / d
    specs.append(ClipSpec(
        name="mood_grumpy", duration_s=d, loop_mode="wrap", requires_mode="stand",
        priority=0, blend_in_s=0.4, blend_out_s=0.5, show_blend_in_s=0.1,
        show_blend_out_s=0.1,
        doc="Grumpy/annoyed mood loop: cocked tilt, low carriage, terse turn-aways.",
        neck_pitch=const(0.02) + sine(f, 0.010, 0.0),
        head_pitch=const(0.06) + sine(f, 0.014, np.pi),   # lowered brow
        head_roll=const(-0.09) + sine(f, 0.014, 0.4),     # persistent cock
        head_yaw=keys([(0.0, 0.0), (2.2, 0.0, "hold"), (2.6, -0.14, "ease_out"),
                       (3.4, -0.12, "hold"), (4.4, 0.0, "smooth"),
                       (8.5, 0.0, "hold")], loop=True, duration=d),  # terse turn-away
        antenna_l=ZERO, antenna_r=ZERO,
        eyes=((0.0, 1), (1.5, 0), (1.7, 1), (5.6, 0), (5.8, 1)),  # terse blinks
    ))

    # M6) Scared / frightened: fear as a STATE, not the startle spike. Held small
    # and withdrawn (chin slightly down), very STILL with a slow uneasy sway and a
    # tiny fast tense micro-tremor (on the sub-follow-floor pitch/neck so it reads
    # as tension without asking the servo to track it), punctuated by two quick
    # darting glances with long frozen stillness between — stillness broken by
    # sharp checks reads as fear far better than continuous motion. Fear is
    # wide-eyed with blinking SUPPRESSED (the sustained wide/fear eye hold), not
    # a per-frame blink track. Antennas ZERO (loop rule).
    # Period 10.0 s: unused by any other background layer so nothing ever syncs.
    d = 10.0
    f = 1.0 / d
    specs.append(ClipSpec(
        name="mood_scared", duration_s=d, loop_mode="wrap", requires_mode="stand",
        priority=0, blend_in_s=0.4, blend_out_s=0.5, show_blend_in_s=0.1,
        show_blend_out_s=0.1,
        doc="Scared/frightened mood loop: held small and wary, tense stillness, darting checks.",
        neck_pitch=const(0.03) + sine(10 * f, 0.006, 1.0),   # slight withdraw + 1 Hz tense tremor
        head_pitch=const(0.06) + sine(10 * f, 0.008, 0.0),   # withdrawn carriage + tense micro-tremor
        head_roll=sine(f, 0.014, 0.0),                       # a slow small uneasy sway
        head_yaw=keys([(0.0, 0.0),
                       (2.0, 0.0, "hold"),
                       (2.3, 0.13, "ease_out"),              # a quick dart to check
                       (2.6, 0.12, "hold"),
                       (2.95, 0.0, "smooth"),                # back to frozen
                       (5.5, 0.0, "hold"),
                       (5.8, -0.11, "ease_out"),             # a dart the other way
                       (6.1, -0.10, "hold"),
                       (6.45, 0.0, "smooth"),
                       (10.0, 0.0, "hold")], loop=True, duration=d),
        antenna_l=ZERO, antenna_r=ZERO,
        # Enter the sustained wide/fear hold (eyes wide, blinking suppressed) and
        # refresh it within the 10 s loop — each fire is well under the eyes'
        # safety timeout, so the hold stays continuous while the mood plays. When
        # the mood ends the hold auto-releases via that timeout (or a calm_down
        # releases it explicitly with a relief burst).
        events=(("eye", "fear", 0.0), ("eye", "fear", 5.0)),
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

    # 9) Yes-nod: an EMPHATIC, legible "yes" built the right way — from head_pitch,
    # which has ~4x the headroom of the (maxed) neck_pitch axis. A real nod is not
    # a symmetric sine: it has a small anticipation lift, a SHARP down-beat
    # ("ease_out" = quick chin-drop that decelerates into the pose), a gentler
    # recovery, and 3 nods of decreasing amplitude that settle. neck_pitch adds
    # only a hint of full-body bob (it is the binding axis, so it stays small).
    specs.append(ClipSpec(
        name="nod_yes", authoring_path="blender", duration_s=2.2, loop_mode="once", requires_mode="any",
        priority=20, blend_in_s=0.15, blend_out_s=0.25,
        doc="Emphatic affirmative nod (yes): anticipation + a sharp down-beat, 3 decaying nods.",
        head_pitch=keys([(0.0, 0.0),
                         (0.20, -0.05, "ease_out"),   # anticipation: a small lift first
                         (0.46, 0.24, "ease_out"),    # nod 1: sharp chin-down (the accent)
                         (0.74, 0.01, "smooth"),      # recovery (gentler than the down-beat)
                         (1.00, 0.16, "ease_out"),    # nod 2: down, smaller
                         (1.26, 0.01, "smooth"),      # recovery
                         (1.50, 0.09, "ease_out"),    # nod 3: down, smallest
                         (1.80, 0.0, "smooth"),       # settle to rest
                         (2.2, 0.0, "hold")]),
        neck_pitch=keys([(0.0, 0.0),
                         (0.46, 0.035, "ease_out"),   # a hint of full-body bob (neck is maxed: tiny)
                         (0.74, 0.0, "smooth"),
                         (1.00, 0.02, "ease_out"),
                         (1.26, 0.0, "smooth"),
                         (2.2, 0.0, "hold")]),
        head_roll=sine(0.7, 0.02, 0.0),               # a touch of asymmetry: not a dead-straight machine nod
        antenna_l=(pulse(0.46, 0.5, 0.14) + pulse(1.0, 0.5, 0.10)),  # small crisp bob with the nods
        antenna_r=(pulse(0.46, 0.5, 0.12) + pulse(1.0, 0.5, 0.09)),
    ))

    # 9b) Soft yes: a small, polite acknowledging nod — a single gentle dip with a
    # tiny second beat. A genuinely different message from the emphatic yes: "noted"
    # rather than "YES". Small enough for any mode.
    specs.append(ClipSpec(
        name="nod_yes_soft", authoring_path="blender", duration_s=1.6, loop_mode="once", requires_mode="any",
        priority=13, blend_in_s=0.15, blend_out_s=0.3,
        doc="Soft polite acknowledging nod: a single gentle dip. 'Noted', not 'YES'.",
        head_pitch=keys([(0.0, 0.0),
                         (0.30, 0.10, "ease_out"),    # one gentle nod down
                         (0.70, 0.01, "smooth"),
                         (1.00, 0.05, "ease_out"),    # a tiny second beat
                         (1.55, 0.0, "smooth")]),
        neck_pitch=keys([(0.0, 0.0), (0.30, 0.02, "ease_out"), (0.70, 0.0, "smooth"),
                         (1.6, 0.0, "hold")]),
        head_roll=sine(0.9, 0.015, 0.0),
        antenna_l=pulse(0.30, 0.4, 0.06),             # a single tiny crisp bob
        antenna_r=pulse(0.30, 0.4, 0.05),
    ))

    # 10) No-shake: a DECISIVE, legible "no" built from head_yaw (which has huge
    # headroom). A small wind-up one way (anticipation), then firm alternating
    # swings that decay and settle centred. Amplitude (0.42) is well up from the
    # old 0.34 for an unambiguous read, but the swing rate is kept ~1.3 Hz so the
    # soft kp=8 head servo can still track it (fast oscillation would lag). A hair
    # of counter-roll keeps it from reading mechanical. Distinct from idle scans:
    # bigger, faster, alternating, and it returns dead-centre.
    specs.append(ClipSpec(
        name="shake_no", authoring_path="blender", duration_s=2.5, loop_mode="once", requires_mode="stand",
        priority=20, blend_in_s=0.15, blend_out_s=0.3,
        doc="Decisive negative shake (no): wind-up + firm alternating swings that decay.",
        head_yaw=keys([(0.0, 0.0),
                       (0.25, -0.10, "ease_out"),     # anticipation: a small wind-up
                       (0.58, 0.42, "ease_out"),      # swing 1 (decisive)
                       (0.98, -0.38, "smooth"),       # swing 2
                       (1.35, 0.26, "smooth"),        # swing 3, decaying
                       (1.70, -0.16, "smooth"),       # swing 4
                       (2.00, 0.06, "smooth"),        # settle swing
                       (2.30, 0.0, "smooth"),         # dead centre
                       (2.5, 0.0, "hold")]),
        head_roll=sine(1.2, 0.03, 0.0),               # slight natural counter-roll
        antenna_l=(pulse(0.75, 0.5, 0.12) + pulse(1.35, 0.5, -0.08)),  # trailing follow-through
        antenna_r=(pulse(0.75, 0.5, 0.10) + pulse(1.35, 0.5, -0.06)),
    ))

    # 10b) Reluctant no: a slower, smaller shake with the chin sinking (aversion)
    # and a slight persistent downward tilt. A hesitant "...no", genuinely
    # different from the firm refusal above.
    specs.append(ClipSpec(
        name="shake_no_reluctant", authoring_path="blender", duration_s=2.8, loop_mode="once", requires_mode="stand",
        priority=13, blend_in_s=0.25, blend_out_s=0.4,
        doc="Reluctant/hesitant no: a slow small shake with the chin dropping in aversion.",
        head_yaw=keys([(0.0, 0.0),
                       (0.60, -0.17, "ease_out"),     # a slow, small turn away
                       (1.35, 0.13, "smooth"),        # slow return past centre
                       (2.00, -0.09, "smooth"),       # a smaller second turn
                       (2.8, 0.0, "smooth")]),
        head_pitch=keys([(0.0, 0.0),
                         (0.70, 0.11, "ease_out"),     # chin sinks: reluctance / aversion
                         (2.00, 0.10, "hold"),
                         (2.8, 0.0, "smooth")]),
        head_roll=keys([(0.0, 0.0), (0.70, 0.05, "ease_out"), (2.00, 0.05, "hold"),
                        (2.8, 0.0, "smooth")]),        # a slight held tilt
        antenna_l=keys([(0.0, 0.0), (0.80, -0.20, "ease_out"), (2.00, -0.18, "hold"),
                        (2.8, 0.0, "smooth")]),        # a reluctant half-fold
        antenna_r=keys([(0.0, 0.0), (0.80, -0.20, "ease_out"), (2.00, -0.18, "hold"),
                        (2.8, 0.0, "smooth")]),
        eyes=((0.0, 1), (1.0, 0), (1.3, 1)),          # a slow reluctant blink
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

    # ---- C2. Emotional one-shot beats (once) --------------------------------
    # A genuine range of triggered feelings. Distinguished from each other by
    # ENERGY and TIMING as much as direction (happy vs grumpy use similar
    # amplitudes; sharpness/rhythm/recovery separate them). Built from head
    # ROLL/PITCH/YAW/timing/eyes — neck_pitch stays tiny (it is maxed). Antennas
    # ARE used here: a brief crisp flick/fold reads as ears and is a momentary
    # gesture, not the sustained loop buzz the owner objected to — kept short and
    # under the slew cap. Priorities slot above the mood loops (0) so any of
    # these preempts the current mood; startle (30) still preempts all.

    # E1) Excited / delighted: high-energy sharp triple bob + wiggle + rapid
    # double blink + bright antenna flicks. Happy_bounce's louder cousin.
    specs.append(ClipSpec(
        name="excited", authoring_path="blender", duration_s=2.2, loop_mode="once",
        requires_mode="stand", priority=22, blend_in_s=0.1, blend_out_s=0.3,
        doc="Excited/delighted: sharp triple bob, quick wiggle, rapid double blink.",
        head_pitch=(pulse(0.35, 0.26, -0.11) + pulse(0.75, 0.26, -0.10)
                    + pulse(1.15, 0.26, -0.08)),          # bright chin-up bobs
        neck_pitch=(pulse(0.35, 0.28, -0.05) + pulse(0.75, 0.28, -0.045)
                    + pulse(1.15, 0.28, -0.04)),
        head_yaw=(pulse(0.55, 0.22, 0.14) + pulse(0.95, 0.22, -0.14)
                  + pulse(1.35, 0.22, 0.10)),             # quick side-to-side
        head_roll=pulse(1.55, 0.4, 0.09),                 # a bright tilt to finish
        antenna_l=(pulse(0.35, 0.24, 0.42) + pulse(0.8, 0.24, 0.42)
                   + pulse(1.25, 0.24, 0.34)),
        antenna_r=(pulse(0.35, 0.24, 0.42) + pulse(0.8, 0.24, 0.42)
                   + pulse(1.25, 0.24, 0.34)),
        events=(("eye", "happy", 0.35), ("eye", "happy", 1.15)),  # rapid double
    ))

    # E2) Grumpy / annoyed: one sharp dismissive turn-away with a cocked tilt and
    # a terse antenna fold, then a grudging settle. Terse, not an oscillation.
    specs.append(ClipSpec(
        name="grumpy_annoyed", authoring_path="blender", duration_s=2.0,
        loop_mode="once", requires_mode="stand", priority=21, blend_in_s=0.12,
        blend_out_s=0.3,
        doc="Grumpy/annoyed: sharp cocked turn-away with a terse antenna fold.",
        head_yaw=keys([(0.0, 0.0), (0.35, -0.28, "ease_out"), (1.0, -0.26, "hold"),
                       (2.0, 0.0, "smooth")]),
        head_roll=keys([(0.0, 0.0), (0.4, -0.10, "ease_out"), (1.0, -0.09, "hold"),
                        (2.0, 0.0, "smooth")]),           # cocked "hmph"
        head_pitch=keys([(0.0, 0.0), (0.4, 0.06, "ease_out"), (1.0, 0.055, "hold"),
                         (2.0, 0.0, "smooth")]),
        neck_pitch=pulse(0.5, 0.4, 0.03),
        antenna_l=keys([(0.0, 0.0), (0.3, -0.28, "ease_out"), (1.2, -0.26, "hold"),
                        (2.0, 0.0)]),                     # fold back, annoyed
        antenna_r=keys([(0.0, 0.0), (0.3, -0.28, "ease_out"), (1.2, -0.26, "hold"),
                        (2.0, 0.0)]),
        events=(("eye", "blink", 0.35),),
    ))

    # E3) Confused / puzzled: the classic quizzical DOUBLE tilt — roll one way,
    # hold, roll the OTHER way, hold — with asymmetric antennas (one up, one
    # down) and a slow blink. Leans hard on the roll channel.
    specs.append(ClipSpec(
        name="confused_puzzled", authoring_path="blender", duration_s=3.0,
        loop_mode="once", requires_mode="any", priority=12, blend_in_s=0.25,
        blend_out_s=0.35,
        doc="Confused/puzzled: quizzical double head-tilt with asymmetric antennas.",
        head_roll=keys([(0.0, 0.0), (0.7, 0.15, "ease_out"), (1.3, 0.15, "hold"),
                        (1.9, -0.13, "smooth"), (2.4, -0.13, "hold"),
                        (3.0, 0.0, "smooth")]),
        head_yaw=keys([(0.0, 0.0), (0.7, 0.08, "ease_out"), (1.9, -0.07, "smooth"),
                       (3.0, 0.0, "smooth")]),
        head_pitch=keys([(0.0, 0.0), (0.7, 0.04, "ease_out"), (3.0, 0.0)]),
        neck_pitch=pulse(0.8, 0.9, -0.025),
        antenna_l=keys([(0.0, 0.0), (0.7, 0.22, "ease_out"), (2.4, 0.15, "hold"),
                        (3.0, 0.0)]),                     # one ear up ...
        antenna_r=keys([(0.0, 0.0), (0.7, -0.18, "ease_out"), (2.4, -0.12, "hold"),
                        (3.0, 0.0)]),                     # ... one ear down (quizzical)
        eyes=((0.0, 1), (1.4, 0), (1.78, 1)),             # slow blink at the switch
    ))

    # E4) Proud / pleased: a dignified slow puff-up — chin up, neck lifts,
    # antennas raised and held, a slow content blink. Slow and held, not a snap.
    specs.append(ClipSpec(
        name="proud_pleased", authoring_path="blender", duration_s=2.6,
        loop_mode="once", requires_mode="stand", priority=18, blend_in_s=0.25,
        blend_out_s=0.4,
        doc="Proud/pleased: dignified slow chest-puff, chin up, antennas raised.",
        head_pitch=keys([(0.0, 0.0), (0.8, -0.15, "ease_out"), (1.8, -0.14, "hold"),
                         (2.6, 0.0, "smooth")]),
        neck_pitch=keys([(0.0, 0.0), (0.8, -0.05, "ease_out"), (1.8, -0.05, "hold"),
                         (2.6, 0.0)]),
        head_roll=keys([(0.0, 0.0), (1.0, 0.05, "ease_out"), (1.8, 0.05, "hold"),
                        (2.6, 0.0)]),
        head_yaw=pulse(1.3, 0.6, 0.08),                   # a slow survey
        antenna_l=keys([(0.0, 0.0), (0.7, 0.32, "ease_out"), (1.9, 0.30, "hold"),
                        (2.6, 0.05)]),
        antenna_r=keys([(0.0, 0.0), (0.7, 0.32, "ease_out"), (1.9, 0.30, "hold"),
                        (2.6, 0.05)]),
        eyes=((0.0, 1), (1.5, 0), (1.9, 1)),              # slow content blink
    ))

    # E5) Timid / shy: shrink back and away — lower, turn away with a tilt,
    # antennas fold, then a small shy peek back. Lateral, unlike sad's frontal sink.
    specs.append(ClipSpec(
        name="timid_shy", authoring_path="blender", duration_s=3.0, loop_mode="once",
        requires_mode="stand", priority=16, blend_in_s=0.3, blend_out_s=0.45,
        doc="Timid/shy: shrink and turn away with a tilt, fold antennas, peek back.",
        head_yaw=keys([(0.0, 0.0), (0.9, -0.28, "ease_out"), (1.6, -0.26, "hold"),
                       (2.1, -0.14, "smooth"), (2.5, -0.20, "smooth"),
                       (3.0, 0.0, "smooth")]),            # turn away, tiny peek back
        head_pitch=keys([(0.0, 0.0), (0.9, 0.09, "ease_out"), (2.2, 0.08, "hold"),
                         (3.0, 0.0, "smooth")]),
        head_roll=keys([(0.0, 0.0), (0.9, -0.10, "ease_out"), (2.2, -0.09, "hold"),
                        (3.0, 0.0, "smooth")]),           # tilt into the shoulder
        neck_pitch=pulse(1.1, 1.0, 0.03),
        antenna_l=keys([(0.0, 0.0), (0.8, -0.26, "ease_out"), (2.2, -0.24, "hold"),
                        (3.0, 0.0)]),
        antenna_r=keys([(0.0, 0.0), (0.8, -0.26, "ease_out"), (2.2, -0.24, "hold"),
                        (3.0, 0.0)]),
        eyes=((0.0, 1), (1.0, 0), (1.4, 1)),
    ))

    # E6) Disappointed: the let-down — a small hopeful lift, then a slow sink
    # with a sigh and a turn-away. The anticipation beat separates it from sad.
    specs.append(ClipSpec(
        name="disappointed", authoring_path="blender", duration_s=3.0,
        loop_mode="once", requires_mode="stand", priority=16, blend_in_s=0.3,
        blend_out_s=0.5,
        doc="Disappointed: a hopeful lift then a slow let-down sink with a sigh.",
        head_pitch=keys([(0.0, 0.0), (0.5, -0.08, "ease_out"), (0.9, -0.07, "hold"),
                         (1.8, 0.14, "smooth"), (2.5, 0.12, "hold"),
                         (3.0, 0.0, "smooth")]),          # hope up -> sink down
        neck_pitch=keys([(0.0, 0.0), (0.5, -0.03), (1.8, 0.05, "smooth"), (3.0, 0.0)]),
        head_roll=keys([(0.0, 0.0), (1.8, 0.06, "ease_out"), (3.0, 0.0, "smooth")]),
        head_yaw=pulse(2.1, 0.7, -0.08),                  # turn away as it sinks
        antenna_l=keys([(0.0, 0.0), (0.5, 0.14, "ease_out"), (0.9, 0.12, "hold"),
                        (1.9, -0.22, "smooth"), (2.6, -0.20, "hold"), (3.0, 0.0)]),
        antenna_r=keys([(0.0, 0.0), (0.5, 0.14, "ease_out"), (0.9, 0.12, "hold"),
                        (1.9, -0.22, "smooth"), (2.6, -0.20, "hold"), (3.0, 0.0)]),
        eyes=((0.0, 1), (1.8, 0), (2.2, 1)),              # slow blink on the sink
    ))

    # E7) Suspicious / wary: a cocked tilt held, a slight lean-in, a slow narrow
    # scan, wary half-back antennas. Suspicion lives in the sustained cock.
    specs.append(ClipSpec(
        name="suspicious_wary", authoring_path="blender", duration_s=3.4,
        loop_mode="once", requires_mode="stand", priority=14, blend_in_s=0.3,
        blend_out_s=0.4,
        doc="Suspicious/wary: cocked tilt, lean-in, slow narrow scan, wary antennas.",
        head_roll=keys([(0.0, 0.0), (0.8, 0.11, "ease_out"), (2.6, 0.11, "hold"),
                        (3.4, 0.0, "smooth")]),           # held cock
        head_yaw=keys([(0.0, 0.0), (1.2, -0.18, "ease_out"), (1.8, -0.17, "hold"),
                       (2.8, 0.14, "smooth"), (3.4, 0.0, "smooth")]),  # narrow scan
        neck_pitch=keys([(0.0, 0.0), (0.8, 0.045, "ease_out"), (2.6, 0.045, "hold"),
                         (3.4, 0.0)]),                    # slight lean-in
        head_pitch=keys([(0.0, 0.0), (0.8, 0.05, "ease_out"), (2.6, 0.05, "hold"),
                         (3.4, 0.0)]),
        antenna_l=keys([(0.0, 0.0), (0.7, -0.18, "ease_out"), (2.6, -0.16, "hold"),
                        (3.4, 0.0)]),
        antenna_r=keys([(0.0, 0.0), (0.7, -0.18, "ease_out"), (2.6, -0.16, "hold"),
                        (3.4, 0.0)]),
        eyes=((0.0, 1), (1.0, 0), (1.5, 1)),
    ))

    # E8) Sleepy yawn: a big slow stretch — head tilts back and up, a long eye
    # close (the yawn), antennas stretch up then flop, then a drowsy settle down.
    specs.append(ClipSpec(
        name="sleepy_yawn", authoring_path="blender", duration_s=3.6, loop_mode="once",
        requires_mode="stand", priority=17, blend_in_s=0.3, blend_out_s=0.5,
        doc="Sleepy yawn: a big slow back-and-up stretch, long eye close, settle.",
        head_pitch=keys([(0.0, 0.0), (1.0, -0.16, "ease_out"), (1.6, -0.15, "hold"),
                         (2.6, 0.12, "smooth"), (3.1, 0.10, "hold"),
                         (3.6, 0.0, "smooth")]),          # stretch back -> settle down
        neck_pitch=keys([(0.0, 0.0), (1.0, -0.05, "ease_out"), (2.6, 0.04, "smooth"),
                         (3.6, 0.0)]),
        head_roll=keys([(0.0, 0.0), (1.3, 0.07, "ease_out"), (2.6, 0.05, "smooth"),
                        (3.6, 0.0)]),                     # slow loll
        antenna_l=keys([(0.0, 0.0), (1.0, 0.28, "ease_out"), (1.6, 0.26, "hold"),
                        (2.6, -0.10, "smooth"), (3.6, 0.0)]),  # stretch then flop
        antenna_r=keys([(0.0, 0.0), (1.0, 0.28, "ease_out"), (1.6, 0.26, "hold"),
                        (2.6, -0.10, "smooth"), (3.6, 0.0)]),
        # The long yawn eye-close, as a real heavy lid close via the slow_blink
        # cue (on binary LEDs the close duration is the cue's ~0.55 s, not the
        # authored 1.4 s, but it is a genuine slow blink, not a crisp flick).
        events=(("eye", "slow_blink", 0.8),),
    ))

    # E9) Affectionate: a warm lean-in nuzzle — tilt and lean toward, a soft bob,
    # relaxed antennas forward, a soft slow content blink.
    specs.append(ClipSpec(
        name="affectionate", authoring_path="blender", duration_s=2.8, loop_mode="once",
        requires_mode="any", priority=14, blend_in_s=0.3, blend_out_s=0.4,
        doc="Affectionate: a warm tilt-lean nuzzle with a soft bob and slow blink.",
        head_roll=keys([(0.0, 0.0), (0.8, 0.12, "ease_out"), (1.8, 0.11, "hold"),
                        (2.8, 0.0, "smooth")]),           # warm tilt-lean
        neck_pitch=keys([(0.0, 0.0), (0.8, 0.05, "ease_out"), (1.4, 0.02, "smooth"),
                         (2.0, 0.05, "smooth"), (2.8, 0.0)]),  # soft nuzzle bob
        head_pitch=pulse(1.4, 0.6, 0.06),
        head_yaw=pulse(1.0, 0.9, 0.06),                   # lean toward
        antenna_l=keys([(0.0, 0.0), (0.9, 0.16, "ease_out"), (2.0, 0.14, "hold"),
                        (2.8, 0.0)]),
        antenna_r=keys([(0.0, 0.0), (0.9, 0.16, "ease_out"), (2.0, 0.14, "hold"),
                        (2.8, 0.0)]),
        eyes=((0.0, 1), (1.3, 0), (1.7, 1)),
    ))

    # E10) Flustered / embarrassed: a quick wavering look-away + a downward tuck,
    # with the RAPID "fluster" energy carried by the antennas + a rapid double
    # blink (both can move fast; the soft kp=8 head servo cannot track fast head
    # jitter, so the head motion is kept clean and the flutter lives in the ears
    # and eyes — a deliberate craft choice, not a compromise).
    specs.append(ClipSpec(
        name="flustered", authoring_path="blender", duration_s=2.2, loop_mode="once",
        requires_mode="stand", priority=20, blend_in_s=0.1, blend_out_s=0.25,
        doc="Flustered/embarrassed: wavering look-away + tuck, rapid antenna+eye flutter.",
        head_yaw=keys([(0.0, 0.0), (0.5, -0.17, "ease_out"), (0.9, -0.14, "hold"),
                       (1.3, -0.06, "smooth"), (1.7, -0.12, "smooth"),
                       (2.2, 0.0, "smooth")]),            # a wavering look-away
        head_pitch=keys([(0.0, 0.0), (0.5, 0.07, "ease_out"), (1.4, 0.06, "hold"),
                         (2.2, 0.0, "smooth")]),          # embarrassed look-down tuck
        head_roll=(pulse(0.6, 0.45, -0.05) + pulse(1.4, 0.45, 0.05)),  # small tilt waver
        neck_pitch=pulse(0.9, 0.8, 0.025),
        antenna_l=(pulse(0.4, 0.2, 0.30) + pulse(0.8, 0.2, 0.30)
                   + pulse(1.2, 0.2, -0.20)),             # rapid ear flutter
        antenna_r=(pulse(0.4, 0.2, 0.30) + pulse(0.8, 0.2, 0.30)
                   + pulse(1.2, 0.2, -0.20)),
        events=(("eye", "double", 0.4),),                 # rapid double blink
    ))

    # E11) Content sigh: a relaxed exhale — a gentle lift then a slow soft settle
    # with a soft loll and a soft blink. A low-energy positive beat.
    specs.append(ClipSpec(
        name="content_sigh", authoring_path="blender", duration_s=2.8, loop_mode="once",
        requires_mode="any", priority=12, blend_in_s=0.3, blend_out_s=0.45,
        doc="Content sigh: a gentle lift then a slow relaxed exhale settle.",
        head_pitch=keys([(0.0, 0.0), (0.7, -0.07, "ease_out"), (1.2, -0.06, "hold"),
                         (2.2, 0.05, "smooth"), (2.8, 0.0, "smooth")]),  # inhale->exhale
        neck_pitch=keys([(0.0, 0.0), (0.7, -0.03), (2.2, 0.03, "smooth"), (2.8, 0.0)]),
        head_roll=keys([(0.0, 0.0), (1.4, 0.06, "ease_out"), (2.8, 0.0, "smooth")]),
        antenna_l=keys([(0.0, 0.0), (0.8, 0.10, "ease_out"), (2.0, 0.08, "hold"),
                        (2.8, 0.0)]),
        antenna_r=keys([(0.0, 0.0), (0.8, 0.10, "ease_out"), (2.0, 0.08, "hold"),
                        (2.8, 0.0)]),
        events=(("eye", "slow_blink", 1.2),),             # a soft slow content blink
    ))

    # E12) Greeting: a friendly "hello" — a warm double bob with a tilt, antennas
    # raised bright, a happy double blink. Social, distinct from nod_yes's assent.
    specs.append(ClipSpec(
        name="greeting", authoring_path="blender", duration_s=2.4, loop_mode="once",
        requires_mode="stand", priority=18, blend_in_s=0.12, blend_out_s=0.3,
        doc="Greeting: a friendly double bob with a tilt and a bright antenna raise.",
        head_pitch=(pulse(0.45, 0.35, 0.12) + pulse(1.05, 0.35, 0.10)),  # warm nods
        neck_pitch=(pulse(0.45, 0.4, 0.04) + pulse(1.05, 0.4, 0.035)),
        head_roll=pulse(1.3, 0.5, 0.09),                  # friendly tilt
        head_yaw=pulse(0.7, 0.9, 0.06),
        antenna_l=keys([(0.0, 0.0), (0.4, 0.30, "ease_out"), (1.4, 0.26, "hold"),
                        (2.4, 0.05)]),
        antenna_r=keys([(0.0, 0.0), (0.4, 0.30, "ease_out"), (1.4, 0.26, "hold"),
                        (2.4, 0.05)]),
        events=(("eye", "happy", 0.45),),                 # friendly double blink
    ))

    # ---- CS. Scared / fear one-shot beats -----------------------------------
    #
    # "Acting scared" is distinct from `startle` (a 1.6 s bidirectional spike).
    # These are fear beats you enter and — crucially — a `calm_down` that exits
    # fear back to neutral, so the emotion never looks stuck. Antennas are hugely
    # useful here: a fast pin-BACK (ears flattened) reads unmistakably as fear, and
    # a slow un-pin reads as the tension leaving. Eyes go WIDE (the runtime `wide`
    # event holds them open ~1 s). All are stand-only: the poses are withdrawn and
    # large enough that they don't belong over a gait.

    # S1) Flinch: a fast aversive recoil — head snaps back and AVERTS to the side,
    # ears pin back, eyes snap wide — then a SLOW, tentative return (the recovery
    # is what separates a flinch from a startle: it comes back warily, not briskly).
    specs.append(ClipSpec(
        name="flinch", authoring_path="blender", duration_s=2.4, loop_mode="once",
        requires_mode="stand", priority=26, blend_in_s=0.06, blend_out_s=0.4,
        doc="Flinch: a fast aversive recoil away, then a slow tentative return.",
        head_pitch=keys([(0.0, 0.0), (0.14, -0.09, "ease_out"), (0.5, -0.05, "smooth"),
                         (1.3, 0.03, "smooth"), (2.4, 0.0, "smooth")]),   # quick pull back, slow settle
        head_yaw=keys([(0.0, 0.0), (0.15, -0.22, "ease_out"), (0.55, -0.17, "smooth"),
                       (1.5, -0.05, "smooth"), (2.4, 0.0, "smooth")]),     # avert away fast, return slow
        head_roll=keys([(0.0, 0.0), (0.17, -0.09, "ease_out"), (0.6, -0.07, "smooth"),
                        (1.6, -0.02, "smooth"), (2.4, 0.0, "smooth")]),    # flinch tilt away
        neck_pitch=keys([(0.0, 0.0), (0.15, -0.05, "ease_out"), (0.5, -0.03, "smooth"),
                         (2.4, 0.0, "smooth")]),                            # small recoil back
        antenna_l=keys([(0.0, 0.0), (0.26, -0.48, "smooth"), (1.2, -0.40, "hold"),
                        (2.4, -0.03, "smooth")]),                          # ears pin back fast, slow release
        antenna_r=keys([(0.0, 0.0), (0.26, -0.48, "smooth"), (1.2, -0.40, "hold"),
                        (2.4, -0.03, "smooth")]),
        events=(("eye", "fear", 0.15), ("eye", "release", 1.9)),           # snap wide (held), relief on the return
    ))

    # S2) Cower: shrink and make itself small — tuck the head down, turn and tilt
    # away, ears pinned back and HELD, eyes wide. A sustained fear pose (not a
    # spike). A tiny tense micro-shift during the hold keeps it from freezing dead.
    specs.append(ClipSpec(
        name="cower", authoring_path="blender", duration_s=3.0, loop_mode="once",
        requires_mode="stand", priority=25, blend_in_s=0.12, blend_out_s=0.5,
        doc="Cower: shrink small — head tucked and turned away, ears pinned, eyes wide, held.",
        head_pitch=keys([(0.0, 0.0), (0.4, 0.14, "ease_out"), (2.4, 0.13, "hold"),
                         (3.0, 0.0, "smooth")]),                            # tuck down, hold small
        neck_pitch=keys([(0.0, 0.0), (0.4, 0.045, "ease_out"), (2.4, 0.045, "hold"),
                         (3.0, 0.0, "smooth")]),
        head_roll=keys([(0.0, 0.0), (0.45, -0.09, "ease_out"), (2.4, -0.085, "hold"),
                        (3.0, 0.0, "smooth")]),                            # withdrawn tilt away, held
        head_yaw=keys([(0.0, 0.0), (0.45, -0.11, "ease_out"), (1.2, -0.10, "hold"),
                       (1.8, -0.13, "smooth"), (2.4, -0.10, "smooth"),
                       (3.0, 0.0, "smooth")]),                             # turned away + a tense micro-check
        antenna_l=keys([(0.0, 0.0), (0.3, -0.45, "smooth"), (2.4, -0.42, "hold"),
                        (3.0, -0.03, "smooth")]),                          # ears pinned back, held
        antenna_r=keys([(0.0, 0.0), (0.3, -0.45, "smooth"), (2.4, -0.42, "hold"),
                        (3.0, -0.03, "smooth")]),
        # Sustained wide/fear hold through the cower; release with a relief burst
        # as it un-shrinks. A cancelled cower is caught by the eyes' safety
        # timeout, so it can never stay wide-eyed forever.
        events=(("eye", "fear", 0.2), ("eye", "release", 2.4)),
    ))

    # S3) Nervous look-around: tense scanning for a threat — quick darting checks
    # left/right that decay, over a slightly withdrawn carriage, ears held wary
    # half-back, eyes wide. The darts are quick but not oscillatory (each reaches
    # and briefly holds) so the soft head servo still tracks them.
    specs.append(ClipSpec(
        name="nervous_lookaround", authoring_path="blender", duration_s=3.2, loop_mode="once",
        requires_mode="stand", priority=23, blend_in_s=0.1, blend_out_s=0.35,
        doc="Nervous look-around: tense darting threat-checks over a withdrawn, wary carriage.",
        head_pitch=keys([(0.0, 0.0), (0.4, 0.05, "ease_out"), (2.6, 0.045, "hold"),
                         (3.2, 0.0, "smooth")]),                            # slightly withdrawn
        head_yaw=keys([(0.0, 0.0),
                       (0.4, 0.20, "ease_out"), (0.65, 0.19, "hold"),       # check right
                       (1.05, -0.22, "smooth"), (1.3, -0.21, "hold"),       # snap-check left
                       (1.7, 0.15, "smooth"), (1.95, 0.14, "hold"),         # check right, smaller
                       (2.4, -0.10, "smooth"), (2.65, -0.09, "hold"),
                       (3.2, 0.0, "smooth")]),
        head_roll=sine(0.8, 0.02, 0.0),                                     # a slight uneasy waver
        antenna_l=keys([(0.0, 0.0), (0.3, -0.25, "smooth"), (2.6, -0.22, "hold"),
                        (3.2, -0.02, "smooth")]),                           # wary half-back
        antenna_r=keys([(0.0, 0.0), (0.3, -0.25, "smooth"), (2.6, -0.22, "hold"),
                        (3.2, -0.02, "smooth")]),
        events=(("eye", "fear", 0.1), ("eye", "release", 2.8)),             # wide/held while scanning, relief at the end
    ))

    # S4) Calm down: the RECOVERY beat — exit fear back to neutral so the emotion
    # never looks stuck. Starts in a held-tense withdrawn pose, then releases: the
    # chin lifts (relief), the head returns to face forward (safe), the ears un-pin
    # and relax, and a burst of relieved blinks settles to calm. THIS is the clip
    # that explicitly releases a sustained wide/fear hold (e.g. from mood_scared),
    # whose release fires the relief burst. Use after any fear beat, or to
    # transition mood_scared -> a neutral/content mood.
    specs.append(ClipSpec(
        name="calm_down", authoring_path="blender", duration_s=3.4, loop_mode="once",
        requires_mode="stand", priority=19, blend_in_s=0.25, blend_out_s=0.5,
        doc="Calm down: release from fear to neutral — un-tuck, un-pin ears, a burst of relieved blinks.",
        head_pitch=keys([(0.0, 0.10), (0.5, 0.10, "hold"), (1.1, -0.05, "ease_out"),
                         (1.9, 0.03, "smooth"), (3.4, 0.0, "smooth")]),    # held tense -> release/lift -> soft settle
        neck_pitch=keys([(0.0, 0.03), (0.5, 0.03, "hold"), (1.2, -0.02, "smooth"),
                         (3.4, 0.0, "smooth")]),
        head_yaw=keys([(0.0, -0.10), (0.5, -0.10, "hold"), (1.3, 0.03, "smooth"),
                       (3.4, 0.0, "smooth")]),                             # turned away -> face forward (safe)
        head_roll=keys([(0.0, -0.06), (0.5, -0.06, "hold"), (1.3, 0.02, "smooth"),
                        (3.4, 0.0, "smooth")]),                            # untilt
        antenna_l=keys([(0.0, -0.40), (0.5, -0.40, "hold"), (1.3, 0.06, "ease_out"),
                        (2.3, 0.0, "smooth"), (3.4, 0.0, "hold")]),        # ears un-pin and relax
        antenna_r=keys([(0.0, -0.40), (0.5, -0.40, "hold"), (1.3, 0.06, "ease_out"),
                        (2.3, 0.0, "smooth"), (3.4, 0.0, "hold")]),
        # Release the sustained wide/fear hold as the chin lifts: this fires the
        # burst of relieved blinks. A single soft settling blink afterwards keeps
        # it alive when calm_down is played with nothing held (standalone).
        events=(("eye", "release", 1.15),),
        eyes=((0.0, 1), (2.3, 0), (2.4, 1)),
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
