#!/usr/bin/env python3
"""Author the Open Duck Mini v2 **dock-only full-body** clip(s).

This is a *separate* authoring entry point from ``author_clips.py`` (which owns
the head-masked library). It exists because full-body clips are a different,
tightly-restricted category:

* They animate the **legs**, so they are legal **only** with
  ``requires_mode="dock"`` (plan §6.2). On the dock the legs are not
  load-bearing and no policy runs, so leg motion is safe; standing/walking is
  not, and the compiler + engine both reject a full-body clip in any other mode.
* They must never enter the idle service's candidate lists — a full-body clip
  firing unattended on a robot that might not be docked is exactly the failure
  the architecture avoids. These clips are **deliberately-triggered only**.

Both the head channels AND the leg channels are safety-checked before writing:

* Head: the same ×0.5 hardware-derated head envelope as the head library
  (per-axis deflection box + combined L2 budget + slew). Built from head roll /
  pitch / yaw + timing, NOT neck (neck_pitch is the binding axis at ~0.12 rad
  derated).
* Legs: the conservative dock leg envelope (:mod:`open_duck_anim.leg_envelope`)
  — a small deflection box around the dock hold, intersected with the MJCF
  ``jnt_range``, ×0.5 derated for first hardware use. Authored *inside* the
  derated box so the runtime clamp is a no-op, and additionally checked against
  the 5.24 rad/s per-joint rate limit at 50 Hz.

The compiler is shared (no logic duplicated): we build the 16 joint angles per
frame directly (legs = dock hold + authored deflection; head = authored absolute
angle; antennas per calibration) and feed them through
``open_duck_anim.compiler`` via ``export_and_compile``.

Usage::

    python experiments/animation/author_dock_clips.py
    python experiments/animation/author_dock_clips.py --only dock_wiggle
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_MAIN_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_MAIN_REPO, os.path.join(_MAIN_REPO, "blender")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from open_duck_anim import compiler  # noqa: E402
from open_duck_anim.envelope import DEFAULT_ENVELOPE, HARDWARE_DERATING  # noqa: E402
from open_duck_anim.joint_order import JOINT_ORDER_16  # noqa: E402
from open_duck_anim.leg_envelope import (  # noqa: E402
    DOCK_LEG_HOLD,
    DERATED_LEG_ENVELOPE,
    LEG_NAMES,
)
from open_duck_anim.limits import MAX_MOTOR_VELOCITY  # noqa: E402
from open_duck_anim_blender import export as export_mod  # noqa: E402

# Reuse the exact curve primitives the head library is authored with, so a dock
# clip reads the same way and shares the craft vocabulary (keys/pulse/sine/...).
from author_clips import (  # noqa: E402
    Track, TrackLike, as_track, const, ZERO, keys, sine, pulse, smootherstep,
    FPS, HEAD_CHANNELS, ANTENNA_CAL, SOURCE_JSON_DIR, SOURCE_BLEND, REPO_CLIPS_DIR,
)

# Index of every joint by name in the 16-DOF reference order (compiler input).
_J16 = {n: i for i, n in enumerate(JOINT_ORDER_16)}
# Dock hold indexed by name (LEG_NAMES aligns with the leg block order).
_HOLD_BY_NAME = {n: float(DOCK_LEG_HOLD[i]) for i, n in enumerate(LEG_NAMES)}


@dataclass
class DockClipSpec:
    """A dock-only full-body clip. ``layer_mask`` and ``requires_mode`` are FIXED.

    Leg tracks are authored as a **deflection from the dock hold** (rad); the
    build adds the hold so the shipped joint angles are absolute. Head tracks are
    absolute head angles (rad), antennas normalised [-1,1].
    """

    name: str
    duration_s: float
    priority: int
    doc: str = ""
    loop_mode: str = "once"           # dock beats are one-shots, never loops
    blend_in_s: float = 0.25
    blend_out_s: float = 0.4
    show_blend_in_s: float = 0.1
    show_blend_out_s: float = 0.1
    # Head channels (absolute, rad). Build from roll/pitch/yaw + timing, not neck.
    neck_pitch: TrackLike = 0.0
    head_pitch: TrackLike = 0.0
    head_yaw: TrackLike = 0.0
    head_roll: TrackLike = 0.0
    # Leg channels — deflection from the dock hold (rad), one per LEG_NAMES entry.
    left_hip_yaw: TrackLike = 0.0
    left_hip_roll: TrackLike = 0.0
    left_hip_pitch: TrackLike = 0.0
    left_knee: TrackLike = 0.0
    left_ankle: TrackLike = 0.0
    right_hip_yaw: TrackLike = 0.0
    right_hip_roll: TrackLike = 0.0
    right_hip_pitch: TrackLike = 0.0
    right_knee: TrackLike = 0.0
    right_ankle: TrackLike = 0.0
    # Antennas (normalised) + discrete events + eye track.
    antenna_l: TrackLike = 0.0
    antenna_r: TrackLike = 0.0
    events: Sequence[Tuple[str, str, float]] = field(default_factory=tuple)
    eyes: Sequence[Tuple[float, int]] = field(default_factory=tuple)

    layer_mask: str = "full_body"      # FIXED
    requires_mode: str = "dock"        # FIXED

    @property
    def frame_count(self) -> int:
        return int(round(self.duration_s * FPS))

    def leg_tracks(self) -> Dict[str, Track]:
        return {n: as_track(getattr(self, n)) for n in LEG_NAMES}

    def head_tracks(self) -> Dict[str, Track]:
        return {c: as_track(getattr(self, c)) for c in HEAD_CHANNELS}


# =============================================================================
# THE DOCK LIBRARY.  Currently one deliverable: the happy full-body wiggle.
# =============================================================================
#
# Leg deflection sign conventions (deflection from the dock hold, rad):
#   hip_yaw   twists the body about vertical (the "wag" — the lead axis)
#   hip_roll  rocks the body side to side
#   hip_pitch / knee / ankle  a small vertical bounce (kept tiny: self-collision)
# Both legs are driven the SAME sign on yaw/roll so the whole lower body swings
# together (a body twist + rock), not a splay.
#
# Amplitudes are kept INSIDE the ×0.5 derated caps so nothing is clamped at
# runtime. Derated deflection caps (rad): hip_yaw 0.10, hip_roll 0.06,
# hip_pitch 0.05, knee 0.04, ankle 0.04. Head derated ceilings: neck ±0.12,
# head_pitch ±0.27, head_yaw ±0.52, head_roll ±0.17 (shared L2 budget 0.7).


def _decaying_wag(peaks: Sequence[Tuple[float, float]], settle_t: float,
                  duration: float) -> Track:
    """A hip-led wag: alternating decaying peaks that settle to 0 and hold.

    ``peaks`` = ``[(t_sec, value), ...]``; the first segment eases out (snap into
    the wind-up), the rest ease smoothly (decaying oscillation), then a smooth
    settle to 0 at ``settle_t`` held to ``duration``. This is the anticipation →
    overshoot → decaying repetitions → settle shape in one call.
    """
    pts: List[Tuple] = [(0.0, 0.0)]
    for i, (t, v) in enumerate(peaks):
        pts.append((t, v, "ease_out" if i == 0 else "smooth"))
    pts.append((settle_t, 0.0, "smooth"))
    pts.append((duration, 0.0, "hold"))
    return keys(pts)


def build_dock_specs() -> List[DockClipSpec]:
    specs: List[DockClipSpec] = []

    D = 3.0  # the wiggle: ~3 s, one-shot, deliberately triggered.

    # Hip YAW — the LEAD. Energy starts here: a tiny anticipation wind (opposite),
    # then an overshooting first wag and ~5 decaying repetitions at ~2.2 Hz.
    yaw_wag = _decaying_wag(
        [(0.12, -0.025),   # anticipation (wind the other way first)
         (0.30, 0.090),    # overshoot: the biggest wag
         (0.52, -0.075),
         (0.74, 0.060),
         (0.96, -0.045),
         (1.18, 0.030),
         (1.40, -0.020),
         (1.62, 0.012)],
        settle_t=2.05, duration=D)

    # Hip ROLL — the side rock, coupled to the yaw with a slight lag (a figure-8
    # feel), smaller amplitude and one fewer beat.
    roll_wag = _decaying_wag(
        [(0.20, 0.052),
         (0.42, -0.045),
         (0.64, 0.034),
         (0.86, -0.026),
         (1.08, 0.018),
         (1.30, -0.011)],
        settle_t=1.9, duration=D)

    # A small vertical bounce shared by hip_pitch/knee/ankle (tiny — knee/ankle
    # excursions are the self-collision risk, so they barely move). Crouch-and-
    # rise synced loosely to the wag energy, decaying.
    bounce = keys([(0.0, 0.0), (0.30, 1.0, "ease_out"), (0.60, 0.0, "smooth"),
                   (0.92, 0.6, "smooth"), (1.25, 0.0, "smooth"),
                   (1.65, 0.3, "smooth"), (2.05, 0.0, "smooth"),
                   (D, 0.0, "hold")])

    # Head joins in, TRAILING the hips (heavier, lags): a happy tilt-and-turn that
    # rocks with the body, plus a slight bright chin-up. Built from roll/yaw/pitch
    # (+ a whisper of neck), never leaning on neck_pitch.
    head_roll_wag = _decaying_wag(
        [(0.40, 0.110),
         (0.66, -0.088),
         (0.94, 0.058),
         (1.28, -0.034),
         (1.66, 0.016)],
        settle_t=2.2, duration=D)
    head_yaw_wag = _decaying_wag(
        [(0.36, 0.095),
         (0.62, -0.075),
         (0.90, 0.050),
         (1.24, -0.028)],
        settle_t=2.1, duration=D)

    specs.append(DockClipSpec(
        name="dock_wiggle",
        duration_s=D,
        priority=25,
        doc="Happy full-body wiggle for the DOCK ONLY: hips lead a decaying "
            "side-to-side wag, body rocks, head and antennas join, eyes bright.",
        blend_in_s=0.2, blend_out_s=0.4,
        # --- legs (deflection from dock hold) ---
        left_hip_yaw=yaw_wag, right_hip_yaw=yaw_wag,      # twist together
        left_hip_roll=roll_wag, right_hip_roll=roll_wag,  # rock together
        left_hip_pitch=(bounce * 0.020), right_hip_pitch=(bounce * 0.020),
        left_knee=(bounce * 0.015), right_knee=(bounce * 0.015),
        left_ankle=(bounce * -0.015), right_ankle=(bounce * -0.015),
        # --- head (absolute, trailing) ---
        head_roll=head_roll_wag,
        head_yaw=head_yaw_wag,
        head_pitch=keys([(0.0, 0.0), (0.45, -0.055, "ease_out"),
                         (1.6, -0.035, "hold"), (2.4, 0.0, "smooth"),
                         (D, 0.0, "hold")]),              # bright chin-up
        neck_pitch=pulse(0.5, 0.5, -0.02),               # tiny lift, well under 0.12
        # --- antennas: bright perks bouncing on the first wags, decaying ---
        antenna_l=(pulse(0.30, 0.18, 0.26) + pulse(0.74, 0.18, 0.18)
                   + pulse(1.18, 0.18, 0.10)),
        antenna_r=(pulse(0.30, 0.18, 0.26) + pulse(0.74, 0.18, 0.18)
                   + pulse(1.18, 0.18, 0.10)),
        events=(("eye", "happy", 0.30),),                # bright happy blink
    ))

    return specs


# =============================================================================
# Frame build + safety self-checks
# =============================================================================


def _eval(spec: DockClipSpec):
    n = spec.frame_count
    t = np.arange(n, dtype=np.float64) / FPS
    head4 = np.stack([spec.head_tracks()[c](t) for c in HEAD_CHANNELS], axis=1)
    legs = {name: tr(t) for name, tr in spec.leg_tracks().items()}  # deflections
    a_l = as_track(spec.antenna_l)(t)
    a_r = as_track(spec.antenna_r)(t)
    return t, head4, legs, np.stack([a_l, a_r], axis=1)


def build_episode(spec: DockClipSpec) -> Dict:
    """Build the 59-float episode with legs = hold + deflection, head absolute."""
    _, head4, legs, antennas = _eval(spec)
    n = spec.frame_count
    episode = export_mod.new_episode(fps=FPS, contacts_valid=False)
    zeros3, zeros16, zeros2 = [0.0] * 3, [0.0] * 16, [0.0, 0.0]
    ident_quat = [0.0, 0.0, 0.0, 1.0]
    for i in range(n):
        joints16 = [0.0] * 16
        # legs: absolute = hold + authored deflection
        for name in LEG_NAMES:
            joints16[_J16[name]] = _HOLD_BY_NAME[name] + float(legs[name][i])
        # head: absolute authored angle
        joints16[_J16["neck_pitch"]] = float(head4[i, 0])
        joints16[_J16["head_pitch"]] = float(head4[i, 1])
        joints16[_J16["head_yaw"]] = float(head4[i, 2])
        joints16[_J16["head_roll"]] = float(head4[i, 3])
        # antennas: left stores value, right stores -value (calibration rad=-1..1,
        # right sign -1 → runtime output == authored right value).
        joints16[_J16["left_antenna"]] = float(antennas[i, 0])
        joints16[_J16["right_antenna"]] = float(-antennas[i, 1])
        frame = export_mod.assemble_frame(
            root_position=zeros3, root_quaternion=ident_quat,
            joint_positions=joints16, left_toe_pos=zeros3, right_toe_pos=zeros3,
            world_linear_vel=zeros3, world_angular_vel=zeros3,
            joint_velocities=zeros16, left_toe_vel=zeros3, right_toe_vel=zeros3,
            foot_contacts=zeros2,
        )
        episode["Frames"].append(frame)
    return episode


def spec_to_meta(spec: DockClipSpec) -> Dict:
    n = spec.frame_count
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
    events = [{"frame": max(0, min(n - 1, int(round(t * FPS)))),
               "type": str(typ), "value": str(val)}
              for (typ, val, t) in spec.events]
    return {
        "name": spec.name,
        "loop_mode": spec.loop_mode,
        "requires_mode": spec.requires_mode,   # "dock"
        "priority": int(spec.priority),
        "layer_mask": spec.layer_mask,         # "full_body"
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


def head_envelope_report(spec: DockClipSpec, derating: float = HARDWARE_DERATING) -> Dict:
    """Head safety: run the authored head offsets through the derated envelope."""
    _, head4, _, _ = _eval(spec)
    env = DEFAULT_ENVELOPE.derated(derating)
    L = env._L
    prev = None
    max_clamp_delta = 0.0
    max_l2 = 0.0
    max_defl = np.zeros(4)
    for row in head4:
        clamped = env.clamp(row, prev_command_head=prev, dt=1.0 / FPS)
        max_clamp_delta = max(max_clamp_delta, float(np.max(np.abs(clamped - row))))
        max_defl = np.maximum(max_defl, np.abs(row))
        max_l2 = max(max_l2, float(np.sqrt(np.sum((row / L) ** 2))))
        prev = clamped
    return {"max_clamp_delta_rad": max_clamp_delta, "max_l2_norm": max_l2,
            "l2_budget": env.l2_budget,
            "max_abs_deflection": {c: float(max_defl[i]) for i, c in enumerate(HEAD_CHANNELS)}}


def leg_safety_report(spec: DockClipSpec) -> Dict:
    """Leg safety: derated-envelope clamp delta + per-frame rate check at 50 Hz."""
    _, _, legs, _ = _eval(spec)
    n = spec.frame_count
    env = DERATED_LEG_ENVELOPE
    # absolute leg targets per frame, LEG_NAMES order
    abs_legs = np.stack(
        [np.array([_HOLD_BY_NAME[name] + float(legs[name][i]) for name in LEG_NAMES])
         for i in range(n)], axis=0)
    max_clamp_delta = 0.0
    for row in abs_legs:
        clamped = env.clamp(row)
        max_clamp_delta = max(max_clamp_delta, float(np.max(np.abs(clamped - row))))
    # rate check: per-frame step vs max_motor_velocity * dt
    dt = 1.0 / FPS
    max_step = float(np.max(np.abs(np.diff(abs_legs, axis=0)))) if n > 1 else 0.0
    rate_budget = MAX_MOTOR_VELOCITY * dt
    max_defl = np.max(np.abs(abs_legs - DOCK_LEG_HOLD), axis=0)
    return {"max_clamp_delta_rad": max_clamp_delta, "max_step_rad": max_step,
            "rate_budget_rad_per_frame": rate_budget,
            "max_abs_deflection": {LEG_NAMES[i]: float(max_defl[i]) for i in range(len(LEG_NAMES))}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=None, help="subset of clip names")
    ap.add_argument("--out-dir", default=REPO_CLIPS_DIR)
    ap.add_argument("--allow-clamp", action="store_true",
                    help="inspect only: write even if a channel would be clamped")
    args = ap.parse_args()

    specs = build_dock_specs()
    if args.only:
        specs = [s for s in specs if s.name in set(args.only)]
        if not specs:
            raise SystemExit("no dock specs match --only %r" % (args.only,))

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(SOURCE_JSON_DIR, exist_ok=True)
    for spec in specs:
        hrep = head_envelope_report(spec)
        lrep = leg_safety_report(spec)
        problems = []
        if hrep["max_clamp_delta_rad"] > 1e-4:
            problems.append("head clamped by derated envelope (Δ=%.4g rad, ||c/L||=%.3f > %.2f)"
                            % (hrep["max_clamp_delta_rad"], hrep["max_l2_norm"], hrep["l2_budget"]))
        if lrep["max_clamp_delta_rad"] > 1e-4:
            problems.append("legs clamped by derated leg envelope (Δ=%.4g rad)"
                            % lrep["max_clamp_delta_rad"])
        if lrep["max_step_rad"] > lrep["rate_budget_rad_per_frame"] - 1e-9:
            problems.append("leg step %.4g rad/frame exceeds rate budget %.4g"
                            % (lrep["max_step_rad"], lrep["rate_budget_rad_per_frame"]))
        if problems and not args.allow_clamp:
            raise SystemExit("clip %r unsafe to ship: %s" % (spec.name, "; ".join(problems)))

        episode = build_episode(spec)
        out_path = os.path.join(args.out_dir, spec.name + ".duckanim")
        source_path = os.path.join(SOURCE_JSON_DIR, spec.name + ".source.json")
        meta = spec_to_meta(spec)
        res = export_mod.export_and_compile(episode, meta, source_path, out_path)
        print("[dock] %-14s frames=%-4d head||c/L||=%.3f headΔ=%.4g legΔ=%.4g "
              "legstep=%.4g/%.4g  -> %s"
              % (spec.name, spec.frame_count, hrep["max_l2_norm"],
                 hrep["max_clamp_delta_rad"], lrep["max_clamp_delta_rad"],
                 lrep["max_step_rad"], lrep["rate_budget_rad_per_frame"],
                 os.path.basename(res["duckanim_path"])))


if __name__ == "__main__":
    main()
