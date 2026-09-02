#!/usr/bin/env python3
"""Dock / full-body validation harness (sibling to ``phase4_integrated_sim.py``).

The existing phase-4 harness assumes a *balancing* robot: it stands the duck on
a floor, runs the RL policy for the legs and layers a head-masked clip on top.
That model is exactly wrong for a dock-only full-body clip, where:

* there is no policy and no balance requirement (the dock/cradle carries the
  weight), and
* the legs are animated directly.

So this harness validates the things that actually matter for a docked
full-body clip:

1. **Joint-limit compliance** - every engine-emitted leg and head target stays
   inside the MJCF ``jnt_range`` (belt-and-braces on top of the engine clamp).
2. **Velocity-limit compliance** - successive bus targets never step faster than
   ``max_motor_velocity`` (5.24 rad/s) at the control rate.
3. **No self-collision / no clearance loss** - the point of a full-body clip is
   that the legs can now reach poses the head never could, so links that never
   used to move relative to each other now do. The shipped MJCF only marks the
   two foot pads collidable (for the floor) and its CAD meshes are built to
   *touch* without interpenetrating, so a ``data.ncon`` contact count is a
   vacuous self-collision test. Instead we measure the **signed distance**
   between every pair of geoms on non-adjacent bodies with ``mj_geomDistance``
   (which ignores contype/margin and resolves true geometry; negative == overlap).
   We record the clearance of every pair at the neutral dock-hold pose as a
   baseline (six pairs already overlap by design - nested head/trunk meshes),
   then assert the wiggle never (a) turns a clear pair into an overlapping one,
   (b) reduces any pair's clearance below a hard floor, or (c) makes an existing
   design-overlap measurably worse. A sensitivity self-test perturbs the hips
   well past the envelope to show the metric actually moves, proving it is live.

Run it with the phase-4 mujoco venv, e.g.::

    OPEN_DUCK_ANIM_HOME=<repo> MJCF=<xmls>/open_duck_mini_v2.xml \
        <phase4-venv>/bin/python experiments/animation/phase4_dock_fullbody_sim.py

Exit code is non-zero if any limit/velocity/clearance assertion fails.
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np

try:
    import mujoco
except Exception as exc:  # pragma: no cover - env dependent
    sys.stderr.write(
        "mujoco is required for this harness; run it with the phase4 venv "
        f"(import failed: {exc!r})\n"
    )
    raise SystemExit(2)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.environ.get("OPEN_DUCK_ANIM_HOME") or os.path.abspath(
    os.path.join(_HERE, "..", "..")
)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from open_duck_anim import clip as clipmod  # noqa: E402
from open_duck_anim.blend import Engine, Triggers, MODE_DOCK  # noqa: E402
from open_duck_anim.leg_envelope import DOCK_LEG_HOLD, LEG_NAMES  # noqa: E402
from open_duck_anim.limits import MAX_MOTOR_VELOCITY  # noqa: E402

CTRL_DT = 0.02  # 50 Hz control, matches the runtime + clip fps
HEAD_JOINTS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")

# Clearance policy (metres). Two independent guards:
#   * a pair that is comfortably clear at hold (>= PROXIMITY_FLOOR) must never be
#     driven closer than PROXIMITY_FLOOR by the wiggle (approach-to-contact guard);
#   * a pair that is already close or overlaps by design at hold (< PROXIMITY_FLOOR,
#     e.g. the six nested head/trunk meshes) must not get worse than WORSEN_TOL.
# Far-apart links whose separation merely changes as the head tilts are ignored -
# only genuine proximity matters for self-collision and cable strain.
PROXIMITY_FLOOR = 0.005      # 5 mm hard clearance floor
WORSEN_TOL = 0.003           # design-overlap / snug pairs may not deepen by >3 mm
PAIR_TRACK_CUTOFF = 0.20     # only track pairs closer than 20 cm at hold

_DEFAULT_MJCF = (
    "/Users/clancey/.copilot/session-state/"
    "9d7d4839-8a6c-44d0-8b98-328aebd93579/files/upstream/Open_Duck_Playground/"
    "playground/open_duck_mini_v2/xmls/open_duck_mini_v2.xml"
)


def _find_mjcf() -> str:
    cand = os.environ.get("MJCF") or _DEFAULT_MJCF
    if os.path.exists(cand):
        return cand
    raise SystemExit(f"MJCF not found: {cand} (set MJCF=<path to open_duck_mini_v2.xml>)")


def _find_clip() -> str:
    cand = os.environ.get("DOCK_CLIP") or os.path.join(_HERE, "clips", "dock_wiggle.duckanim")
    if os.path.exists(cand):
        return cand
    raise SystemExit(f"clip not found: {cand}")


def _joint_qposadr(model, name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise SystemExit(f"joint {name!r} not in MJCF")
    return int(model.jnt_qposadr[jid])


def _bname(model, bid):
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)


def _non_adjacent_geom_pairs(model):
    parent = {i: int(model.body_parentid[i]) for i in range(model.nbody)}

    def adjacent(b1, b2):
        return b1 == b2 or parent[b1] == b2 or parent[b2] == b1

    pairs = []
    for a, b in itertools.combinations(range(model.ngeom), 2):
        ba, bb = int(model.geom_bodyid[a]), int(model.geom_bodyid[b])
        if adjacent(ba, bb):
            continue
        pairs.append((a, b, ba, bb))
    return pairs


def _set_pose(model, data, qadr_leg, qadr_head, leg_vals, head_vals):
    for adr, v in zip(qadr_leg, leg_vals):
        data.qpos[adr] = float(v)
    for adr, v in zip(qadr_head, head_vals):
        data.qpos[adr] = float(v)
    mujoco.mj_forward(model, data)


def _pair_distances(model, data, pairs):
    return np.array(
        [mujoco.mj_geomDistance(model, data, a, b, 1.0, None) for (a, b, _, _) in pairs]
    )


def main() -> int:
    mjcf = _find_mjcf()
    clip_path = _find_clip()

    model = mujoco.MjModel.from_xml_path(mjcf)
    data = mujoco.MjData(model)

    qadr_leg = [_joint_qposadr(model, n) for n in LEG_NAMES]
    qadr_head = [_joint_qposadr(model, n) for n in HEAD_JOINTS]

    def _jrange(name):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lo, hi = model.jnt_range[jid]
        return float(lo), float(hi)

    leg_range = {n: _jrange(n) for n in LEG_NAMES}
    head_range = {n: _jrange(n) for n in HEAD_JOINTS}

    all_pairs = _non_adjacent_geom_pairs(model)

    # baseline clearances at the neutral dock-hold pose
    _set_pose(model, data, qadr_leg, qadr_head, DOCK_LEG_HOLD, np.zeros(4))
    baseline = _pair_distances(model, data, all_pairs)

    track_idx = [i for i, d0 in enumerate(baseline) if d0 < PAIR_TRACK_CUTOFF]
    tracked = [all_pairs[i] for i in track_idx]
    base_track = baseline[track_idx]

    # --- sensitivity self-test: prove the metric moves -----------------------
    st_leg = np.array(DOCK_LEG_HOLD, dtype=float)
    st_leg[LEG_NAMES.index("left_hip_yaw")] += 0.45
    st_leg[LEG_NAMES.index("right_hip_yaw")] -= 0.45
    st_leg[LEG_NAMES.index("left_hip_roll")] += 0.35
    st_leg[LEG_NAMES.index("right_hip_roll")] -= 0.35
    _set_pose(model, data, qadr_leg, qadr_head, st_leg, np.zeros(4))
    st_dist = _pair_distances(model, data, tracked)
    st_max_shift = float(np.max(np.abs(st_dist - base_track))) if len(tracked) else 0.0

    # --- run the engine over the clip in DOCK -------------------------------
    clip = clipmod.load_clip(clip_path)
    eng = Engine()
    dur = clip.n_frames / float(clip.fps)
    n_ticks = int(round((dur + 0.5) / CTRL_DT))  # +0.5 s tail to observe the settle

    prev_leg = None
    prev_head = None
    max_leg_vel = np.zeros(len(LEG_NAMES))
    max_head_vel = np.zeros(len(HEAD_JOINTS))
    leg_min = np.full(len(LEG_NAMES), np.inf)
    leg_max = np.full(len(LEG_NAMES), -np.inf)
    head_min = np.full(len(HEAD_JOINTS), np.inf)
    head_max = np.full(len(HEAD_JOINTS), -np.inf)
    limit_violations = []
    vel_violations = []
    min_track = base_track.copy()
    frames_with_legs = 0

    for k in range(n_ticks):
        t = k * CTRL_DT
        trig = Triggers(clips=[clip]) if k == 0 else None
        out = eng.evaluate(t, MODE_DOCK, trig)
        if out.leg_targets is None or out.head_targets is None:
            continue
        frames_with_legs += 1
        leg = np.asarray(out.leg_targets, dtype=float)
        head = np.asarray(out.head_targets, dtype=float)

        leg_min = np.minimum(leg_min, leg)
        leg_max = np.maximum(leg_max, leg)
        head_min = np.minimum(head_min, head)
        head_max = np.maximum(head_max, head)

        for i, n in enumerate(LEG_NAMES):
            lo, hi = leg_range[n]
            if leg[i] < lo - 1e-6 or leg[i] > hi + 1e-6:
                limit_violations.append((round(t, 3), n, float(leg[i]), lo, hi))
        for i, n in enumerate(HEAD_JOINTS):
            lo, hi = head_range[n]
            if head[i] < lo - 1e-6 or head[i] > hi + 1e-6:
                limit_violations.append((round(t, 3), n, float(head[i]), lo, hi))

        if prev_leg is not None:
            lv = np.abs(leg - prev_leg) / CTRL_DT
            max_leg_vel = np.maximum(max_leg_vel, lv)
            for i, n in enumerate(LEG_NAMES):
                if lv[i] > MAX_MOTOR_VELOCITY + 1e-6:
                    vel_violations.append((round(t, 3), n, float(lv[i])))
        if prev_head is not None:
            hv = np.abs(head - prev_head) / CTRL_DT
            max_head_vel = np.maximum(max_head_vel, hv)
            for i, n in enumerate(HEAD_JOINTS):
                if hv[i] > MAX_MOTOR_VELOCITY + 1e-6:
                    vel_violations.append((round(t, 3), n, float(hv[i])))
        prev_leg, prev_head = leg, head

        _set_pose(model, data, qadr_leg, qadr_head, leg, head)
        dist = _pair_distances(model, data, tracked)
        min_track = np.minimum(min_track, dist)

    clearance_violations = []
    worst = []
    for j, (a, b, ba, bb) in enumerate(tracked):
        d0 = float(base_track[j])
        dmin = float(min_track[j])
        loss = d0 - dmin
        worst.append((loss, d0, dmin, _bname(model, ba), _bname(model, bb)))
        if d0 >= PROXIMITY_FLOOR:
            # comfortably clear at hold: must not be driven to the contact floor
            if dmin < PROXIMITY_FLOOR:
                clearance_violations.append(
                    {"pair": [_bname(model, ba), _bname(model, bb)],
                     "reason": "approach_to_contact", "hold": d0, "min": dmin})
        else:
            # already close / design-overlap: must not deepen beyond tol
            if loss > WORSEN_TOL:
                clearance_violations.append(
                    {"pair": [_bname(model, ba), _bname(model, bb)],
                     "reason": "overlap_worsened", "hold": d0, "min": dmin, "loss": loss})
    worst.sort(reverse=True)

    ok = (
        not limit_violations
        and not vel_violations
        and not clearance_violations
        and frames_with_legs > 0
    )

    summary = {
        "mjcf": mjcf,
        "clip": os.path.basename(clip_path),
        "duration_s": round(dur, 3),
        "control_ticks": n_ticks,
        "frames_with_leg_targets": frames_with_legs,
        "non_adjacent_geom_pairs": len(all_pairs),
        "tracked_pairs_within_20cm": len(tracked),
        "baseline_design_overlaps": int(np.sum(baseline < 0)),
        "sensitivity_selftest_max_clearance_shift_m": round(st_max_shift, 4),
        "sensitivity_selftest_detects_motion": bool(st_max_shift > 0.002),
        "clip_min_clearance_m": round(float(np.min(min_track)), 4) if len(tracked) else None,
        "closest_approach_pairs": [
            {"pair": [_bname(model, tracked[j][2]), _bname(model, tracked[j][3])],
             "hold_m": round(float(base_track[j]), 4),
             "min_m": round(float(min_track[j]), 4)}
            for j in np.argsort(min_track)[:5]
        ] if len(tracked) else [],
        "worst_clearance_loss": [
            {"pair": [p[3], p[4]], "hold_m": round(p[1], 4),
             "min_m": round(p[2], 4), "loss_m": round(p[0], 4)}
            for p in worst[:5]
        ],
        "clearance_violations": clearance_violations,
        "leg_deflection_from_hold": {
            n: [round(float(leg_min[i] - DOCK_LEG_HOLD[i]), 4),
                round(float(leg_max[i] - DOCK_LEG_HOLD[i]), 4)]
            for i, n in enumerate(LEG_NAMES)
        },
        "leg_range_used_vs_mjcf": {
            n: {"used": [round(float(leg_min[i]), 4), round(float(leg_max[i]), 4)],
                "mjcf": [round(leg_range[n][0], 4), round(leg_range[n][1], 4)]}
            for i, n in enumerate(LEG_NAMES)
        },
        "head_range_used_vs_mjcf": {
            n: {"used": [round(float(head_min[i]), 4), round(float(head_max[i]), 4)],
                "mjcf": [round(head_range[n][0], 4), round(head_range[n][1], 4)]}
            for i, n in enumerate(HEAD_JOINTS)
        },
        "max_leg_velocity": {n: round(float(max_leg_vel[i]), 4) for i, n in enumerate(LEG_NAMES)},
        "max_head_velocity": {n: round(float(max_head_vel[i]), 4) for i, n in enumerate(HEAD_JOINTS)},
        "max_motor_velocity": MAX_MOTOR_VELOCITY,
        "limit_violations": limit_violations,
        "velocity_violations": vel_violations,
        "PASS": ok,
    }
    print(json.dumps(summary, indent=2))

    if not summary["sensitivity_selftest_detects_motion"]:
        sys.stderr.write(
            "WARNING: sensitivity self-test saw no clearance change; the "
            "distance metric may not be resolving geometry.\n"
        )
    if not ok:
        sys.stderr.write("FAIL: dock full-body validation failed (see summary)\n")
        return 1
    print("\nPASS: dock_wiggle stays within joint/velocity limits and never "
          "reduces any non-adjacent link clearance beyond tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
