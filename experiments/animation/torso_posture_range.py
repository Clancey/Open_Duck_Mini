#!/usr/bin/env python3
"""Torso-posture range: MJCF facts now, trained-policy sweep for Phase 3.

Full-body *sustained* emotion (sad sag, proud puff, alert tall) is driven by the
torso **height** and **orientation** COMMANDS added to the standing policy
(``Open_Duck_Playground .../standing.py``; Disney BD-X arXiv:2501.05204 Eq.5).
Because torso posture directly affects balance, the safe command range MUST be a
*measured* envelope against the trained policy — exactly as the head envelope was
swept (``experiments/animation/envelope_sweep.py``; plan §6.5): a genuinely-fine
first-onset outward sweep with >=5 s holds (head instability proved non-monotonic
and time-dependent). No torso policy has been trained yet, so this script:

1. ``report_mjcf_facts()`` — reads the MJCF and prints the robust, auditable
   facts that bound the range: the nominal foot-flat standing height and the
   per-leg-joint squat/straighten headroom. These are KINEMATIC bounds (what the
   duck *could* reach), NOT the balance-holdable range (a subset, unknown until
   swept). Runs today with only mujoco + numpy.

2. ``validate_policy_tracking(onnx)`` — the Phase-3 sweep to run ONCE a checkpoint
   exists: command a grid of torso heights/orientations, hold each >=5 s in
   MuJoCo, and report the held error, the first-onset of instability, and any
   falls. This is what turns the provisional ``open_duck_anim.torso_envelope``
   placeholders into a measured envelope. Skips (prints the plan) without an onnx.

Usage:
    python torso_posture_range.py            # MJCF facts only
    python torso_posture_range.py --onnx PATH # + policy tracking sweep (Phase 3)
"""

import argparse
import os
import sys

import numpy as np

# Provisional command envelope shipped in open_duck_anim.torso_envelope. Mirrored
# here (not imported) so the harness stands alone next to the Playground clone.
PROVISIONAL_ENVELOPE = {
    "torso_height_delta_m": (-0.020, 0.020),  # about a ~0.16 m nominal
    "torso_grav_x": (-0.12, 0.12),            # ~ sin(pitch); +/-0.12 ~ 6.9 deg
    "torso_grav_y": (-0.06, 0.06),            # ~ -sin(roll);  +/-0.06 ~ 3.4 deg
}
# Kinematic reach established in Phase 1 (foot-flat, CoM-over-support search).
KINEMATIC_HEIGHT_BAND_M = (0.127, 0.195)


def _find_mjcf():
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.environ.get("DUCK_SCENE_XML", ""),
        os.path.expanduser(
            "~/.copilot/session-state/9d7d4839-8a6c-44d0-8b98-328aebd93579/files/"
            "upstream/Open_Duck_Playground/playground/open_duck_mini_v2/xmls/"
            "scene_flat_terrain.xml"
        ),
        os.path.join(here, "scene_flat_terrain.xml"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def report_mjcf_facts(scene_xml):
    import mujoco

    m = mujoco.MjModel.from_xml_path(scene_xml)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

    base_z = float(d.qpos[2])
    # Lowest foot-bottom geom z (negative = slight ground penetration in keyframe).
    foot_z = []
    for g in range(m.ngeom):
        gn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if "foot_bottom" in gn.lower():
            foot_z.append(float(d.geom_xpos[g][2]))
    lowest_foot = min(foot_z) if foot_z else 0.0
    nominal_flat_height = base_z - lowest_foot  # base height with feet exactly on ground

    print("=" * 74)
    print("MJCF FACTS (kinematic bounds — NOT the balance-holdable range)")
    print("=" * 74)
    print("scene: %s" % scene_xml)
    print("home keyframe base z ...... %.4f m" % base_z)
    print("lowest foot-bottom geom z . %+.4f m (keyframe penetration)" % lowest_foot)
    print("nominal foot-flat height .. %.4f m  (standing.py torso_height_nominal=0.16)"
          % nominal_flat_height)
    print()
    print("Leg-joint squat/straighten headroom (home -> jnt_range edge):")
    for n in ("left_knee", "right_knee", "left_hip_pitch", "right_hip_pitch",
              "left_ankle", "right_ankle"):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        adr = m.jnt_qposadr[j]
        home = float(d.qpos[adr])
        lo, hi = float(m.jnt_range[j][0]), float(m.jnt_range[j][1])
        print("  %-16s home %+.3f  range [%+.3f, %+.3f]  (down %+.3f / up %+.3f)"
              % (n, home, lo, hi, hi - home, lo - home))
    print()
    print("Knees sit near +1.37 rad (max +1.571) => only ~0.2 rad DEEPER squat is")
    print("kinematically available; lots of room to STRAIGHTEN (stand taller).")
    print("Phase-1 foot-flat + CoM-over-support search => reachable base height")
    print("  ~%.3f m (deep squat) .. %.3f m (near-straight), about a %.2f m nominal."
          % (KINEMATIC_HEIGHT_BAND_M[0], KINEMATIC_HEIGHT_BAND_M[1], nominal_flat_height))
    print()
    print("SHIPPED PROVISIONAL COMMAND ENVELOPE (open_duck_anim.torso_envelope) —")
    print("tighter than the reach because the HOLDABLE range is unknown until swept:")
    for ch, (lo, hi) in PROVISIONAL_ENVELOPE.items():
        print("  %-20s [%+.3f, %+.3f]" % (ch, lo, hi))
    print()
    print(">>> These are UNSWEPT. Run validate_policy_tracking() once a standing")
    print(">>> checkpoint exists, BEFORE any hardware use (plan §6.5 methodology).")
    return nominal_flat_height


def validate_policy_tracking(onnx_path, scene_xml, hold_s=5.0):
    """Phase-3 sweep: command torso posture, hold, measure tracking + falls.

    This is the scaffold to complete once ``standing.py`` has been trained. It
    mirrors the head ``envelope_sweep.py`` protocol for the torso command:

      * for each commanded torso height in a fine grid across the training range,
        and each commanded (grav_x, grav_y), ramp the command in over ~0.5 s and
        HOLD for ``hold_s`` (>=5 s: the head's topples took up to ~3.4 s);
      * log held base-height error, held projected-gravity error, peak base tilt,
        and whether the robot fell (get_gravity(data)[2] < 0 or min height);
      * the SAFE limit per channel is SAFETY_FRACTION (0.5) * the first-onset
        deflection at which a >=5 s hold violates the stability criterion, taken
        as the most conservative across the grid — identical rule to the head.

    The observation/command wiring must match ``standing.py`` (obs 68-wide,
    command length 10: [vx,vy,wz, neck,hp,hy,hr, h_torso, grav_x, grav_y], with
    locomotion forced to zero). Left as an explicit TODO because it requires the
    trained checkpoint and the mujoco_playground env, which are not available in
    this sim-only environment.
    """
    if not onnx_path or not os.path.exists(onnx_path):
        print("\n" + "=" * 74)
        print("PHASE-3 POLICY TRACKING SWEEP — NOT RUN (no checkpoint)")
        print("=" * 74)
        print("No trained standing checkpoint supplied (--onnx). Once standing.py")
        print("is trained (~300M steps), run this sweep to measure the holdable")
        print("torso range and replace the provisional envelope. Protocol:")
        print("  height grid : linspace(0.13, 0.19, 13)  (2 x training range res)")
        print("  orient grid : grav_x,grav_y in {-0.2..0.2} x {-0.1..0.1}, fine")
        print("  per point   : 0.5 s ramp + %.1f s hold; log height/orient error," % hold_s)
        print("                peak base tilt, fell?; first-onset rule; x0.5 margin.")
        print("  ALSO verify head-command tracking still holds WITH a torso command")
        print("  applied (compose cmd[3:7] head sweep on top of a non-neutral torso).")
        return None

    # --- With a checkpoint, the sweep would run here. ---
    raise NotImplementedError(
        "Torso tracking sweep requires the mujoco_playground standing env + the "
        "trained checkpoint; wire obs(68)/command(10) exactly as standing.py and "
        "follow the protocol in this function's docstring."
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", default=None, help="trained standing checkpoint (Phase 3)")
    ap.add_argument("--scene", default=None, help="path to scene_flat_terrain.xml")
    ap.add_argument("--hold-s", type=float, default=5.0)
    args = ap.parse_args(argv)

    scene = args.scene or _find_mjcf()
    if not scene:
        print("ERROR: could not locate scene_flat_terrain.xml; pass --scene PATH "
              "or set DUCK_SCENE_XML.", file=sys.stderr)
        return 2

    report_mjcf_facts(scene)
    validate_policy_tracking(args.onnx, scene, hold_s=args.hold_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
