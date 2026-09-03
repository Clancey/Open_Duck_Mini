#!/usr/bin/env python3
"""Generate a *standing* full-body wiggle reference motion for the episodic policy.

Disney's episodic policies (arXiv:2501.05204, Section V, Eq. 4) imitate a single
one-shot reference clip. Their "excited motion" is a full-body torso shake -- the
exact use case here. This script authors that reference in the 59-float-per-frame
format that ``EpisodicReferenceMotion`` consumes:

    frame = root_pos(3) + root_quat(4) + joints_pos(16) + left_toe(3) + right_toe(3)
          + world_lin_vel(3) + world_ang_vel(3) + joints_vel(16) + left_toe_vel(3)
          + right_toe_vel(3) + foot_contacts(2)                              = 59

Design (a preview-quality *standing* wiggle, not the docked one):
  * Joint trajectory is derived from the already-authored, hip-led happy wiggle
    ``dock_wiggle.duckanim`` (baseline == the standing "home" pose). Because the
    policy -- not a dock -- now carries the weight and balances, the reference can
    be more expressive: the hip/head oscillation is amplified while the knees and
    ankles stay tiny so both feet remain planted.
  * A torso shake is added to the *floating base* (root) orientation: roll + yaw
    sinusoids gated to a phase window [shake_start, shake_end] by a smooth
    envelope. This gives a clear one-shot "settle -> shake -> settle" profile and,
    crucially, a nonzero root angular velocity during the shake -- which is what the
    Eq. 13 phase-windowed angular-velocity-tracking boost rewards.
  * Foot contacts are asserted [1, 1] for every frame (both feet planted) -- checked
    deliberately, not inherited from a hardcoded exporter default (defect D3).

The joint order (16) matches the reference/duckanim convention:
  0-4  left leg (hip_yaw, hip_roll, hip_pitch, knee, ankle)
  5-8  head     (neck_pitch, head_pitch, head_yaw, head_roll)
  9-10 antennas (left, right)   <- removed later by reward_imitation
  11-15 right leg (hip_yaw, hip_roll, hip_pitch, knee, ankle)
"""

import argparse
import json
import os
import sys

import numpy as np

# Make the repo-root ``open_duck_anim`` package importable when this script is run
# directly from experiments/animation/.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# --- standing "home" pose (from scene_flat_terrain.xml keyframe "home") ----------
HOME_ROOT_Z = 0.15
# 16-joint home pose (antennas = 0). Legs match the keyframe; head/antennas zero.
HOME_JOINTS = np.array(
    [
        0.002, 0.053, -0.63, 1.368, -0.784,  # left leg
        0.0, 0.0, 0.0, 0.0,                  # head: neck_pitch, head_pitch, head_yaw, head_roll
        0.0, 0.0,                            # antennas
        -0.003, -0.065, 0.635, 1.379, -0.796,  # right leg
    ],
    dtype=np.float64,
)

FPS = 50
DEFAULT_DURATION_S = 3.0

# phase window (in [0,1]) over which the torso actively shakes. Must match the
# ang_vel_boost window in episodic.py default_config for the Eq. 13 boost to line up.
SHAKE_START = 0.25
SHAKE_END = 0.85

# joint-oscillation gains applied to the dock-wiggle oscillatory component.
#
# BALANCE-SAFE redesign: this robot has NO torso actuator, so any demanded base
# (trunk) angular velocity can only be produced by tipping the whole robot over.
# The original aggressive reference (hip-led rock + large trunk shake) rewarded
# exactly that -- the policy learned to FALL to match the demanded 1.75 rad/s of
# base rotation (episode length shrank 47->35 under Eq.13's x4 ang-vel boost).
# The visible "happy wiggle" is therefore driven by the HEAD (4 DOF, does not
# affect balance), with only a GENTLE hip sway and a near-zero base-rotation demand.
LEG_HIP_GAIN = 0.8   # gentle hip sway, does not fight balance
HEAD_GAIN = 3.5      # head leads the visible wiggle (yaw/roll ~0.32 rad = ~18 deg)
KNEE_ANKLE_GAIN = 0.5  # tiny so feet stay planted

# trunk (floating base) shake amplitudes, radians, at the shake frequency.
# Kept small: peak base ang-vel ~ 2*pi*f*amp ~= 0.38 rad/s, achievable by ankle/
# hip weight-shifting WITHOUT tipping (vs 1.75 rad/s before, which required a fall).
TRUNK_ROLL_AMP = 0.03
TRUNK_YAW_AMP = 0.02
SHAKE_FREQ_HZ = 2.0  # "happy wiggle" cadence

# joint soft limits (rad) from scene_flat_terrain.xml, order = 16-joint convention.
JOINT_LOWER = np.array([
    -0.52, -0.44, -1.22, -1.57, -1.57,   # left leg
    -0.35, -0.79, -2.79, -0.52,          # head
    -3.14, -3.14,                        # antennas (unused)
    -0.52, -0.44, -0.52, -1.57, -1.57,   # right leg
])
JOINT_UPPER = np.array([
    0.52, 0.44, 0.52, 1.57, 1.57,
    1.13, 0.79, 2.79, 0.52,
    3.14, 3.14,
    0.52, 0.44, 1.22, 1.57, 1.57,
])


def _smooth_window(phase, start, end, edge=0.08):
    """Smooth 0->1->0 envelope that is ~1 inside [start,end] with cosine edges."""
    w = np.zeros_like(phase)
    for i, p in enumerate(phase):
        if p < start - edge or p > end + edge:
            w[i] = 0.0
        elif p < start + edge:
            x = np.clip((p - (start - edge)) / (2 * edge), 0, 1)
            w[i] = 0.5 - 0.5 * np.cos(np.pi * x)
        elif p > end - edge:
            x = np.clip(((end + edge) - p) / (2 * edge), 0, 1)
            w[i] = 0.5 - 0.5 * np.cos(np.pi * x)
        else:
            w[i] = 1.0
    return w


def _quat_from_rpy(roll, pitch, yaw):
    """XYZW quaternion (scipy ``as_quat()`` order) from roll/pitch/yaw.

    The 59-float reference format stores ``root_quat`` in **scipy XYZW**
    (scalar-last) order — the same order the reference-motion generator emits via
    ``R.from_matrix(...).as_quat()``. A previous version of this helper returned
    ``[w, x, y, z]`` (scalar-first) and stored it unchanged, which transposed the
    implied angular-velocity axes relative to the stored ``world_ang_vel`` and
    silently broke episodic training. Emit XYZW here and let
    :func:`open_duck_anim.reference_validator.validate_reference` guard it.
    """
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([x, y, z, w])


def load_dock_wiggle_oscillation(path):
    """Return (n, 16) oscillatory component (deltas from per-joint mean) of the dock clip."""
    d = json.load(open(path))
    order = d["joints"]["order"]
    expected = [
        "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
        "neck_pitch", "head_pitch", "head_yaw", "head_roll",
        "left_antenna", "right_antenna",
        "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
    ]
    assert order == expected, f"unexpected joint order in {path}: {order}"
    frames = np.array(d["joints"]["frames"], dtype=np.float64)  # (n, 16)
    osc = frames - frames.mean(axis=0, keepdims=True)
    return osc, d["fps"]


def build_reference(dock_path, duration_s=DEFAULT_DURATION_S):
    n = int(round(duration_s * FPS))
    phase = np.linspace(0.0, 1.0, n)
    dt = 1.0 / FPS

    # --- joint trajectory: home pose + amplified dock-wiggle oscillation ----------
    osc, dock_fps = load_dock_wiggle_oscillation(dock_path)
    # resample dock oscillation to n frames
    src_phase = np.linspace(0.0, 1.0, osc.shape[0])
    osc_rs = np.stack(
        [np.interp(phase, src_phase, osc[:, j]) for j in range(osc.shape[1])], axis=1
    )
    gains = np.ones(16)
    gains[[0, 1, 11, 12]] = LEG_HIP_GAIN     # hip yaw/roll both legs
    gains[[2, 13]] = 1.2                      # hip pitch mild
    gains[[3, 4, 14, 15]] = KNEE_ANKLE_GAIN  # knees/ankles tiny
    gains[[5, 6, 7, 8]] = HEAD_GAIN          # head
    gains[[9, 10]] = 0.0                      # antennas not actuated in this XML

    # gate the joint wiggle by the same shake envelope (settle -> wiggle -> settle)
    env = _smooth_window(phase, SHAKE_START, SHAKE_END)
    joints = HOME_JOINTS[None, :] + osc_rs * gains[None, :] * env[:, None]
    # safety: keep joints inside their limits (with a small margin).
    joints = np.clip(joints, JOINT_LOWER[None, :] + 0.02, JOINT_UPPER[None, :] - 0.02)

    # --- trunk (floating base) shake: roll + yaw during the window ----------------
    shake = 2 * np.pi * SHAKE_FREQ_HZ * (phase * duration_s)
    roll = TRUNK_ROLL_AMP * np.sin(shake) * env
    yaw = TRUNK_YAW_AMP * np.sin(shake * 0.5 + 0.5) * env  # slower yaw sway
    pitch = np.zeros(n)
    root_quat = np.stack([_quat_from_rpy(roll[i], pitch[i], yaw[i]) for i in range(n)])

    root_pos = np.tile([0.0, 0.0, HOME_ROOT_Z], (n, 1))

    # --- velocities: match the canonical backward-difference / quaternion
    #     convention that reference_validator.validate_reference enforces, so the
    #     stored derived fields are self-consistent with the pose trajectory.
    from open_duck_anim.reference_validator import angular_velocity_from_quats

    def _backward_diff(x):
        d = np.zeros_like(x)
        d[1:] = (x[1:] - x[:-1]) / dt
        d[0] = d[1]
        return d

    joints_vel = _backward_diff(joints)
    world_lin_vel = _backward_diff(root_pos)  # ~0 (standing)
    # world angular velocity implied by the root_quat (XYZW) trajectory, using the
    # exact convention the validator recomputes: rotvec(q_i (x) conj(q_{i-1}))/dt.
    world_ang_vel = angular_velocity_from_quats(root_quat, dt)

    # --- toes (unused by the reward; fill with zeros) -----------------------------
    left_toe = np.zeros((n, 3))
    right_toe = np.zeros((n, 3))
    left_toe_vel = np.zeros((n, 3))
    right_toe_vel = np.zeros((n, 3))

    # --- foot contacts: BOTH feet planted throughout (asserted, not inherited) ----
    foot_contacts = np.ones((n, 2))

    frames = np.concatenate(
        [
            root_pos,          # 0:3
            root_quat,         # 3:7
            joints,            # 7:23
            left_toe,          # 23:26
            right_toe,         # 26:29
            world_lin_vel,     # 29:32
            world_ang_vel,     # 32:35
            joints_vel,        # 35:51
            left_toe_vel,      # 51:54
            right_toe_vel,     # 54:57
            foot_contacts,     # 57:59
        ],
        axis=1,
    )
    assert frames.shape == (n, 59), frames.shape

    # deliberate contact assertion (D3): a STANDING wiggle keeps both feet down.
    assert np.all(frames[:, -2:] == 1.0), "standing wiggle must have [1,1] contacts"

    out = {
        "LoopMode": "Wrap",
        "FPS": FPS,
        "EnableCycleOffsetPosition": True,
        "EnableCycleOffsetRotation": False,
        "Joints": [],
        "Vel_x": [],
        "Vel_y": [],
        "Yaw": [],
        "Placo": [],
        "Frame_offset": [],
        "Frame_size": [],
        "MotionWeight": 1,
        "Frames": frames.tolist(),
    }
    return out, frames


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument(
        "--dock",
        default=os.path.join(here, "clips", "dock_wiggle.duckanim"),
        help="source docked wiggle .duckanim",
    )
    ap.add_argument("--out", required=True, help="output reference JSON path")
    ap.add_argument("--duration", type=float, default=DEFAULT_DURATION_S)
    ap.add_argument(
        "--no-validate",
        action="store_true",
        help="skip the kinematic-consistency guard (NOT recommended)",
    )
    args = ap.parse_args()

    out, frames = build_reference(args.dock, args.duration)

    # --- authoring guard: a bad reference must never be written silently ----------
    # Recompute every derived field from the pose trajectory and refuse to emit a
    # reference with kinematic inconsistencies (transposed ang-vel axes, a zeroed
    # velocity channel that should be filled, wrong-dt finite differences, ...).
    # This is the exact class of defect that cost four episodic-policy runs.
    if not args.no_validate:
        from open_duck_anim.reference_validator import validate_reference

        result = validate_reference(frames, FPS, motion_type="standing_wiggle")
        if result.warnings:
            print("reference validator warnings:")
            for w in result.warnings:
                print("  " + str(w))
        if not result.ok:
            print("reference validator REJECTED the generated motion:", file=sys.stderr)
            print(result.summary(), file=sys.stderr)
            raise SystemExit(
                "refusing to write an inconsistent reference to %s "
                "(use --no-validate to override)" % args.out
            )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"))

    # --- report -------------------------------------------------------------------
    n = frames.shape[0]
    ang_vel = frames[:, 32:35]
    ang_speed = np.linalg.norm(ang_vel, axis=1)
    win_lo, win_hi = int(SHAKE_START * n), int(SHAKE_END * n)
    print(f"wrote {args.out}: {n} frames @ {FPS} FPS ({n/FPS:.2f} s)")
    print(f"  foot_contacts: all [1,1] = {bool(np.all(frames[:, -2:] == 1.0))}")
    print(f"  peak trunk ang-speed (shake window): "
          f"{ang_speed[win_lo:win_hi].max():.3f} rad/s")
    print(f"  ang-speed outside window (max): "
          f"{max(ang_speed[:win_lo].max(), ang_speed[win_hi:].max()):.3f} rad/s")
    j = frames[:, 7:23]
    print(f"  hip_roll(L) half-range: {(j[:,1].max()-j[:,1].min())/2:.3f} rad")
    print(f"  head_yaw   half-range: {(j[:,7].max()-j[:,7].min())/2:.3f} rad")
    print(f"  knee(L)    half-range: {(j[:,3].max()-j[:,3].min())/2:.3f} rad (should stay tiny)")


if __name__ == "__main__":
    main()
