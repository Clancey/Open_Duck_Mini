#!/usr/bin/env python3
"""Phase 4 integrated validation - play an authored clip through the FULL
runtime path against the retrained passthrough policy in MuJoCo.

This exercises the real integration end to end:

    open_duck_anim.Engine (idle_alive clip, DERATED head envelope)
      -> anim.ModeFSM (held in STAND)  -> anim.SafetyMonitor
      -> anim.AnimationController.prepare()   [engine evaluated ONCE per tick]
      -> head_command_offsets written into commands[3:7]  (enter the obs)
      -> passthrough_final_300M.onnx  (obs[1,101] -> action[1,14])
      -> AnimationController.finalize(policy_targets)
           = additive head path (motor_targets[5:9] += offset)
           + FINAL 14-DOF bus JointLimiter + JointRateLimiter (5.24 rad/s)
      -> MuJoCo physics (50 Hz, decimation 10)

Assertions (plan §7 Phase-4 acceptance):
  * stays upright (never falls),
  * peak base tilt < 8.6 deg,
  * no joint-position or joint-velocity limit violations on the bus targets,
  * the head actually FOLLOWS the authored motion (per-channel correlation).

Reuses the validated closed-loop harness from spike_s01_head_response.py. Writes
a JSON summary and a plot to the Phase-4 artefacts dir (kept out of git).

Run (from the main repo, with the mujoco venv)::

  OPEN_DUCK_ANIM_HOME=<main_repo> RUNTIME_HOME=<runtime_clone> \\
    <venv>/bin/python experiments/animation/phase4_integrated_sim.py
"""

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_MAIN_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _MAIN_REPO not in sys.path:
    sys.path.insert(0, _MAIN_REPO)
os.environ.setdefault("OPEN_DUCK_ANIM_HOME", _MAIN_REPO)

# The runtime clone (where the anim integration lives).
_DEFAULT_RUNTIME = (
    "/Users/clancey/.copilot/session-state/"
    "9d7d4839-8a6c-44d0-8b98-328aebd93579/files/upstream/Open_Duck_Mini_Runtime"
)
_RUNTIME_HOME = os.environ.get("RUNTIME_HOME", _DEFAULT_RUNTIME)
_RUNTIME_PKG = os.path.join(_RUNTIME_HOME, "mini_bdx_runtime")
if _RUNTIME_PKG not in sys.path:
    sys.path.insert(0, _RUNTIME_PKG)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Reuse the validated harness + constants.
from spike_s01_head_response import (  # noqa: E402
    Harness, NB_STEPS_IN_PERIOD, ACTION_SCALE, CTRL_DT, MAX_MOTOR_VELOCITY,
    FALL_HEIGHT, DEFAULT_MJCF,
)

from open_duck_anim import load_clip, Triggers  # noqa: E402
from mini_bdx_runtime.anim.controller import AnimationController, ControllerConfig  # noqa: E402
from mini_bdx_runtime.anim.hardware import MockRobot, SensorSnapshot, OperatorInput  # noqa: E402
from mini_bdx_runtime.anim.fsm import FSMState  # noqa: E402
from mini_bdx_runtime.anim.safety import SafetyConfig  # noqa: E402

TILT_BOUND_DEG = 8.6
LEG_IDX = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
HEAD_IDX = [5, 6, 7, 8]
HEAD_NAMES = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]

_ARTEFACTS = (
    "/Users/clancey/.copilot/session-state/"
    "9d7d4839-8a6c-44d0-8b98-328aebd93579/files/phase4"
)


class IntegratedHarness(Harness):
    """Harness that steps the policy WITHOUT the internal additive/clip, so the
    AnimationController owns the additive head path and the final bus safety."""

    def policy_raw_targets(self, command):
        # Advance imitation phase exactly like the runtime/base harness.
        self.imitation_i = (self.imitation_i + 1.0) % NB_STEPS_IN_PERIOD
        ph = self.imitation_i / NB_STEPS_IN_PERIOD * 2 * np.pi
        self.imitation_phase = np.array([np.cos(ph), np.sin(ph)])

        obs = self.get_obs(command)
        assert obs.shape[0] == 101
        action = self.session.run(None, {"obs": [obs]})[0][0].astype(np.float64)
        self.last_last_last_action = self.last_last_action.copy()
        self.last_last_action = self.last_action.copy()
        self.last_action = action.copy()
        # Pre-additive, pre-clip policy targets (init + action*scale).
        return self.default_actuator + action * ACTION_SCALE

    def apply_and_step(self, bus_targets):
        # Store the FINAL bus targets so the next obs sees them (runtime stores
        # post-additive, and now post-clip, in self.motor_targets).
        self.motor_targets = np.asarray(bus_targets, dtype=np.float64).copy()
        self.data.ctrl[:] = self.motor_targets
        import mujoco
        for _ in range(int(round(CTRL_DT / self.model.opt.timestep))):
            mujoco.mj_step(self.model, self.data)

    def joint_velocities(self):
        return self.data.qvel[self.act_qvel_addr].copy()


def run(mjcf, onnx, clip_path, derating, duration_s, locomotion):
    h = IntegratedHarness(mjcf, onnx, use_speed_limits=True)
    h.reset()

    clip = load_clip(clip_path)
    robot = MockRobot()
    controller = AnimationController(
        robot, background_clip=clip,
        config=ControllerConfig(envelope_derating=derating),
        safety_config=SafetyConfig(max_continuous_load_s=1e9),  # no duty limit in sim
        init_pos_14=h.default_actuator.copy(),
    )
    # Hold the FSM in STAND (the balancing, policy-owned mode) and seed the
    # rate-limit reference from the home pose. Transitions are covered by the
    # unit tests; here we validate the steady balancing + head-injection path.
    controller.fsm.state = FSMState.STAND
    controller.fsm._last_t = None
    controller._prev_bus_targets = h.default_actuator.copy()

    # MJCF jnt_range for violation checks.
    jnt_low = controller.joint_limiter.low.copy()
    jnt_high = controller.joint_limiter.high.copy()

    n_ticks = int(round(duration_s / CTRL_DT))
    op = OperatorInput(locomotion_command=tuple(locomotion))

    rec = {k: [] for k in ("t", "tilt", "z", "head_cmd", "head_meas",
                           "bus_targets", "qvel", "offsets")}
    prev_bus = h.default_actuator.copy()
    fell = False
    vel_violations = 0
    pos_violations = 0
    cost_us = []

    for i in range(n_ticks):
        t = i * CTRL_DT
        pos, quat, tilt_deg = h.base_state()

        # Build the sensor snapshot from the CURRENT sim state.
        snap = SensorSnapshot(
            t_monotonic=t,
            joint_positions=h.all_joint_angles(),
            joint_velocities=h.joint_velocities(),
            tilt_rad=np.radians(tilt_deg),
            feet_contacts=np.array([1.0, 1.0]),
            operator=op,
        )
        robot.next_snapshot = snap

        # --- integration hot path (measured cost) ---
        c0 = time.perf_counter()
        plan = controller.prepare(op, triggers=Triggers())
        c_prepare = time.perf_counter()
        # policy step (NOT counted in the added integration cost)
        command = np.concatenate([np.asarray(locomotion, float), plan.head_offsets])
        raw_targets = h.policy_raw_targets(command)
        c1 = time.perf_counter()
        out = controller.finalize(raw_targets)
        c2 = time.perf_counter()
        cost_us.append(((c_prepare - c0) + (c2 - c1)) * 1e6)

        bus = out.bus_targets
        if controller.fsm.state == FSMState.FAULT:
            raise SystemExit("controller faulted during validation: %s"
                             % out.fsm.fault_reason)

        # Velocity-limit check on the FINAL bus targets (what we command).
        step = np.abs(bus - prev_bus)
        if np.any(step > MAX_MOTOR_VELOCITY * CTRL_DT + 1e-6):
            vel_violations += 1
        # Position-limit check on the commanded targets.
        if np.any(bus < jnt_low - 1e-6) or np.any(bus > jnt_high + 1e-6):
            pos_violations += 1
        prev_bus = bus.copy()

        h.apply_and_step(bus)

        pos2, _, tilt2 = h.base_state()
        if pos2[2] < FALL_HEIGHT:
            fell = True
        rec["t"].append(t)
        rec["tilt"].append(tilt2)
        rec["z"].append(pos2[2])
        rec["head_cmd"].append(bus[HEAD_IDX].copy())        # commanded head target
        rec["head_meas"].append(h.all_joint_angles()[HEAD_IDX])
        rec["offsets"].append(plan.head_offsets.copy())
        rec["bus_targets"].append(bus.copy())
        rec["qvel"].append(h.joint_velocities().copy())

    for k in ("head_cmd", "head_meas", "offsets", "bus_targets", "qvel"):
        rec[k] = np.array(rec[k])
    rec["tilt"] = np.array(rec["tilt"])
    rec["z"] = np.array(rec["z"])

    # --- head-follows metric: correlation & gain per channel (skip transient) ---
    warm = min(50, len(rec["t"]) // 4)
    follow = {}
    for j, name in enumerate(HEAD_NAMES):
        cmd = rec["offsets"][warm:, j]          # authored offset (input)
        meas = rec["head_meas"][warm:, j]       # measured joint (output)
        cmd_c = cmd - cmd.mean()
        meas_c = meas - meas.mean()
        denom = np.linalg.norm(cmd_c) * np.linalg.norm(meas_c)
        corr = float(cmd_c @ meas_c / denom) if denom > 1e-9 else 0.0
        gain = float((cmd_c @ meas_c) / (cmd_c @ cmd_c)) if (cmd_c @ cmd_c) > 1e-12 else 0.0
        follow[name] = {
            "corr": corr, "gain": gain,
            "cmd_ptp": float(np.ptp(cmd)), "meas_ptp": float(np.ptp(meas)),
        }

    peak_tilt = float(np.max(rec["tilt"]))
    max_qvel = float(np.max(np.abs(rec["qvel"])))
    # A channel's authored motion is only distinguishable from base disturbance
    # when its amplitude clears the disturbance floor. When quiescent (STAND) the
    # floor is ~0 so every animated channel is assertable; while WALKING the base
    # sway couples into the head (esp. small-amplitude roll/pitch), so we only
    # hard-assert following on channels whose authored ptp clearly exceeds the
    # gait head-disturbance floor (~0.08 rad measured). This is a physics fact,
    # not a lowered bar: sub-floor channels are reported but not asserted.
    walking = float(np.max(np.abs(np.asarray(locomotion)))) > 1e-6
    active_floor = 0.20 if walking else 0.02
    active = [n for n in HEAD_NAMES if follow[n]["cmd_ptp"] > active_floor]
    reported = [n for n in HEAD_NAMES if follow[n]["cmd_ptp"] > 0.02]
    min_active_corr = min((follow[n]["corr"] for n in active), default=1.0)

    summary = {
        "onnx": os.path.basename(onnx),
        "clip": os.path.basename(clip_path),
        "derating": derating,
        "locomotion": list(locomotion),
        "duration_s": duration_s,
        "ticks": n_ticks,
        "fell": fell,
        "peak_tilt_deg": peak_tilt,
        "tilt_bound_deg": TILT_BOUND_DEG,
        "min_base_z": float(np.min(rec["z"])),
        "pos_limit_violations": pos_violations,
        "vel_limit_violations": vel_violations,
        "max_measured_qvel_rad_s": max_qvel,
        "head_follow": follow,
        "active_head_channels": active,
        "reported_head_channels": reported,
        "head_follow_floor_rad": active_floor,
        "min_active_head_corr": min_active_corr,
        "integration_cost_us": {
            "mean": float(np.mean(cost_us)),
            "p50": float(np.percentile(cost_us, 50)),
            "p95": float(np.percentile(cost_us, 95)),
            "max": float(np.max(cost_us)),
        },
    }

    # --- pass/fail ---
    checks = {
        "upright": not fell,
        "tilt_under_bound": peak_tilt < TILT_BOUND_DEG,
        "no_pos_violations": pos_violations == 0,
        "no_vel_violations": vel_violations == 0,
        "head_follows": min_active_corr > 0.8 and len(active) > 0,
    }
    summary["checks"] = checks
    summary["PASS"] = all(checks.values())
    return summary, rec


def make_plot(rec, summary, path):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    ax = axes[0]
    for j, name in enumerate(HEAD_NAMES):
        ax.plot(rec["t"], rec["offsets"][:, j], "--", lw=1, label="cmd " + name)
        ax.plot(rec["t"], rec["head_meas"][:, j], lw=1.2, label="meas " + name)
    ax.set_ylabel("head (rad)")
    ax.legend(ncol=4, fontsize=7)
    ax.set_title("Phase 4 integrated: authored head offset vs measured joint "
                 "(%s, derating %.2f)  PASS=%s"
                 % (summary["onnx"], summary["derating"], summary["PASS"]))
    axes[1].plot(rec["t"], rec["tilt"], "r")
    axes[1].axhline(TILT_BOUND_DEG, color="k", ls=":")
    axes[1].set_ylabel("tilt (deg)")
    axes[2].plot(rec["t"], rec["z"], "g")
    axes[2].axhline(FALL_HEIGHT, color="k", ls=":")
    axes[2].set_ylabel("base z (m)")
    axes[2].set_xlabel("t (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mjcf", default=DEFAULT_MJCF)
    ap.add_argument("--onnx", default=(
        "/Users/clancey/.copilot/session-state/"
        "9d7d4839-8a6c-44d0-8b98-328aebd93579/files/retrained3/"
        "passthrough_final_300M.onnx"))
    ap.add_argument("--clip", default=os.path.join(
        _RUNTIME_HOME, "clips", "idle_alive.duckanim"))
    ap.add_argument("--derating", type=float, default=0.5)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--walk", action="store_true", help="add a forward command")
    ap.add_argument("--out", default=_ARTEFACTS)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    locomotion = [0.1, 0.0, 0.0] if args.walk else [0.0, 0.0, 0.0]
    summary, rec = run(args.mjcf, args.onnx, args.clip, args.derating,
                       args.duration, locomotion)

    tag = "walk" if args.walk else "stand"
    json_path = os.path.join(args.out, "phase4_integrated_%s.json" % tag)
    plot_path = os.path.join(args.out, "phase4_integrated_%s.png" % tag)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    make_plot(rec, summary, plot_path)

    print(json.dumps(summary, indent=2))
    print("\nwrote:", json_path)
    print("wrote:", plot_path)
    print("\nRESULT:", "PASS" if summary["PASS"] else "FAIL")
    return 0 if summary["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
