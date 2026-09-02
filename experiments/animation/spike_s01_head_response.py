#!/usr/bin/env python3
"""Spike S0.1 - Current-ONNX head-command response characterisation.

Gates the D1 (double-count) decision from docs/animation_system_plan.md sec.7.
Measures whether the deployed walk policy's OWN head output tracks
commands[3:7] when the additive head lines (v2_rl_walk_mujoco.py:310-311)
are REMOVED.

The harness is a faithful re-implementation of the validated Open Duck
Playground MuJoCo closed-loop (playground/open_duck_mini_v2/mujoco_infer.py
+ mujoco_infer_base.py): 50 Hz control, sim_dt=0.002, decimation=10,
obs layout of 101, action_scale=0.25, motor speed limiting, and the
accelerometer[0]+=1.3 bias used by that harness. The walk policy always
receives an advancing imitation phase (nb_steps_in_period=27), exactly as
the runtime does, so "standing" == locomotion command zero with the gait
phase still cycling (a realistic observation).

Modes:
  policy_only  additive head lines REMOVED (the configuration under test)
  additive     head lines PRESENT (current runtime behaviour, for the
               empirical double-count factor)

Conditions:
  stand  [vx,vy,wz] = [0,0,0]
  walk   [vx,vy,wz] = [0.1,0,0]

Measures per head channel: DC gain (dJoint/dCommand), cross-coupling
(off-diagonal / diagonal), sinusoid attenuation & phase lag, and leg /
base disturbance. Emits a machine-readable JSON summary and saves plots.

Pass thresholds (plan sec.7 S0.1): per-channel gain >= 0.6 and
cross-coupling <= 0.2 of the diagonal.
"""

import argparse
import json
import os
import sys

import numpy as np

try:
    import mujoco
except ImportError:
    sys.exit("mujoco not importable - activate the spike venv (see script header).")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import onnxruntime as ort

# ---------------------------------------------------------------------------
# Constants from the plan (Appendix A) and the runtime.
# ---------------------------------------------------------------------------
# 14-DOF hardware / action / actuator order. Head joints are indices 5..8.
HEAD_CHANNELS = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]
HEAD_JOINT_IDX = {"neck_pitch": 5, "head_pitch": 6, "head_yaw": 7, "head_roll": 8}
# command vector is [vx, vy, wz, neck_pitch, head_pitch, head_yaw, head_roll]
HEAD_CMD_IDX = {"neck_pitch": 3, "head_pitch": 4, "head_yaw": 5, "head_roll": 6}
# Training command ranges (joystick.py:94-101 / plan sec.6.3).
CMD_RANGE = {
    "neck_pitch": (-0.34, 1.1),
    "head_pitch": (-0.78, 0.78),
    "head_yaw": (-1.5, 1.5),
    "head_roll": (-0.5, 0.5),
}
ACTION_SCALE = 0.25
SIM_DT = 0.002
DECIMATION = 10
CTRL_DT = SIM_DT * DECIMATION  # 0.02 s -> 50 Hz
MAX_MOTOR_VELOCITY = 5.24  # rad/s
NB_STEPS_IN_PERIOD = 27  # gait period 0.54 s @ 50 Hz (poly coeff pkl)
ACCEL_X_BIAS = 1.3  # matches playground mujoco_infer.py get_obs
FALL_HEIGHT = 0.08  # base z below this => considered fallen (home z = 0.15)

DEFAULT_MJCF = (
    "/Users/clancey/.copilot/session-state/"
    "9d7d4839-8a6c-44d0-8b98-328aebd93579/files/upstream/Open_Duck_Playground/"
    "playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml"
)


class Harness:
    """Faithful closed-loop MuJoCo harness for the deployed ONNX walk policy."""

    def __init__(self, mjcf_path, onnx_path, use_speed_limits=True):
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.model.opt.timestep = SIM_DT
        self.data = mujoco.MjData(self.model)
        self.num_dofs = self.model.nu
        assert self.num_dofs == 14, f"expected 14 actuators, got {self.num_dofs}"
        self.use_speed_limits = use_speed_limits

        self.actuator_names = [self.model.actuator(k).name for k in range(self.model.nu)]
        # qpos / qvel addresses for the actuated joints, in actuator order.
        self.act_qpos_addr = np.array(
            [self.model.jnt_qposadr[self.model.actuator(k).trnid[0]] for k in range(self.model.nu)]
        )
        self.act_qvel_addr = np.array(
            [self.model.jnt_dofadr[self.model.actuator(k).trnid[0]] for k in range(self.model.nu)]
        )

        def sadr(name):
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            return self.model.sensor_adr[sid]

        self.gyro_addr = sadr("gyro")
        self.accel_addr = sadr("accelerometer")

        self.default_actuator = np.array(self.model.keyframe("home").ctrl, dtype=np.float64)
        self.home_qpos = np.array(self.model.keyframe("home").qpos, dtype=np.float64)

        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        i, o = self.session.get_inputs()[0], self.session.get_outputs()[0]
        assert list(i.shape) == [1, 101], f"ONNX input must be [1,101], got {i.shape}"
        assert list(o.shape) == [1, 14], f"ONNX output must be [1,14], got {o.shape}"

        self._floor_id = self.data.body("floor").id
        self._foot_l_id = self.data.body("foot_assembly").id
        self._foot_r_id = self.data.body("foot_assembly_2").id

    # --- helpers -----------------------------------------------------------
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.home_qpos
        self.data.ctrl[:] = self.default_actuator
        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)
        self.motor_targets = self.default_actuator.copy()
        self.prev_motor_targets = self.default_actuator.copy()
        self.imitation_i = 0.0
        self.imitation_phase = np.array([0.0, 0.0])
        mujoco.mj_forward(self.model, self.data)

    def _contact(self, b1, b2):
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
            if (g1 == b1 and g2 == b2) or (g1 == b2 and g2 == b1):
                return 1.0
        return 0.0

    def get_obs(self, command):
        sd = self.data.sensordata
        gyro = sd[self.gyro_addr : self.gyro_addr + 3].copy()
        accel = sd[self.accel_addr : self.accel_addr + 3].copy()
        accel[0] += ACCEL_X_BIAS
        joint_angles = self.data.qpos[self.act_qpos_addr]
        joint_vel = self.data.qvel[self.act_qvel_addr]
        contacts = np.array(
            [
                self._contact(self._foot_l_id, self._floor_id),
                self._contact(self._foot_r_id, self._floor_id),
            ]
        )
        obs = np.concatenate(
            [
                gyro,
                accel,
                command,
                joint_angles - self.default_actuator,
                joint_vel * 0.05,
                self.last_action,
                self.last_last_action,
                self.last_last_last_action,
                self.motor_targets,
                contacts,
                self.imitation_phase,
            ]
        )
        return obs

    def control_tick(self, command, mode):
        # advance imitation phase every tick (walk policy, like the runtime)
        self.imitation_i = (self.imitation_i + 1.0) % NB_STEPS_IN_PERIOD
        ph = self.imitation_i / NB_STEPS_IN_PERIOD * 2 * np.pi
        self.imitation_phase = np.array([np.cos(ph), np.sin(ph)])

        obs = self.get_obs(command)
        action = self.session.run(None, {"obs": [obs]})[0][0].astype(np.float64)

        self.last_last_last_action = self.last_last_action.copy()
        self.last_last_action = self.last_action.copy()
        self.last_action = action.copy()

        self.motor_targets = self.default_actuator + action * ACTION_SCALE
        if self.use_speed_limits:
            lo = self.prev_motor_targets - MAX_MOTOR_VELOCITY * CTRL_DT
            hi = self.prev_motor_targets + MAX_MOTOR_VELOCITY * CTRL_DT
            self.motor_targets = np.clip(self.motor_targets, lo, hi)
        # prev tracks the pre-additive target (matches runtime line ordering)
        self.prev_motor_targets = self.motor_targets.copy()

        if mode == "additive":
            # v2_rl_walk_mujoco.py:310-311 - the lines under test
            self.motor_targets[5:9] = np.asarray(command)[3:7] + self.motor_targets[5:9]

        self.data.ctrl[:] = self.motor_targets
        for _ in range(DECIMATION):
            mujoco.mj_step(self.model, self.data)

    def head_joint_angles(self):
        return self.data.qpos[self.act_qpos_addr[[5, 6, 7, 8]]].copy()

    def all_joint_angles(self):
        return self.data.qpos[self.act_qpos_addr].copy()

    def base_state(self):
        q = self.home_qpos  # placeholder unused
        pos = self.data.qpos[0:3].copy()
        quat = self.data.qpos[3:7].copy()  # w,x,y,z
        # tilt: angle of body z-axis from world vertical
        w, x, y, z = quat / (np.linalg.norm(quat) + 1e-12)
        # body-z expressed in world = 3rd column of rotation matrix
        bz = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
        tilt = np.degrees(np.arccos(np.clip(bz[2], -1, 1)))
        return pos, quat, tilt


# ---------------------------------------------------------------------------
# Simulation drivers
# ---------------------------------------------------------------------------
def base_locomotion(condition):
    return [0.1, 0.0, 0.0] if condition == "walk" else [0.0, 0.0, 0.0]


def run_static(h, mode, condition, driven_channel, cmd_value, settle_ticks, sample_periods):
    """Hold a static head command; return steady-state means and disturbance."""
    h.reset()
    command = np.zeros(7)
    command[0:3] = base_locomotion(condition)
    if driven_channel is not None:
        command[HEAD_CMD_IDX[driven_channel]] = cmd_value

    sample_ticks = sample_periods * NB_STEPS_IN_PERIOD
    head_hist, leg_hist, tilt_hist, z_hist = [], [], [], []
    fell = False
    for t in range(settle_ticks):
        h.control_tick(command, mode)
        pos, quat, tilt = h.base_state()
        if pos[2] < FALL_HEIGHT:
            fell = True
        if t >= settle_ticks - 2 * sample_ticks:
            head_hist.append(h.head_joint_angles())
            legs = h.all_joint_angles()[[0, 1, 2, 3, 4, 9, 10, 11, 12, 13]]
            leg_hist.append(legs)
            tilt_hist.append(tilt)
            z_hist.append(pos[2])
    head_hist = np.array(head_hist)  # (2*sample_ticks, 4)
    leg_hist = np.array(leg_hist)
    win = head_hist[-sample_ticks:]
    prev = head_hist[-2 * sample_ticks : -sample_ticks]
    drift = float(np.max(np.abs(win.mean(0) - prev.mean(0))))  # settling metric
    ripple = float(np.max(win.std(0)))
    return {
        "head_mean": win.mean(0).tolist(),  # [neck,hp,hy,hr] joint rad
        "head_ripple_std": ripple,
        "settle_drift": drift,
        "leg_mean": leg_hist[-sample_ticks:].mean(0).tolist(),
        "tilt_mean_deg": float(np.mean(tilt_hist[-sample_ticks:])),
        "tilt_max_deg": float(np.max(tilt_hist)),
        "base_z_min": float(np.min(z_hist)),
        "fell": fell,
    }


def run_sine(h, mode, condition, channel, freq, transient_s, meas_periods):
    """Drive a sinusoid on one channel; fit amplitude & phase of the response."""
    h.reset()
    lo, hi = CMD_RANGE[channel]
    center = 0.5 * (lo + hi)
    amp = 0.25 * (hi - lo)  # peak-to-peak = 50% of range
    jidx_in_head = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"].index(channel)

    transient_ticks = int(round(transient_s / CTRL_DT))
    meas_ticks = int(round(meas_periods / freq / CTRL_DT))
    cmd_series, joint_series, tvec = [], [], []
    command = np.zeros(7)
    command[0:3] = base_locomotion(condition)
    for t in range(transient_ticks + meas_ticks):
        tt = t * CTRL_DT
        c = center + amp * np.sin(2 * np.pi * freq * tt)
        command[HEAD_CMD_IDX[channel]] = c
        h.control_tick(command, mode)
        if t >= transient_ticks:
            cmd_series.append(c)
            joint_series.append(h.head_joint_angles()[jidx_in_head])
            tvec.append(tt)
    tvec = np.array(tvec)
    joint_series = np.array(joint_series)
    # least-squares fit joint(t) = A cos(wt) + B sin(wt) + D
    w = 2 * np.pi * freq
    M = np.column_stack([np.cos(w * tvec), np.sin(w * tvec), np.ones_like(tvec)])
    coef, *_ = np.linalg.lstsq(M, joint_series, rcond=None)
    A, B, _D = coef
    resp_amp = float(np.hypot(A, B))
    resp_phase = float(np.arctan2(-A, B))  # phase of A cos + B sin as sin(wt+phi)... see note
    # command is amp*sin(wt) -> phase 0. response sin phase:
    # A cos + B sin = R sin(wt + phi) with phi = atan2(A, B)
    resp_phase = float(np.arctan2(A, B))
    phase_lag_deg = float(-np.degrees(resp_phase))  # lag positive if response delayed
    # wrap to (-180,180]
    phase_lag_deg = (phase_lag_deg + 180) % 360 - 180
    attenuation = resp_amp / amp
    return {
        "freq_hz": freq,
        "cmd_amp": amp,
        "resp_amp": resp_amp,
        "attenuation": attenuation,
        "phase_lag_deg": phase_lag_deg,
        "t": tvec.tolist(),
        "cmd": cmd_series,
        "joint": joint_series.tolist(),
    }


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def experiment_dc(h, mode, condition, n_points, settle_ticks, sample_periods, plotdir):
    results = {}
    # baseline (all head commands zero) for sanity + subtraction
    sweep_data = {}
    for ch in HEAD_CHANNELS:
        lo, hi = CMD_RANGE[ch]
        cmds = np.linspace(lo, hi, n_points)
        rows = []
        for cv in cmds:
            r = run_static(h, mode, condition, ch, cv, settle_ticks, sample_periods)
            rows.append(r)
        sweep_data[ch] = {"cmds": cmds.tolist(), "rows": rows}

    # Fit gains + cross-coupling.
    for ch in HEAD_CHANNELS:
        cmds = np.array(sweep_data[ch]["cmds"])
        heads = np.array([row["head_mean"] for row in sweep_data[ch]["rows"]])  # (n,4)
        legs = np.array([row["leg_mean"] for row in sweep_data[ch]["rows"]])  # (n,10)
        diag_idx = HEAD_CHANNELS.index(ch)
        # linear slope joint vs command for each of the 4 head joints
        slopes = []
        for j in range(4):
            A = np.column_stack([cmds, np.ones_like(cmds)])
            m, _b = np.linalg.lstsq(A, heads[:, j], rcond=None)[0]
            slopes.append(float(m))
        diag_gain = slopes[diag_idx]
        # leg disturbance: peak-to-peak of each leg joint's steady mean across the
        # command sweep (how much the legs move as this head command varies).
        leg_excursion = float(np.max(np.ptp(legs, axis=0)))
        cross = {
            HEAD_CHANNELS[j]: (abs(slopes[j]) / abs(diag_gain) if abs(diag_gain) > 1e-9 else float("inf"))
            for j in range(4)
            if j != diag_idx
        }
        max_cross = max(cross.values()) if cross else 0.0
        results[ch] = {
            "dc_gain": diag_gain,
            "gain_pass": bool(diag_gain >= 0.6),
            "per_joint_slope": {HEAD_CHANNELS[j]: slopes[j] for j in range(4)},
            "max_abs_offdiag_slope": float(max(abs(slopes[j]) for j in range(4) if j != diag_idx)),
            "cross_coupling": cross,
            "max_cross_coupling": max_cross,
            "cross_pass": bool(max_cross <= 0.2),
            "cross_meaningful": bool(abs(diag_gain) >= 0.1),
            "leg_excursion_ptp_rad": leg_excursion,
            "max_ripple_std": float(np.max([row["head_ripple_std"] for row in sweep_data[ch]["rows"]])),
            "max_settle_drift": float(np.max([row["settle_drift"] for row in sweep_data[ch]["rows"]])),
            "max_base_tilt_deg": float(np.max([row["tilt_max_deg"] for row in sweep_data[ch]["rows"]])),
            "min_base_z": float(np.min([row["base_z_min"] for row in sweep_data[ch]["rows"]])),
            "any_fell": bool(any(row["fell"] for row in sweep_data[ch]["rows"])),
        }

    _plot_dc(sweep_data, mode, condition, plotdir)
    return results, sweep_data


def _plot_dc(sweep_data, mode, condition, plotdir):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, ch in zip(axes.ravel(), HEAD_CHANNELS):
        cmds = np.array(sweep_data[ch]["cmds"])
        heads = np.array([row["head_mean"] for row in sweep_data[ch]["rows"]])
        diag_idx = HEAD_CHANNELS.index(ch)
        ax.plot(cmds, cmds, "k--", lw=1, label="ideal (gain=1)")
        for j, cj in enumerate(HEAD_CHANNELS):
            style = "-o" if j == diag_idx else "-."
            lw = 2 if j == diag_idx else 1
            ax.plot(cmds, heads[:, j], style, lw=lw, ms=4, label=f"{cj} joint")
        ax.set_title(f"drive {ch}")
        ax.set_xlabel("command (rad)")
        ax.set_ylabel("steady joint (rad)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(f"S0.1 DC sweep - mode={mode} condition={condition}")
    fig.tight_layout()
    p = os.path.join(plotdir, f"dc_sweep_{mode}_{condition}.png")
    fig.savefig(p, dpi=110)
    plt.close(fig)


def experiment_sine(h, mode, condition, freqs, transient_s, meas_periods, plotdir):
    results = {}
    series_for_plot = {}
    for ch in HEAD_CHANNELS:
        results[ch] = {}
        for f in freqs:
            r = run_sine(h, mode, condition, ch, f, transient_s, meas_periods)
            results[ch][f"{f}Hz"] = {
                "attenuation": r["attenuation"],
                "phase_lag_deg": r["phase_lag_deg"],
                "resp_amp": r["resp_amp"],
                "cmd_amp": r["cmd_amp"],
            }
            if abs(f - freqs[0]) < 1e-9:
                series_for_plot[ch] = r
    _plot_sine(series_for_plot, mode, condition, freqs[0], plotdir)
    return results


def _plot_sine(series, mode, condition, freq, plotdir):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, ch in zip(axes.ravel(), HEAD_CHANNELS):
        r = series[ch]
        ax.plot(r["t"], r["cmd"], label="command", lw=1.5)
        ax.plot(r["t"], r["joint"], label="joint", lw=1.5)
        ax.set_title(f"{ch}  atten={r['attenuation']:.2f} lag={r['phase_lag_deg']:.0f}deg")
        ax.set_xlabel("t (s)")
        ax.set_ylabel("rad")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(f"S0.1 sinusoid {freq}Hz - mode={mode} condition={condition}")
    fig.tight_layout()
    p = os.path.join(plotdir, f"sine_{freq}Hz_{mode}_{condition}.png")
    fig.savefig(p, dpi=110)
    plt.close(fig)


def sanity_check(h, mode, condition, settle_ticks, sample_periods):
    r = run_static(h, mode, condition, None, 0.0, settle_ticks, sample_periods)
    head = np.array(r["head_mean"])
    return {
        "zero_cmd_head_mean_rad": r["head_mean"],
        "zero_cmd_head_abs_max_rad": float(np.max(np.abs(head))),
        "base_tilt_mean_deg": r["tilt_mean_deg"],
        "base_tilt_max_deg": r["tilt_max_deg"],
        "base_z_min": r["base_z_min"],
        "fell": r["fell"],
        "stable": bool((not r["fell"]) and r["tilt_max_deg"] < 30.0),
    }


# ---------------------------------------------------------------------------
def run_one(onnx, mjcf, mode, condition, cfg, plotdir):
    h = Harness(mjcf, onnx)
    settle_ticks = int(round(cfg["settle_s"] / CTRL_DT))
    sanity = sanity_check(h, mode, condition, settle_ticks, cfg["sample_periods"])
    dc, _ = experiment_dc(
        h, mode, condition, cfg["dc_points"], settle_ticks, cfg["sample_periods"], plotdir
    )
    sine = experiment_sine(
        h, mode, condition, cfg["freqs"], cfg["sine_transient_s"], cfg["sine_periods"], plotdir
    )
    verdict = {
        ch: {
            "gain": dc[ch]["dc_gain"],
            "gain_pass": dc[ch]["gain_pass"],
            "max_cross_coupling": dc[ch]["max_cross_coupling"],
            "cross_pass": dc[ch]["cross_pass"],
        }
        for ch in HEAD_CHANNELS
    }
    overall_pass = all(v["gain_pass"] and v["cross_pass"] for v in verdict.values())
    return {
        "mode": mode,
        "condition": condition,
        "onnx": os.path.basename(onnx),
        "mjcf": mjcf,
        "config": cfg,
        "sanity": sanity,
        "dc": dc,
        "sine": sine,
        "verdict_per_channel": verdict,
        "overall_pass": overall_pass,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", default="BEST_WALK_ONNX_2.onnx")
    ap.add_argument("--mjcf", default=DEFAULT_MJCF)
    ap.add_argument("--mode", choices=["policy_only", "additive"], default="policy_only")
    ap.add_argument("--condition", choices=["stand", "walk"], default="stand")
    ap.add_argument("--all", action="store_true", help="run all mode x condition combos")
    ap.add_argument("--outdir", default=".", help="dir for JSON + PNG output")
    ap.add_argument("--settle-s", type=float, default=3.0)
    ap.add_argument("--sample-periods", type=int, default=2, help="gait periods to average steady state")
    ap.add_argument("--dc-points", type=int, default=9)
    ap.add_argument("--freqs", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--sine-transient-s", type=float, default=2.0)
    ap.add_argument("--sine-periods", type=float, default=4.0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cfg = {
        "settle_s": args.settle_s,
        "sample_periods": args.sample_periods,
        "dc_points": args.dc_points,
        "freqs": args.freqs,
        "sine_transient_s": args.sine_transient_s,
        "sine_periods": args.sine_periods,
        "action_scale": ACTION_SCALE,
        "sim_dt": SIM_DT,
        "decimation": DECIMATION,
        "nb_steps_in_period": NB_STEPS_IN_PERIOD,
        "use_speed_limits": True,
        "accel_x_bias": ACCEL_X_BIAS,
    }

    combos = (
        [(m, c) for m in ("policy_only", "additive") for c in ("stand", "walk")]
        if args.all
        else [(args.mode, args.condition)]
    )
    out = {"spike": "S0.1", "results": []}
    for mode, condition in combos:
        sys.stderr.write(f"[S0.1] running mode={mode} condition={condition} ...\n")
        sys.stderr.flush()
        res = run_one(args.onnx, args.mjcf, mode, condition, cfg, args.outdir)
        out["results"].append(res)
        # per-run JSON too
        with open(os.path.join(args.outdir, f"s01_{mode}_{condition}.json"), "w") as f:
            json.dump(res, f, indent=2)

    with open(os.path.join(args.outdir, "s01_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    # compact machine-readable summary to stdout
    print(json.dumps(_compact(out), indent=2))


def _compact(out):
    c = {"spike": "S0.1", "runs": []}
    for r in out["results"]:
        c["runs"].append(
            {
                "mode": r["mode"],
                "condition": r["condition"],
                "overall_pass": r["overall_pass"],
                "sanity_stable": r["sanity"]["stable"],
                "channels": {
                    ch: {
                        "gain": round(r["dc"][ch]["dc_gain"], 4),
                        "gain_pass": r["dc"][ch]["gain_pass"],
                        "max_cross": round(r["dc"][ch]["max_cross_coupling"], 4),
                        "cross_pass": r["dc"][ch]["cross_pass"],
                        "atten": {
                            k: round(v["attenuation"], 3) for k, v in r["sine"][ch].items()
                        },
                        "phase_lag_deg": {
                            k: round(v["phase_lag_deg"], 1) for k, v in r["sine"][ch].items()
                        },
                    }
                    for ch in HEAD_CHANNELS
                },
            }
        )
    return c


if __name__ == "__main__":
    main()
