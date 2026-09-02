#!/usr/bin/env python3
"""Envelope sweep - empirical safe head operating envelope (defect D13 / risk R16).

Gates the D13/R16 safety finding from ``docs/animation_system_plan.md`` (§3.4,
§6.5, §7 Phase 3/4): in the *additive* head mode that ships today
(``v2_rl_walk_mujoco.py:310-311``), step inputs at ``neck_pitch``/``head_yaw``
range extremes topple the robot. Before any authored animation drives the head
on real hardware we must know the safe envelope.

This script BUILDS ON the validated S0.1 harness
(:mod:`spike_s01_head_response`) rather than re-implementing the closed loop, so
the numbers stay consistent with the S0.1 measurements (same 50 Hz control,
``sim_dt=0.002``, decimation 10, ``action_scale=0.25``, obs 101, accel[0]+=1.3,
motor speed limit 5.24 rad/s, advancing imitation phase). Its zero-command
sanity check reproduces S0.1 (stand tilt ~5.6 deg, z~0.16).

It derives, empirically and conservatively, for additive mode in both STAND and
WALK:

1. Static deflection limit  (fine outward sweep + long hold; first-onset rule).
2. Slew-rate limit           (fastest command slew that does not destabilise).
3. Combined / worst case     (multi-axis; L2 budget across channels).
4. Sustained / oscillatory   (continuous sinusoid within the derived envelope).

WHY A FINE SWEEP AND NOT BISECTION.  The additive instability is *non-monotonic*
and *time-dependent*: for neck_pitch/stand, +0.7..+0.9 topple but +1.1 is a
(unreachable) stable island, and a fall can take up to ~3.4 s to develop.
Bisection assumes a single monotone threshold and would report the island as
"safe". We therefore sweep outward from nominal (0) in fine increments, hold each
level long enough for slow divergence to manifest, and take the FIRST onset of
instability as the failure threshold. Anything at or beyond a stable island past
that onset is treated as unsafe because it cannot be reached without crossing the
unstable band.

STABILITY CRITERION ("stably upright with margin"), stated up front and used
consistently for every classification:

    UPRIGHT  <=>  (never fell: base z >= 0.08 m at all times)
              AND (peak base tilt <= 8.6 deg == 0.15 rad  over the run)
              AND (min base z >= 0.12 m over the run)

The 8.6 deg / 0.15 rad tilt bound is the plan's own Phase 4 hardware-acceptance
bound; the zero-command baseline (stand 5.6 deg, walk 3.1 deg) sits comfortably
below it, so it leaves genuine margin while still flagging a lean well before an
actual topple. The safe limit is then a conservative fraction (default 0.5) of
the measured failure threshold.

Outputs a machine-readable JSON summary and PNG plots. Raw artefacts are written
OUTSIDE the repo (pass --outdir).

Re-runnable, e.g.::

    python envelope_sweep.py --experiment all \
        --outdir /path/outside/repo/envelope
"""

import argparse
import json
import os
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the validated S0.1 harness and its constants verbatim.
import spike_s01_head_response as s01

# The deliverable envelope module (repo root two levels up from this file).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from open_duck_anim import envelope as odenv  # noqa: E402

HEAD_CHANNELS = s01.HEAD_CHANNELS  # [neck_pitch, head_pitch, head_yaw, head_roll]
HEAD_CMD_IDX = s01.HEAD_CMD_IDX
CMD_RANGE = s01.CMD_RANGE
CTRL_DT = s01.CTRL_DT
FALL_HEIGHT = s01.FALL_HEIGHT

# Leg joint indices within the 14-DOF actuator/qpos vector (head is 5..8).
LEG_IDX = np.array([0, 1, 2, 3, 4, 9, 10, 11, 12, 13])

# --- Stability criterion (see module docstring) ------------------------------
STABLE_TILT_DEG = 8.6   # 0.15 rad; plan Phase 4 acceptance bound
STABLE_Z_MIN = 0.12     # m; home z ~= 0.15, allow ~20% squat
# base z below this at any instant == a fall (harness constant, home z ~= 0.15).

# Conservative fraction of the measured failure threshold used as the safe limit.
SAFETY_FRACTION = 0.5


# ---------------------------------------------------------------------------
# Core driver
# ---------------------------------------------------------------------------
def _leg_baseline(h, mode, condition, settle_ticks=100):
    """Settled leg pose under zero head command (for excursion measurement)."""
    h.reset()
    command = np.zeros(7)
    command[0:3] = s01.base_locomotion(condition)
    for _ in range(settle_ticks):
        h.control_tick(command, mode)
    return h.all_joint_angles()[LEG_IDX].copy()


def simulate(h, mode, condition, cmd_of_tick, n_ticks, leg_base, eval_from=0):
    """Run ``n_ticks`` control ticks driving ``cmd_of_tick(tick) -> head[4]``.

    Returns a metrics dict over the evaluation window ``[eval_from:]`` plus a
    fall flag that is latched over the WHOLE run (a fall is never masked by the
    eval window).
    """
    h.reset()
    base = np.zeros(7)
    base[0:3] = s01.base_locomotion(condition)
    tilt_hist, z_hist, leg_exc_hist = [], [], []
    fell = False
    for t in range(n_ticks):
        head = cmd_of_tick(t)
        command = base.copy()
        command[3:7] = head
        h.control_tick(command, mode)
        pos, _quat, tilt = h.base_state()
        if pos[2] < FALL_HEIGHT:
            fell = True
        if t >= eval_from:
            tilt_hist.append(tilt)
            z_hist.append(pos[2])
            legs = h.all_joint_angles()[LEG_IDX]
            leg_exc_hist.append(float(np.max(np.abs(legs - leg_base))))
    tilt_hist = np.asarray(tilt_hist)
    z_hist = np.asarray(z_hist)
    return {
        "tilt_max_deg": float(np.max(tilt_hist)),
        "tilt_mean_deg": float(np.mean(tilt_hist)),
        "z_min": float(np.min(z_hist)),
        "leg_peak_excursion_rad": float(np.max(leg_exc_hist)),
        "fell": bool(fell),
    }


def is_upright(m):
    """The single stability predicate used for every classification."""
    return (
        (not m["fell"])
        and (m["tilt_max_deg"] <= STABLE_TILT_DEG)
        and (m["z_min"] >= STABLE_Z_MIN)
    )


def ramp_hold_cmd(channel, target, ramp_ticks, hold_ticks):
    """Command factory: ramp one head channel 0->target then hold."""
    j = HEAD_CHANNELS.index(channel)

    def fn(t):
        head = np.zeros(4)
        frac = min(1.0, (t + 1) / ramp_ticks) if ramp_ticks > 0 else 1.0
        head[j] = target * frac
        return head

    return fn, ramp_ticks + hold_ticks


# ---------------------------------------------------------------------------
# Experiment 1 - static deflection limit (fine outward sweep, first onset)
# ---------------------------------------------------------------------------
def sweep_direction(h, mode, condition, channel, sign, leg_base, cfg):
    """Genuinely fine outward sweep on one channel toward one range edge.

    Returns (failure_onset, last_stable, samples). ``failure_onset`` is None if
    the whole trained range on that side stays upright.

    Reviewer E10: this is a SINGLE monotone outward sweep at ``fine_step``
    resolution from 0 to the range edge, stopping at the FIRST magnitude that
    fails (first-onset, consistent with E8). A coarse-then-refine scheme can jump
    over an unstable band narrower than the coarse step (both coarse endpoints
    stable) and never revisit it; a genuine fine sweep cannot skip any band wider
    than ``fine_step``. ``fine_step`` is therefore the honest resolution limit of
    the derived onset — choose it smaller than the narrowest instability you are
    willing to miss (default 0.01 rad; see PROVENANCE in envelope.py).
    """
    lo, hi = CMD_RANGE[channel]
    edge = hi if sign > 0 else lo
    if edge == 0.0:
        return None, 0.0, []
    ramp_ticks = int(round(cfg["ramp_s"] / CTRL_DT))
    hold_ticks = int(round(cfg["hold_s"] / CTRL_DT))

    samples = []

    def test(mag):
        target = sign * mag
        fn, n = ramp_hold_cmd(channel, target, ramp_ticks, hold_ticks)
        m = simulate(h, mode, condition, fn, n, leg_base)
        m["command"] = float(target)
        samples.append(m)
        return is_upright(m)

    span = abs(edge)
    step = cfg["fine_step"]
    grid = list(np.arange(step, span + 1e-9, step))
    if not grid or grid[-1] < span - 1e-9:
        grid.append(span)
    last_stable = 0.0
    for mag in grid:
        if test(mag):
            last_stable = float(mag)
        else:
            return float(mag), last_stable, samples  # first onset
    return None, span, samples  # whole side is upright


def experiment_static(h, mode, condition, leg_base, cfg, plotdir):
    result = {}
    for ch in HEAD_CHANNELS:
        lo, hi = CMD_RANGE[ch]
        chan = {"range": [lo, hi], "directions": {}}
        for name, sign in (("pos", +1), ("neg", -1)):
            onset, last_stable, samples = sweep_direction(
                h, mode, condition, ch, sign, leg_base, cfg
            )
            if onset is None:
                safe = sign * abs(hi if sign > 0 else lo)
                chan["directions"][name] = {
                    "failure_onset": None,
                    "last_stable": float(sign * last_stable),
                    "safe_limit": float(safe),
                    "note": "no topple within trained range on this side",
                    "samples": samples,
                }
            else:
                safe_mag = min(SAFETY_FRACTION * onset, last_stable)
                chan["directions"][name] = {
                    "failure_onset": float(sign * onset),
                    "last_stable": float(sign * last_stable),
                    "safe_limit": float(sign * safe_mag),
                    "samples": samples,
                }
        result[ch] = chan
    _plot_static(result, mode, condition, plotdir)
    return result


def _plot_static(result, mode, condition, plotdir):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, ch in zip(axes.ravel(), HEAD_CHANNELS):
        cmds, tilts, falls = [], [], []
        for name in ("pos", "neg"):
            for s in result[ch]["directions"][name]["samples"]:
                cmds.append(s["command"])
                tilts.append(min(s["tilt_max_deg"], 90.0))
                falls.append(s["fell"])
        order = np.argsort(cmds)
        cmds = np.asarray(cmds)[order]
        tilts = np.asarray(tilts)[order]
        falls = np.asarray(falls)[order]
        ax.plot(cmds, tilts, "-o", ms=3, lw=1)
        if falls.any():
            ax.plot(cmds[falls], tilts[falls], "rx", ms=8, label="fell")
        ax.axhline(STABLE_TILT_DEG, color="k", ls="--", lw=1, label="upright bound")
        for name in ("pos", "neg"):
            sl = result[ch]["directions"][name]["safe_limit"]
            ax.axvline(sl, color="g", ls=":", lw=1)
        ax.set_title(ch)
        ax.set_xlabel("command (rad)")
        ax.set_ylabel("peak tilt (deg, clipped 90)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(f"Static deflection sweep - additive {condition}")
    fig.tight_layout()
    fig.savefig(os.path.join(plotdir, f"static_{mode}_{condition}.png"), dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment 2 - slew-rate limit
# ---------------------------------------------------------------------------
def experiment_slew(h, mode, condition, static_result, leg_base, cfg, plotdir):
    """At a deflection safely inside the static limit, find the fastest command
    slew (rad/s) that stays upright. Uses the more dangerous sign per channel."""
    result = {}
    hold_ticks = int(round(cfg["slew_hold_s"] / CTRL_DT))
    slews = cfg["slew_rates"]
    for ch in HEAD_CHANNELS:
        # amplitude = the (signed) safe static limit whose magnitude is smaller
        # (more dangerous side); this is the amplitude the animator may actually
        # command, so slew must be safe *to* it.
        dirs = static_result[ch]["directions"]
        cand = [dirs["pos"]["safe_limit"], dirs["neg"]["safe_limit"]]
        cand = [c for c in cand if abs(c) > 1e-6]
        if not cand:
            result[ch] = {"amplitude": 0.0, "max_safe_slew": None, "samples": []}
            continue
        amp = min(cand, key=abs)  # smallest-magnitude safe limit, signed
        samples = []
        max_safe = None
        # First-onset (reviewer E8): sweep slew from SLOW→FAST and stop at the
        # first rate that destabilises; the adopted limit is the last upright
        # rate strictly below that onset. Taking max-over-upright instead could
        # jump a non-monotone unstable band and land on a faster "stable island"
        # above a rate that actually falls — exactly the trap the module warns of.
        onset = None
        for slew in sorted(slews):
            rise_ticks = max(1, int(round(abs(amp) / slew / CTRL_DT)))
            fn, _ = ramp_hold_cmd(ch, amp, rise_ticks, hold_ticks)
            n = rise_ticks + hold_ticks
            m = simulate(h, mode, condition, fn, n, leg_base)
            m["slew"] = float(slew)
            m["rise_ticks"] = rise_ticks
            samples.append(m)
            if is_upright(m):
                max_safe = float(slew)
            else:
                onset = float(slew)
                break
        result[ch] = {
            "amplitude": float(amp),
            "max_safe_slew": max_safe,
            "failure_onset_slew": onset,
            "slew_binding": bool(onset is not None),
            "samples": samples,
        }
    _plot_slew(result, mode, condition, plotdir)
    return result


def _plot_slew(result, mode, condition, plotdir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for ch in HEAD_CHANNELS:
        s = result[ch]["samples"]
        if not s:
            continue
        xs = [d["slew"] for d in s]
        ys = [min(d["tilt_max_deg"], 90.0) for d in s]
        ax.plot(xs, ys, "-o", ms=3, label=f"{ch} (amp={result[ch]['amplitude']:+.2f})")
    ax.axhline(STABLE_TILT_DEG, color="k", ls="--", lw=1, label="upright bound")
    ax.set_xlabel("command slew (rad/s)")
    ax.set_ylabel("peak tilt (deg, clipped 90)")
    ax.set_xscale("log")
    ax.set_title(f"Slew sweep at safe amplitude - additive {condition}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(plotdir, f"slew_{mode}_{condition}.png"), dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment 3 - combined / worst case (multi-axis L2 budget)
# ---------------------------------------------------------------------------
def _safe_mag_vec(static_result):
    """Per-channel safe *magnitude* on the more dangerous side (smaller |limit|).

    These are the L_i used to normalise the combined L2 budget: a command is
    scored as n_i = c_i / L_i, so the tighter (more dangerous) channels weigh
    more heavily in the budget.
    """
    L = {}
    for ch in HEAD_CHANNELS:
        dirs = static_result[ch]["directions"]
        cand = [abs(dirs["pos"]["safe_limit"]), abs(dirs["neg"]["safe_limit"])]
        cand = [c for c in cand if c > 1e-6]
        L[ch] = float(min(cand)) if cand else 0.0
    return L


def _osc_worstcase(h, mode, condition, chans, namp, freqs, phase_set, leg_base, dur_s):
    """Worst tilt / any-fall over freq x phase for a multi-axis sinusoid.

    ``namp`` maps channel -> normalised amplitude (command = namp * L_i). Every
    combination of ``phase_set`` across the driven channels is tried (this is how
    the phase-dependent resonances are found); returns early on the first fall.
    """
    import itertools

    idxs = [HEAD_CHANNELS.index(c) for c in chans]
    L = namp["_L"]
    amps = np.array([namp[c] * L[c] for c in chans])
    n_ticks = int(round(dur_s / CTRL_DT))
    worst = 0.0
    fell = False
    bad = None
    for f in freqs:
        for phases in itertools.product(phase_set, repeat=len(chans)):
            ph = np.array(phases)

            def fn(t, f=f, ph=ph):
                head = np.zeros(4)
                w = 2 * np.pi * f * (t * CTRL_DT)
                for k, ix in enumerate(idxs):
                    head[ix] = amps[k] * np.sin(w + ph[k])
                return head

            m = simulate(h, mode, condition, fn, n_ticks, leg_base)
            worst = max(worst, m["tilt_max_deg"])
            if m["fell"] or m["tilt_max_deg"] > STABLE_TILT_DEG:
                fell = m["fell"]
                bad = {"freq": float(f), "phases": [float(p) for p in phases],
                       "tilt": m["tilt_max_deg"], "fell": m["fell"]}
                if m["fell"]:
                    return worst, True, bad
    return worst, fell, bad


def experiment_combined(h, mode, condition, static_result, leg_base, cfg):
    """Combined / worst-case multi-axis limit.

    Two parts:

    * ``static`` — ramp+hold several channels simultaneously at scale * their own
      safe limit; shows whether the *static* per-axis limits are jointly safe.
    * ``oscillatory`` — the binding case. Drive multiple channels sinusoidally at
      a shared L2 magnitude ``B`` (each channel scaled so ``||c/L||_2 == B``) and
      search freq x phase for the worst tilt / any fall. This exposes the
      phase/frequency resonances that the static test misses (a within-per-axis-
      safe two-axis quadrature oscillation can still topple). We report the
      largest ``B`` for which the worst case stays upright-with-margin.
    """
    L = _safe_mag_vec(static_result)
    # dangerous-sign safe vector for the static probe (signed).
    safe_vec = {}
    for ch in HEAD_CHANNELS:
        dirs = static_result[ch]["directions"]
        cand = [dirs["pos"]["safe_limit"], dirs["neg"]["safe_limit"]]
        cand = [c for c in cand if abs(c) > 1e-6]
        safe_vec[ch] = min(cand, key=abs) if cand else 0.0

    directions = {
        "neck+yaw": ["neck_pitch", "head_yaw"],   # realistic "look toward" turn
        "all4": HEAD_CHANNELS,                     # adversarial worst case
    }
    ramp_ticks = int(round(cfg["ramp_s"] / CTRL_DT))
    hold_ticks = int(round(cfg["hold_s"] / CTRL_DT))
    freqs = cfg["combined_freqs"]
    phase_set = cfg["combined_phases"]

    out = {"L_normalisation": L, "static": {}, "oscillatory": {}}
    for dname, chans in directions.items():
        idxs = [HEAD_CHANNELS.index(c) for c in chans]
        vals = np.array([safe_vec[c] for c in chans])

        # --- static ramp+hold at scale * per-axis safe limit ----------------
        # First-onset (reviewer E8): sweep scale ascending, stop at first failure.
        samples = []
        max_scale = 0.0
        onset_scale = None
        for scale in sorted(cfg["combined_scales"]):
            def fn(t, scale=scale):
                head = np.zeros(4)
                frac = min(1.0, (t + 1) / ramp_ticks) if ramp_ticks > 0 else 1.0
                for k, ix in enumerate(idxs):
                    head[ix] = scale * vals[k] * frac
                return head

            m = simulate(h, mode, condition, fn, ramp_ticks + hold_ticks, leg_base)
            m["scale"] = float(scale)
            samples.append(m)
            if is_upright(m):
                max_scale = float(scale)
            else:
                onset_scale = float(scale)
                break
        out["static"][dname] = {
            "channels": chans,
            "safe_limits_used": {c: float(safe_vec[c]) for c in chans},
            "max_safe_scale_static": max_scale,
            "failure_onset_scale_static": onset_scale,
            "per_axis_jointly_safe_static": bool(max_scale >= 1.0),
            "samples": samples,
        }

        # --- oscillatory L2-budget search -----------------------------------
        # First-onset (reviewer E8): sweep the L2 budget ascending and stop at the
        # first B that destabilises; adopt the last upright B strictly below it.
        # Max-over-upright could otherwise skip a non-monotone unstable band and
        # adopt a LARGER budget above a value that already falls.
        osc_samples = []
        max_safe_B = 0.0
        onset_B = None
        for B in sorted(cfg["combined_l2_grid"]):
            # distribute B equally in normalised space across the driven chans
            per = B / np.sqrt(len(chans))
            namp = {c: per for c in chans}
            namp["_L"] = L
            worst, fell, bad = _osc_worstcase(
                h, mode, condition, chans, namp, freqs, phase_set,
                leg_base, cfg["combined_osc_dur_s"]
            )
            upright = (not fell) and (worst <= STABLE_TILT_DEG)
            osc_samples.append({"l2_budget": float(B), "worst_tilt_deg": float(worst),
                                "fell": bool(fell), "upright": bool(upright),
                                "worst_case": bad})
            if upright:
                max_safe_B = float(B)
            else:
                onset_B = float(B)
                break
        out["oscillatory"][dname] = {
            "channels": chans,
            "max_safe_l2_budget": max_safe_B,
            "failure_onset_l2_budget": onset_B,
            "samples": osc_samples,
        }
    # The governing combined budget = the tightest across directions.
    budgets = [out["oscillatory"][d]["max_safe_l2_budget"] for d in out["oscillatory"]]
    out["governing_l2_budget"] = float(min(budgets)) if budgets else None
    return out


# ---------------------------------------------------------------------------
# Experiment 4 - sustained / oscillatory
# ---------------------------------------------------------------------------
def experiment_sine(h, mode, condition, static_result, leg_base, cfg):
    result = {}
    dur_ticks = int(round(cfg["sine_duration_s"] / CTRL_DT))
    for ch in HEAD_CHANNELS:
        dirs = static_result[ch]["directions"]
        cand = [dirs["pos"]["safe_limit"], dirs["neg"]["safe_limit"]]
        cand = [abs(c) for c in cand if abs(c) > 1e-6]
        amp = min(cand) if cand else 0.0  # peak amplitude within safe envelope
        j = HEAD_CHANNELS.index(ch)
        result[ch] = {"amplitude": amp, "freqs": {}}
        for f in cfg["sine_freqs"]:

            def fn(t, f=f, amp=amp, j=j):
                head = np.zeros(4)
                head[j] = amp * np.sin(2 * np.pi * f * (t * CTRL_DT))
                return head

            m = simulate(h, mode, condition, fn, dur_ticks, leg_base)
            result[ch]["freqs"][f"{f}Hz"] = {
                "tilt_max_deg": m["tilt_max_deg"],
                "z_min": m["z_min"],
                "leg_peak_excursion_rad": m["leg_peak_excursion_rad"],
                "fell": m["fell"],
                "upright": is_upright(m),
            }

    # Combined oscillation: neck_pitch + head_yaw out of phase, each at its safe
    # amplitude, scaled by the neck+yaw combined L2 budget if it binds.
    combo = {}
    for f in cfg["sine_freqs"]:
        an = abs(min([abs(static_result["neck_pitch"]["directions"][d]["safe_limit"])
                      for d in ("pos", "neg")
                      if abs(static_result["neck_pitch"]["directions"][d]["safe_limit"]) > 1e-6]
                     or [0.0]))
        ay = abs(min([abs(static_result["head_yaw"]["directions"][d]["safe_limit"])
                      for d in ("pos", "neg")
                      if abs(static_result["head_yaw"]["directions"][d]["safe_limit"]) > 1e-6]
                     or [0.0]))

        def fn(t, f=f, an=an, ay=ay):
            head = np.zeros(4)
            head[0] = an * np.sin(2 * np.pi * f * (t * CTRL_DT))
            head[2] = ay * np.sin(2 * np.pi * f * (t * CTRL_DT) + np.pi / 2)
            return head

        m = simulate(h, mode, condition, fn, dur_ticks, leg_base)
        combo[f"{f}Hz"] = {
            "tilt_max_deg": m["tilt_max_deg"],
            "z_min": m["z_min"],
            "fell": m["fell"],
            "upright": is_upright(m),
        }
    result["_combined_neck_yaw_quadrature"] = combo
    return result


# ---------------------------------------------------------------------------
# Experiment 5 - end-to-end validation of the enforced open_duck_anim envelope
# ---------------------------------------------------------------------------
def experiment_validate(h, mode, condition, leg_base, cfg):
    """Drive DELIBERATELY-UNSAFE raw head commands through the shipping
    ``open_duck_anim.envelope.clamp_head_envelope`` and confirm the *clamped*
    command keeps the robot upright with margin.

    This closes the loop: it proves the enforced module (not just the measured
    numbers) is safe. Raw inputs deliberately exceed the envelope — full-range
    steps and adversarial multi-axis oscillations (the cases shown to topple the
    raw additive path) — and are passed tick-by-tick through the clamp with the
    previous enforced command as ``prev`` (so the slew guard is active).
    """
    import itertools

    dur_ticks = int(round(cfg["combined_osc_dur_s"] / CTRL_DT))
    freqs = cfg["combined_freqs"]
    phase_set = cfg["combined_phases"]
    envs = {"default": odenv.DEFAULT_ENVELOPE, "hw_derated": odenv.DEFAULT_ENVELOPE.derated()}

    def run_clamped(raw_of_tick, n_ticks, env):
        """Run with each raw command passed through env.clamp (stateful prev)."""
        h.reset()
        base = np.zeros(7)
        base[0:3] = s01.base_locomotion(condition)
        prev = np.zeros(4)
        tilt_max = 0.0
        z_min = 9.0
        fell = False
        for t in range(n_ticks):
            clamped = env.clamp(raw_of_tick(t), prev_command_head=prev, dt=CTRL_DT)
            prev = clamped
            command = base.copy()
            command[3:7] = clamped
            h.control_tick(command, mode)
            pos, _q, tilt = h.base_state()
            tilt_max = max(tilt_max, tilt)
            z_min = min(z_min, pos[2])
            if pos[2] < FALL_HEIGHT:
                fell = True
        return {"tilt_max_deg": float(tilt_max), "z_min": float(z_min), "fell": bool(fell),
                "upright": bool((not fell) and tilt_max <= STABLE_TILT_DEG and z_min >= STABLE_Z_MIN)}

    out = {}
    for ename, env in envs.items():
        cases = {}

        # (a) full-range steps on every channel, both signs (raw = 1.5x range edge).
        step_worst = {"tilt_max_deg": 0.0, "z_min": 9.0, "fell": False, "upright": True}
        for ch in HEAD_CHANNELS:
            j = HEAD_CHANNELS.index(ch)
            lo, hi = CMD_RANGE[ch]
            for edge in (1.5 * lo, 1.5 * hi):
                if edge == 0.0:
                    continue

                def raw(t, j=j, edge=edge):
                    v = np.zeros(4)
                    v[j] = edge
                    return v

                r = run_clamped(raw, int(round(cfg["hold_s"] / CTRL_DT)), env)
                step_worst["tilt_max_deg"] = max(step_worst["tilt_max_deg"], r["tilt_max_deg"])
                step_worst["z_min"] = min(step_worst["z_min"], r["z_min"])
                step_worst["fell"] = step_worst["fell"] or r["fell"]
                step_worst["upright"] = step_worst["upright"] and r["upright"]
        cases["fullrange_steps"] = step_worst

        # (b) adversarial multi-axis oscillation (raw amplitude = 1.5x range edge,
        #     every phase combo, resonant freqs) — the toppling case, now clamped.
        osc_worst = {"tilt_max_deg": 0.0, "z_min": 9.0, "fell": False, "upright": True,
                     "worst_case": None}
        raw_amp = np.array([1.5 * max(abs(CMD_RANGE[c][0]), abs(CMD_RANGE[c][1]))
                            for c in HEAD_CHANNELS])
        for f in freqs:
            for phases in itertools.product(phase_set, repeat=4):
                ph = np.array(phases)

                def raw(t, f=f, ph=ph):
                    w = 2 * np.pi * f * (t * CTRL_DT)
                    return raw_amp * np.sin(w + ph)

                r = run_clamped(raw, dur_ticks, env)
                if r["tilt_max_deg"] > osc_worst["tilt_max_deg"]:
                    osc_worst["tilt_max_deg"] = r["tilt_max_deg"]
                    osc_worst["worst_case"] = {"freq": float(f),
                                               "phases": [float(p) for p in phases]}
                osc_worst["z_min"] = min(osc_worst["z_min"], r["z_min"])
                osc_worst["fell"] = osc_worst["fell"] or r["fell"]
                osc_worst["upright"] = osc_worst["upright"] and r["upright"]
                if r["fell"]:
                    break
            if osc_worst["fell"]:
                break
        cases["adversarial_oscillation"] = osc_worst
        cases["all_upright"] = bool(step_worst["upright"] and osc_worst["upright"])
        out[ename] = cases
    return out


# ---------------------------------------------------------------------------
def run_condition(h, mode, condition, cfg, plotdir, experiments):
    leg_base = _leg_baseline(h, mode, condition)
    # Re-verify zero-command baseline in every configuration (harness-drift guard)
    fn0, n0 = ramp_hold_cmd("neck_pitch", 0.0,
                            int(round(cfg["ramp_s"] / CTRL_DT)),
                            int(round(cfg["hold_s"] / CTRL_DT)))
    base_m = simulate(h, mode, condition, fn0, n0, leg_base)
    out = {
        "mode": mode,
        "condition": condition,
        "baseline_zero_command": {
            "tilt_max_deg": base_m["tilt_max_deg"],
            "z_min": base_m["z_min"],
            "fell": base_m["fell"],
            "upright": is_upright(base_m),
        },
    }
    static_result = None
    if "static" in experiments:
        static_result = experiment_static(h, mode, condition, leg_base, cfg, plotdir)
        out["static"] = static_result
    # validate is self-contained (drives commands through open_duck_anim.envelope);
    # it does not depend on the static sweep, so run it regardless.
    if "validate" in experiments:
        out["validate"] = experiment_validate(h, mode, condition, leg_base, cfg)
    if static_result is None:
        return out
    if "slew" in experiments:
        out["slew"] = experiment_slew(h, mode, condition, static_result, leg_base, cfg, plotdir)
    if "combined" in experiments:
        out["combined"] = experiment_combined(h, mode, condition, static_result, leg_base, cfg)
    if "sine" in experiments:
        out["sine"] = experiment_sine(h, mode, condition, static_result, leg_base, cfg)
    return out


def _summ_static(out):
    """Compact per-channel table across conditions."""
    table = {}
    for ch in HEAD_CHANNELS:
        table[ch] = {}
    for res in out["results"]:
        cond = res["condition"]
        if "static" not in res:
            continue
        for ch in HEAD_CHANNELS:
            d = res["static"][ch]["directions"]
            table[ch][cond] = {
                "onset_pos": d["pos"]["failure_onset"],
                "onset_neg": d["neg"]["failure_onset"],
                "safe_pos": d["pos"]["safe_limit"],
                "safe_neg": d["neg"]["safe_limit"],
            }
    # Cross-condition combined safe limit = most conservative (closest to 0).
    combined = {}
    for ch in HEAD_CHANNELS:
        pos = [table[ch][c]["safe_pos"] for c in table[ch]]
        neg = [table[ch][c]["safe_neg"] for c in table[ch]]
        combined[ch] = {
            "safe_pos": float(min(pos)) if pos else None,
            "safe_neg": float(max(neg)) if neg else None,
        }
    return {"per_condition": table, "combined_across_conditions": combined}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--onnx", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "BEST_WALK_ONNX_2.onnx"))
    ap.add_argument("--mjcf", default=s01.DEFAULT_MJCF)
    ap.add_argument("--mode", choices=["additive", "policy_only"], default="additive",
                    help="additive is what ships today (the mode under test)")
    ap.add_argument("--condition", choices=["stand", "walk", "both"], default="both")
    ap.add_argument("--experiment", default="all",
                    help="comma list of {static,slew,combined,sine} or 'all'")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--ramp-s", type=float, default=0.5,
                    help="ramp time to plateau for static/combined sweeps")
    ap.add_argument("--hold-s", type=float, default=5.0,
                    help="hold time at plateau (>= slow-divergence time-to-fall)")
    ap.add_argument("--coarse-step", type=float, default=0.1,
                    help="DEPRECATED / unused: sweep_direction is now a single "
                         "genuinely-fine outward sweep (reviewer E10). Kept only "
                         "for CLI back-compat.")
    ap.add_argument("--fine-step", type=float, default=0.01,
                    help="resolution of the static first-onset sweep (rad). This "
                         "is the honest resolution limit: no unstable band wider "
                         "than this can be skipped (reviewer E8/E10).")
    ap.add_argument("--slew-hold-s", type=float, default=5.0)
    ap.add_argument("--slew-rates", type=float, nargs="+",
                    default=[0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0])
    ap.add_argument("--combined-scales", type=float, nargs="+",
                    default=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    ap.add_argument("--combined-l2-grid", type=float, nargs="+",
                    default=[0.4, 0.5, 0.6, 0.7, 0.8, 1.0])
    ap.add_argument("--combined-freqs", type=float, nargs="+",
                    default=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
    ap.add_argument("--combined-phases", type=float, nargs="+",
                    default=[0.0, 1.5708, 3.1416, -1.5708])
    ap.add_argument("--combined-osc-dur-s", type=float, default=6.0)
    ap.add_argument("--sine-freqs", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--sine-duration-s", type=float, default=8.0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    experiments = (
        ["static", "slew", "combined", "sine", "validate"]
        if args.experiment == "all"
        else [e.strip() for e in args.experiment.split(",")]
    )
    if any(e in experiments for e in ("slew", "combined", "sine")) and "static" not in experiments:
        experiments = ["static"] + experiments  # downstream depends on static

    cfg = {
        "ramp_s": args.ramp_s,
        "hold_s": args.hold_s,
        "coarse_step": args.coarse_step,
        "fine_step": args.fine_step,
        "slew_hold_s": args.slew_hold_s,
        "slew_rates": args.slew_rates,
        "combined_scales": args.combined_scales,
        "combined_l2_grid": args.combined_l2_grid,
        "combined_freqs": args.combined_freqs,
        "combined_phases": args.combined_phases,
        "combined_osc_dur_s": args.combined_osc_dur_s,
        "sine_freqs": args.sine_freqs,
        "sine_duration_s": args.sine_duration_s,
        "stable_tilt_deg": STABLE_TILT_DEG,
        "stable_z_min": STABLE_Z_MIN,
        "safety_fraction": SAFETY_FRACTION,
    }

    conditions = ["stand", "walk"] if args.condition == "both" else [args.condition]
    h = s01.Harness(args.mjcf, args.onnx)
    out = {"task": "D13/R16 envelope", "mode": args.mode, "config": cfg, "results": []}
    for condition in conditions:
        sys.stderr.write(f"[envelope] {args.mode} {condition}: {experiments}\n")
        sys.stderr.flush()
        res = run_condition(h, args.mode, condition, cfg, args.outdir, experiments)
        out["results"].append(res)
        with open(os.path.join(args.outdir, f"envelope_{args.mode}_{condition}.json"), "w") as f:
            json.dump(res, f, indent=2)

    if "static" in experiments:
        out["summary_static"] = _summ_static(out)
    with open(os.path.join(args.outdir, "envelope_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    if "static" in experiments:
        print(json.dumps(out["summary_static"], indent=2))


if __name__ == "__main__":
    main()
