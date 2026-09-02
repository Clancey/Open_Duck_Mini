"""Empirical safe head operating envelope for the *additive* interim (D13/R16).

The head mode that ships today (``v2_rl_walk_mujoco.py:310-311``, "additive")
adds the animation head offset on top of the policy's motor targets. Spike S0.1
found — and ``experiments/animation/envelope_sweep.py`` quantified — that this
path **topples the robot** at large head deflections and at certain multi-axis
dynamic combinations (defect **D13**, risk **R16**; plan §3.4, §6.5, §7 Phase
3/4). Until the Phase 5 leg/neck reward split retrains command-following, any
authored head motion driven through the additive path must be constrained to a
validated safe envelope so it "cannot reach the toppling extremes"
(plan §7 Phase 4).

This module owns that envelope and enforces it. It composes with, and does not
duplicate, the other limiters:

* :mod:`open_duck_anim.transform` turns an absolute authored head pose into a
  relative command offset and clamps it to the RL **training ranges**.
* This module then clamps that offset to the tighter **empirical safe envelope**
  (per-channel deflection + command slew + a combined multi-axis budget) before
  it is injected additively.
* :mod:`open_duck_anim.limits` still applies the final 14-DOF ``jnt_range`` clamp
  and the ``5.24 rad/s`` bus-target rate limit *after* mode selection. The slew
  guard here is on the animation *command*, which is a distinct concern (limiting
  the command does not by itself constrain the policy's own output — plan §6.4).

WHY THESE NUMBERS ARE NOT PHYSICAL CONSTANTS — AND ARE POLICY-SPECIFIC.
Every value below is an **empirical** property of ONE specific policy checkpoint,
measured in additive mode. They are valid **only** for a policy trained with the
head passthrough active during training (``motor_targets[5:9] = command[3:7] +
motor_targets[5:9]``, mirroring the deployed runtime at
``v2_rl_walk_mujoco.py:310-311``), which is what let the legs learn to balance
while the head is driven through its full command range. The values here were
re-derived on 2026-09-02 against ``passthrough_final_300M.onnx`` (Phase 5
iteration 3; 300M steps; branch ``feature/head-passthrough-training`` @
``08536e1``; eval reward 254.1). See the per-constant provenance below.

**DO NOT apply these numbers to a differently-trained policy.** They are much
wider than the envelope measured against the old ``BEST_WALK_ONNX_2.onnx`` policy
(which had never had its head driven during training and toppled at small head
deflections). If the deployed checkpoint changes — a retrain, a different reward,
a different head-drive wiring, or a revert to a non-passthrough policy — these
limits are INVALID and MUST be re-derived with ``envelope_sweep.py`` against the
new checkpoint before any authored head motion is driven on hardware. The
checkpoint each constant was measured against is named in its provenance; if the
shipping policy is not that checkpoint (or a re-measured successor), treat the
envelope as unverified.

STABILITY CRITERION used to derive the limits ("stably upright with margin"):
peak base tilt ≤ 8.6° (0.15 rad, the plan's Phase 4 hardware-acceptance bound),
min base height ≥ 0.12 m, and no fall — held for ≥5 s so slow divergence
(topples took up to ~3.4 s to develop) is caught. The per-channel *failure
onset* is the first deflection (swept finely outward from nominal, because the
instability is non-monotonic — an unstable band with a stable island beyond it)
at which that criterion is violated. The *safe limit* is a conservative
:data:`SAFETY_FRACTION` of that onset.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

# Head channel order within the length-4 head command / commands[3:7]
# (plan §3.3 / §6.3). Identical to :data:`open_duck_anim.transform.HEAD_CHANNELS`.
HEAD_CHANNELS: Tuple[str, str, str, str] = (
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
)

# Control tick used to derive the per-tick slew step (plan §6.6: ctrl_dt = 0.02).
CTRL_DT: float = 0.02

# Conservative fraction of the measured failure onset adopted as the safe limit.
# 0.5 == a 2x margin below the first sign of instability. Chosen because the
# divergence past the onset is steep (near-instant topple a few hundredths of a
# radian beyond it) and because sim is not reality (see DERATING note below).
SAFETY_FRACTION: float = 0.5

# ---------------------------------------------------------------------------
# Per-channel safe deflection limits (rad), as (low, high) offsets from nominal.
# ---------------------------------------------------------------------------
# EMPIRICAL — POLICY-SPECIFIC. Measured 2026-09-02 by
# experiments/animation/envelope_sweep.py against passthrough_final_300M.onnx
# (Phase 5 iteration 3: head passthrough active during training; 300M steps;
# branch feature/head-passthrough-training @ 08536e1; eval reward 254.1),
# additive mode, conditions stand AND walk, single genuinely-fine outward sweep
# at 0.01 rad resolution, 0.5 s ramp + 5 s hold, first-onset rule. Value =
# SAFETY_FRACTION * failure_onset (or the full training-range edge where no
# topple occurs on that side), taken as the most conservative (closest-to-zero)
# across both conditions. Gates defect D13 / risk R16 (plan §3.4, §6.5, §7).
#
# VALID ONLY for a policy trained with the head passthrough active (see module
# docstring). These are MUCH wider than the old BEST_WALK_ONNX_2.onnx envelope
# — the retrained legs balance while the head is driven — and MUST be re-derived
# if the deployed checkpoint changes.
#
#   channel      failure onset (stand / walk)          safe limit
#   neck_pitch   none / +0.71 ;   none / none          (-0.34, +0.355)
#   head_pitch   none / none  ;   none / none           (-0.78, +0.78) full range
#   head_yaw     none / none  ;   none / none           (-1.50, +1.50) full range
#   head_roll    none / none  ;   none / none           (-0.50, +0.50) full range
#
# "none" == no topple anywhere on that side within the trained command range, so
# the safe limit there is the full training-range edge (transform.py TRAINING_*).
# The only remaining binding constraint is neck_pitch positive under walk
# (onset +0.71 -> safe +0.355); every other channel is safe to its full trained
# range in both conditions. The cases that toppled the OLD policy now survive the
# full range: head_yaw -0.58 walking, neck_pitch +0.62 standing.
# Reproduce (2026-09-02, additive mode, passthrough_final_300M.onnx):
#   envelope_sweep.py --onnx passthrough_final_300M.onnx \
#     --experiment static --condition both --fine-step 0.01 --ramp-s 0.5 --hold-s 5.0
DEFLECTION_LIMITS: Dict[str, Tuple[float, float]] = {
    "neck_pitch": (-0.34, 0.355),
    "head_pitch": (-0.78, 0.78),
    "head_yaw": (-1.50, 1.50),
    "head_roll": (-0.50, 0.50),
}

DEFLECTION_LOW: np.ndarray = np.array(
    [DEFLECTION_LIMITS[c][0] for c in HEAD_CHANNELS], dtype=np.float64
)
DEFLECTION_HIGH: np.ndarray = np.array(
    [DEFLECTION_LIMITS[c][1] for c in HEAD_CHANNELS], dtype=np.float64
)
DEFLECTION_LOW.setflags(write=False)
DEFLECTION_HIGH.setflags(write=False)

# ---------------------------------------------------------------------------
# Command slew-rate limit (rad/s) on the animation head command.
# ---------------------------------------------------------------------------
# EMPIRICAL / adopted. envelope_sweep.py found that, WITHIN the deflection
# envelope, a single-axis command step is stable up to at least 30 rad/s in sim
# (a fast step to a within-envelope amplitude does not topple — the binding
# constraint is sustained deflection and multi-axis coupling, NOT transient
# slew). Slew limiting is therefore a secondary, defence-in-depth guard, not the
# primary balance protection. We adopt the platform motor limit 5.24 rad/s
# (joystick.py:49-60, plan §6.4 / Appendix C) so the head command can never
# demand a rate faster than the actuator bus can honour, with wide measured
# margin. POLICY-SPECIFIC: re-derive if the deployed checkpoint changes.
# (Measured 2026-09-02 against passthrough_final_300M.onnx.)
# Reproduce (2026-09-02, additive mode, passthrough_final_300M.onnx; confirms no
# single-axis step below the adopted 5.24 rad/s destabilises, so the platform
# limit binds first):
#   envelope_sweep.py --onnx passthrough_final_300M.onnx \
#     --experiment static,slew --condition both \
#     --slew-rates 0.5 1.0 2.0 4.0 8.0 16.0 30.0 --slew-hold-s 5.0
SLEW_LIMIT: float = 5.24  # rad/s

# ---------------------------------------------------------------------------
# Combined multi-axis L2 budget (dimensionless, in per-axis-normalised units).
# ---------------------------------------------------------------------------
# EMPIRICAL — POLICY-SPECIFIC. Driving several head channels *simultaneously* —
# the realistic animation case ("a head turn is rarely one axis") — can expose
# phase- and frequency-dependent resonances that the per-axis static limits miss.
# A command is scored as n_i = c_i / L_i, where L_i = min(|safe_low_i|,
# |safe_high_i|) is the magnitude of the TIGHTER (more dangerous) side of each
# channel. Enforcement uses this SAME normaliser (HeadEnvelope._L, envelope.py)
# as the measurement (_safe_mag_vec in envelope_sweep.py) — they must not
# diverge, or the scalar below is applied against looser denominators than it was
# calibrated with and becomes strictly more permissive than anything measured
# (reviewer E2).
#
# Re-derived 2026-09-02 against passthrough_final_300M.onnx (head passthrough
# active during training; 300M steps; feature/head-passthrough-training @
# 08536e1). With the retrained legs the coupled multi-axis case is far more
# tolerant than the old policy, but two methods DISAGREE and the enforced path
# governs:
#   * The open-loop derivation (experiment_combined: smooth, equal-amplitude
#     sinusoids distributed evenly in normalised space) shows NO onset anywhere
#     on the grid {0.4,0.5,0.6,0.7,0.8,1.0} in EITHER condition; the all-four-axis
#     worst-case tilt is 7.80 deg (stand) / 7.94 deg (walk) at ||n||_2 = 1.0. Taken
#     alone this would suggest a budget of ~1.0. It is only a lower bound on the
#     true onset (the grid was not extended past 1.0) and, crucially, it drives
#     SMOOTH sinusoids that the per-channel clamp never has to saturate.
#   * The closed-loop adversarial validation (experiment_validate: raw commands at
#     1.5x the training range driven tick-by-tick through THIS module's clamp,
#     every phase combo over {0, pi/2, pi, -pi/2}, freqs 0.5-2.0 Hz, 6 s each) is
#     HARSHER, because the per-channel clamp turns a saturating command into a
#     clipped, asymmetric waveform richer in destabilising harmonics than the
#     smooth derivation sinusoid. Governing (max of stand/walk) enforced worst-case
#     tilt by budget: 0.50 -> 6.54 deg, 0.60 -> 7.18 deg, 0.70 -> 7.85 deg,
#     0.80 -> 8.82 deg (OVER the 8.6 deg Phase 4 bound), 1.00 -> 10.87 deg.
# Because enforcement is what ships, the enforced-path number governs. We adopt
# 0.70: the largest budget whose adversarial enforced worst-case stays under the
# 8.6 deg bound with margin (7.85 deg, ~0.75 deg margin; 0.80 is rejected). This
# is strictly tighter than the open-loop derivation would allow (reviewer E2's
# lesson: trust the enforced path, not the smooth grid). It independently matches
# the budget the peak_279M checkpoint yielded, and is still 1.27x the old policy's
# 0.55 — the constraint on coupled multi-axis head motion relaxes ~22%.
# Reproduce (2026-09-02, additive mode, passthrough_final_300M.onnx) —
# derivation grid and the governing adversarial check:
#   envelope_sweep.py --onnx passthrough_final_300M.onnx \
#     --experiment static,combined,validate --condition both \
#     --combined-l2-grid 0.4 0.5 0.6 0.7 0.8 1.0 \
#     --combined-phases 0 1.5708 3.1416 -1.5708 \
#     --combined-freqs 0.5 1.0 1.5 2.0 --combined-osc-dur-s 6.0 --hold-s 5.0
# The adopted 0.70 is the largest budget whose experiment_validate worst-case
# tilt stays <= 8.6 deg in BOTH conditions (files/envelope/revalidation_iter3/).
COMBINED_L2_BUDGET: float = 0.70

# Recommended ADDITIONAL derating for first hardware trials (multiplies the
# deflection limits and the combined budget). Sim is not reality: contact,
# mass distribution, friction and servo dynamics are unmodelled, and the additive
# path bypasses the bus rate limit. Note head kp on hardware is 8 (soft) vs legs
# at 30, so the head servo tracks with lag and undershoots the command — real
# head motion transmits *less* destabilising torque than sim at the same command
# (cuts toward safety), but the lag also shifts phase and could excite different
# modes. KEPT AT 0.5 despite the wider iteration-3 envelope: the envelope is now
# far more trustworthy (the policy was trained with the head passthrough active,
# so the legs actually learned to balance under head motion, and the cases that
# toppled the old policy now survive the full range), but this is still the FIRST
# time this head range is driven on physical hardware and sim is still not
# reality — first-on-hardware caution is unchanged in kind even though the
# sim-side envelope grew. Widening the envelope AND relaxing the derating at the
# same time would compound two sources of optimism; hold the derating and relax
# only as on-hardware data accrues (plan §7 Phase 4 acceptance: zero falls over
# 10 one-minute trials at max authored deflection/slew). This is advisory;
# :func:`clamp_head_envelope` does NOT apply it automatically (pass a derated
# :class:`HeadEnvelope`).
HARDWARE_DERATING: float = 0.5

ArrayLike = Union[np.ndarray, Sequence[float]]


def _as_head_vec(x: ArrayLike, name: str) -> np.ndarray:
    """Coerce to a 1-D length-4 float64 head vector.

    Rejects non-numeric input (``TypeError``) and any shape other than exactly
    ``(4,)`` (``ValueError``). We deliberately refuse batched / >1-D input: the
    combined-budget norm sums over the whole array, so a silently-accepted batch
    would compute a meaningless global norm (reviewer E12).
    """
    arr = np.asarray(x)
    if arr.dtype == object or not np.issubdtype(arr.dtype, np.number):
        raise TypeError("%s must be a numeric length-4 head-command array" % name)
    arr = arr.astype(np.float64, copy=False)
    if arr.shape != (4,):
        raise ValueError(
            "%s must be a 1-D length-4 head vector (got shape %r)" % (name, arr.shape)
        )
    return arr


def _sanitize_head(c: np.ndarray, prev: Optional[np.ndarray]) -> Tuple[np.ndarray, bool]:
    """Replace non-finite channels with a safe fallback; report a fault.

    Last line of defence (reviewer E1): NaN/Inf must NEVER reach the clamp math,
    because ``np.clip`` propagates NaN and the combined-budget scale then computes
    NaN, silently disabling the L2 guard. Rather than raise in the hot control
    path (which would drop a control tick), we substitute each non-finite channel
    with the previous *finite* enforced value (best: hold last-good) or ``0.0``,
    and return ``fault=True`` so the caller can latch a fault state.

    ``c`` is mutated in place (it is already a private copy inside :meth:`clamp`).
    """
    bad = ~np.isfinite(c)
    if not bad.any():
        return c, False
    if prev is not None:
        fallback = np.where(np.isfinite(prev), prev, 0.0)
    else:
        fallback = np.zeros(4, dtype=np.float64)
    c[bad] = fallback[bad]
    return c, True


@dataclass
class HeadEnvelope:
    """The enforced safe head envelope: deflection clamp + combined budget + slew.

    Attributes:
        low, high: length-4 per-channel deflection limits (rad), in
            :data:`HEAD_CHANNELS` order.
        slew_limit: max command slew (rad/s) applied per channel given ``dt``.
        l2_budget: max ``||c / L||_2`` where ``L_i = min(|low_i|, high_i)`` — the
            magnitude of the *tighter (more dangerous)* side of each channel. This
            normaliser is identical to the harness's ``_safe_mag_vec`` that the
            budget scalar was measured against (reviewer E2); using the looser
            commanded-side limit here would be strictly more permissive than
            anything measured.

    Guards are applied in order — per-channel clamp, then the combined L2 budget,
    then the slew guard LAST (reviewer E4) so that the per-channel rate cap
    ``|c - prev| <= slew_limit*dt`` genuinely holds on the returned value. Because
    the weighted-L2 unit ball is convex and both ``prev`` (a previous enforced
    output) and the scaled command lie inside it, slewing between them keeps the
    result inside the deflection box AND the L2 budget, so all three invariants
    hold simultaneously.

    A non-finite input channel is replaced with a safe fallback and flagged
    (``clamp(..., return_fault=True)``); it never propagates into the math.
    All operations are stateless (the caller owns ``prev``) and allocation-light.
    """

    low: np.ndarray = field(default_factory=lambda: DEFLECTION_LOW.copy())
    high: np.ndarray = field(default_factory=lambda: DEFLECTION_HIGH.copy())
    slew_limit: float = SLEW_LIMIT
    l2_budget: float = COMBINED_L2_BUDGET
    # Explicit, greppable full-disable sentinel (see :meth:`unbounded`). When set,
    # :meth:`clamp` performs NO processing at all (not even finiteness repair) and
    # returns the input verbatim — the caller has deliberately opted out of safety.
    bypass: bool = False

    def __post_init__(self) -> None:
        # Copy so a caller mutating the array they passed cannot mutate the
        # envelope after construction (reviewer E13), then freeze.
        self.low = np.array(self.low, dtype=np.float64)
        self.high = np.array(self.high, dtype=np.float64)
        if self.low.shape != (4,) or self.high.shape != (4,):
            raise ValueError("low/high must be length-4 head vectors")
        if np.any(self.high < self.low):
            raise ValueError("every high must be >= its low")
        if np.any(self.low > 0) or np.any(self.high < 0):
            raise ValueError("deflection limits must bracket 0 (offsets from nominal)")
        if not (self.slew_limit > 0):
            raise ValueError("slew_limit must be > 0")
        if not (self.l2_budget > 0):
            raise ValueError("l2_budget must be > 0")
        self.low.setflags(write=False)
        self.high.setflags(write=False)
        # Per-axis normaliser L_i = min(|low_i|, high_i): the tighter/dangerous
        # side, matching the harness _safe_mag_vec the budget was measured with.
        L = np.minimum(np.abs(self.low), self.high)
        self._L = np.where(L > 1e-12, L, 1e-12)
        self._L.setflags(write=False)

    @classmethod
    def unbounded(cls) -> "HeadEnvelope":
        """Explicit, greppable escape hatch that disables ALL enforcement.

        Returns an envelope whose :meth:`clamp` is a pure pass-through (no
        deflection clamp, no combined budget, no slew guard, no finiteness
        repair). Use this — never ``None`` — when a caller must deliberately opt
        out of the D13/R16 safety envelope, so the decision is auditable
        (``grep unbounded``). Safety code is opt-OUT (reviewer E3): the default
        :data:`DEFAULT_ENVELOPE` enforces; disabling requires this sentinel.
        """
        inf = np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float64)
        return cls(low=-inf, high=inf, slew_limit=np.inf, l2_budget=np.inf, bypass=True)

    def derated(self, factor: float = HARDWARE_DERATING) -> "HeadEnvelope":
        """Return a copy with the deflection limits scaled by ``factor``.

        Use ``env.derated()`` for the first hardware trials (see
        :data:`HARDWARE_DERATING`). Only ``low``/``high`` are scaled. The
        ``l2_budget`` is intentionally LEFT UNCHANGED (reviewer E5): it is
        expressed in units normalised by ``L = min(|low|, high)``, so scaling the
        deflection limits already derates the absolute combined constraint by
        ``factor``. Scaling the budget too would apply the factor twice
        (``factor**2``) and make the derated per-channel limits unreachable dead
        code. The slew limit is a rate cap already inside the safe region and is
        left unchanged.
        """
        if not 0 < factor <= 1:
            raise ValueError("factor must be in (0, 1]")
        return HeadEnvelope(
            low=self.low * factor,
            high=self.high * factor,
            slew_limit=self.slew_limit,
            l2_budget=self.l2_budget,
        )

    def _combined_scale(self, c: np.ndarray) -> float:
        """Scalar factor (<=1) enforcing ``||c / L||_2 <= l2_budget``.

        ``L_i = min(|low_i|, high_i)`` (the tighter/dangerous side), matching the
        harness normaliser the budget was calibrated against (reviewer E2).
        """
        norm = float(np.sqrt(np.sum((c / self._L) ** 2)))
        if norm <= self.l2_budget or norm == 0.0:
            return 1.0
        return self.l2_budget / norm

    def clamp(
        self,
        command_head: ArrayLike,
        prev_command_head: Optional[ArrayLike] = None,
        dt: float = CTRL_DT,
        out: Optional[np.ndarray] = None,
        return_fault: bool = False,
    ):
        """Enforce the envelope on a length-4 head command offset.

        Args:
            command_head: raw head command offset ``[neck_pitch, head_pitch,
                head_yaw, head_roll]`` (rad, relative to nominal).
            prev_command_head: the previous *enforced* head command, used for the
                slew guard. ``None`` (the first tick, or a fresh trajectory) skips
                slew limiting. Pass this every tick to make the slew guard active.
            dt: tick duration (s) for the slew step; defaults to :data:`CTRL_DT`.
            out: optional length-4 output buffer (hot-path friendly).
            return_fault: when True, return ``(clamped, fault_bool)`` where
                ``fault`` is True iff a non-finite input channel had to be
                repaired (reviewer E1). Default False returns just the array.

        Returns:
            The clamped length-4 head command: within the per-channel deflection
            limits, scaled so ``||c / L||_2 <= l2_budget``, and moved from
            ``prev`` by at most ``slew_limit*dt`` per channel (the slew guard is
            applied last, so this rate postcondition holds exactly).
        """
        c = _as_head_vec(command_head, "command_head").copy()

        if self.bypass:  # deliberate full opt-out (unbounded): verbatim.
            return (self._finish(c, out), False) if return_fault else self._finish(c, out)

        p = None
        if prev_command_head is not None:
            p = _as_head_vec(prev_command_head, "prev_command_head").copy()
            # A poisoned prev must not propagate (reviewer E1): treat any
            # non-finite prev channel as 0 for the slew reference.
            p_bad = ~np.isfinite(p)
            if p_bad.any():
                p[p_bad] = 0.0

        # (0) finiteness: last line of defence, never let NaN/Inf reach the math.
        c, fault = _sanitize_head(c, p)

        # (1) per-channel deflection clamp.
        np.clip(c, self.low, self.high, out=c)

        # (2) combined multi-axis L2 budget (uniform down-scale if exceeded).
        scale = self._combined_scale(c)
        if scale < 1.0:
            c *= scale

        # (3) slew guard LAST so |c - prev| <= slew_limit*dt actually holds on the
        # returned value (reviewer E4). Convexity of the weighted-L2 ball means
        # this preserves both the deflection box and the L2 budget.
        if p is not None:
            if not (dt > 0):
                raise ValueError("dt must be > 0")
            # Bound the slew reference to the deflection box so an out-of-envelope
            # prev cannot enlarge the allowed excursion.
            np.clip(p, self.low, self.high, out=p)
            max_step = self.slew_limit * dt
            c = p + np.clip(c - p, -max_step, max_step)

        result = self._finish(c, out)
        return (result, fault) if return_fault else result

    @staticmethod
    def _finish(c: np.ndarray, out: Optional[np.ndarray]) -> np.ndarray:
        if out is not None:
            out[...] = c
            return out
        return c


# Module-level default envelope built from the measured constants.
DEFAULT_ENVELOPE = HeadEnvelope()


def clamp_head_envelope(
    command_head: ArrayLike,
    prev_command_head: Optional[ArrayLike] = None,
    dt: float = CTRL_DT,
    envelope: HeadEnvelope = DEFAULT_ENVELOPE,
    out: Optional[np.ndarray] = None,
    return_fault: bool = False,
):
    """Enforce the safe head envelope (module-level convenience, plan §6.5/D13).

    Thin wrapper over :meth:`HeadEnvelope.clamp` using :data:`DEFAULT_ENVELOPE`
    (or a supplied — e.g. hardware-derated — envelope). Wire the animation
    engine's head output through this before the additive injection so no
    authored input can reach the D13 toppling extremes. See :meth:`clamp` for
    ``return_fault``.
    """
    return envelope.clamp(
        command_head, prev_command_head, dt=dt, out=out, return_fault=return_fault
    )
