"""Conservative leg operating envelope for **dock full-body** clips (plan §6.2).

Head-masked clips hold the legs; the RL policy owns them while standing or
walking. The *only* mode in which animation is allowed to move the legs is
``DOCK_DEMO``: the robot is docked/cradled, the legs are not load-bearing, no
policy is running, and there is therefore no balance constraint (plan §4.3, §6.2
capability matrix). This module owns the safety envelope for that — and only that
— case.

WHY THIS IS NOT THE HEAD ENVELOPE. The head :mod:`~open_duck_anim.envelope` is an
*empirical balance* envelope: it bounds how far the additive head command may
push a *balancing* biped before it topples, measured per policy. There is no
balance criterion here — the dock carries the weight — so there is nothing to
measure against a falling robot. What still binds the legs on the dock is purely
**mechanical**:

* **Joint limits.** Every leg target must stay inside the MJCF ``jnt_range``
  (``open_duck_mini_v2.xml``) — the same numbers the runtime's final
  :class:`~open_duck_anim.limits.JointLimiter` enforces on the bus targets.
* **Self-collision and cable strain.** The legs can reach poses the head never
  could: a large knee/ankle excursion can drive the foot or shin into the body,
  and a big hip sweep can wrap the servo cabling. The mechanical ``jnt_range`` on
  its own does **not** forbid a self-collision (MJCF ranges are per-joint, not
  pairwise), so we bound each leg channel to a **small deflection around the dock
  hold pose** instead of letting it roam the full range. Modest hip yaw/roll
  (twist/rock) is preferred; knee/ankle are kept nearly still. This keeps the
  wiggle in the region the collision check (``experiments/animation/
  phase4_dock_fullbody_sim.py``) confirms is contact-free, so the duck cannot
  knee itself in the face.
* **Velocity.** The final bus targets are still rate-limited at
  ``max_motor_velocity = 5.24 rad/s`` (:class:`~open_duck_anim.limits.
  JointRateLimiter`); the engine additionally pre-limits the animated leg targets
  it emits so a bad clip cannot demand an out-of-spec step even before the
  downstream limiter sees it.

DERATING (first hardware use). As with the head envelope (plan §6.5, R16), the
per-channel deflections are scaled by :data:`DOCK_LEG_DERATING` (0.5) for the
first time the legs are driven on physical hardware. Sim is not reality —
contact, friction and servo dynamics are unmodelled — so first-on-hardware
caution applies to the legs exactly as it does to the head, and the shipped
``dock_wiggle`` clip is authored to stay inside the *derated* deflections so the
runtime clamp is a no-op (nothing ships clamped). Relax toward 1.0 only as
on-hardware data accrues.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np

from .joint_order import INIT_POS_14, HW_ORDER_14

# Leg DOF indices within the 14-DOF hardware bus order (everything but head 5..8).
_HEAD_NAMES = {"neck_pitch", "head_pitch", "head_yaw", "head_roll"}
LEG_HW_INDICES = tuple(i for i, n in enumerate(HW_ORDER_14) if n not in _HEAD_NAMES)
LEG_NAMES = tuple(HW_ORDER_14[i] for i in LEG_HW_INDICES)
N_LEG = len(LEG_HW_INDICES)  # 10

# Dock hold pose: the load-relieving posture the legs rest in on the dock. This
# is ``init_pos`` for the ten leg DOF (plan §6.2 "held (load-relieving dock
# hold)"); animated leg motion is expressed as a small deflection *around* it.
DOCK_LEG_HOLD: np.ndarray = INIT_POS_14[list(LEG_HW_INDICES)].astype(np.float64).copy()
DOCK_LEG_HOLD.setflags(write=False)

# ---------------------------------------------------------------------------
# MJCF jnt_range for the ten leg DOF (open_duck_mini_v2.xml), in LEG_NAMES order.
# ---------------------------------------------------------------------------
# Identical numbers to the runtime controller's ``_MJCF_JOINT_LOW/HIGH`` leg
# slice (single source of truth for the mechanical clamp). The engine clamps the
# *deflection box* to a small window around the hold, then intersects with these
# so a leg target can never leave the mechanical range even if a deflection cap
# is later widened.
LEG_JNT_LOW: np.ndarray = np.array([
    -0.5236,  # left_hip_yaw
    -0.4363,  # left_hip_roll
    -1.2217,  # left_hip_pitch
    -1.5708,  # left_knee
    -1.5708,  # left_ankle
    -0.5236,  # right_hip_yaw
    -0.4363,  # right_hip_roll
    -0.5236,  # right_hip_pitch
    -1.5708,  # right_knee
    -1.5708,  # right_ankle
], dtype=np.float64)
LEG_JNT_HIGH: np.ndarray = np.array([
    0.5236,   # left_hip_yaw
    0.4363,   # left_hip_roll
    0.5236,   # left_hip_pitch
    1.5708,   # left_knee
    1.5708,   # left_ankle
    0.5236,   # right_hip_yaw
    0.4363,   # right_hip_roll
    1.2217,   # right_hip_pitch
    1.5708,   # right_knee
    1.5708,   # right_ankle
], dtype=np.float64)
LEG_JNT_LOW.setflags(write=False)
LEG_JNT_HIGH.setflags(write=False)

# ---------------------------------------------------------------------------
# Per-channel maximum deflection from the dock hold (rad), in LEG_NAMES order.
# ---------------------------------------------------------------------------
# CONSERVATIVE, CHOSEN FOR SELF-COLLISION / CABLE-STRAIN SAFETY — NOT MEASURED
# BALANCE. The dance is deliberately hip-led: hip **yaw** (body twist) and hip
# **roll** (side rock) get the most room because they are the safe, expressive
# "wag" axes and cannot fold a limb into the body. hip **pitch**, **knee** and
# **ankle** are kept nearly still: a large knee/ankle excursion is exactly what
# could drive the foot/shin into the chassis or wrap a cable, and it buys little
# expressively on the dock. Every number is a small fraction of the mechanical
# range above and leaves the docked crouch essentially intact (knee stays within
# ~0.14 rad of its docked 1.37 rad). Verified contact-free in MuJoCo by
# experiments/animation/phase4_dock_fullbody_sim.py.
DOCK_LEG_MAX_DEFLECTION: np.ndarray = np.array([
    0.20,   # left_hip_yaw   — primary twist ("wag") axis
    0.12,   # left_hip_roll  — side rock
    0.10,   # left_hip_pitch — small
    0.08,   # left_knee      — nearly still (self-collision guard)
    0.08,   # left_ankle     — nearly still
    0.20,   # right_hip_yaw
    0.12,   # right_hip_roll
    0.10,   # right_hip_pitch
    0.08,   # right_knee
    0.08,   # right_ankle
], dtype=np.float64)
DOCK_LEG_MAX_DEFLECTION.setflags(write=False)

# First-hardware derating (matches the head envelope's HARDWARE_DERATING, R16).
DOCK_LEG_DERATING: float = 0.5

ArrayLike = Union[np.ndarray, Sequence[float]]


@dataclass
class LegDockEnvelope:
    """Mechanical safety clamp for animated leg targets on the dock (plan §6.2).

    Bounds each of the ten leg targets to ``hold ± max_deflection``, then
    intersects that window with the MJCF ``jnt_range`` so a target can never
    leave the mechanical range. This is a *position* clamp only — velocity is
    handled by the engine's leg rate limit and the downstream
    :class:`~open_duck_anim.limits.JointRateLimiter`.

    Attributes:
        hold: length-10 dock hold pose (rad), LEG_NAMES order.
        max_deflection: length-10 per-channel max deflection from ``hold`` (rad).
        low, high: length-10 MJCF ``jnt_range`` (rad); the deflection window is
            intersected with ``[low, high]``.
    """

    hold: np.ndarray = None  # type: ignore[assignment]
    max_deflection: np.ndarray = None  # type: ignore[assignment]
    low: np.ndarray = None  # type: ignore[assignment]
    high: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.hold is None:
            self.hold = DOCK_LEG_HOLD.copy()
        if self.max_deflection is None:
            self.max_deflection = DOCK_LEG_MAX_DEFLECTION.copy()
        if self.low is None:
            self.low = LEG_JNT_LOW.copy()
        if self.high is None:
            self.high = LEG_JNT_HIGH.copy()
        self.hold = np.array(self.hold, dtype=np.float64)
        self.max_deflection = np.array(self.max_deflection, dtype=np.float64)
        self.low = np.array(self.low, dtype=np.float64)
        self.high = np.array(self.high, dtype=np.float64)
        for name, a in (("hold", self.hold), ("max_deflection", self.max_deflection),
                        ("low", self.low), ("high", self.high)):
            if a.shape != (N_LEG,):
                raise ValueError("%s must be length-%d, got %r" % (name, N_LEG, a.shape))
        if np.any(self.max_deflection < 0):
            raise ValueError("max_deflection must be >= 0")
        if np.any(self.high < self.low):
            raise ValueError("every jnt_range high must be >= its low")
        # Effective per-channel window: intersection of the deflection box around
        # the hold with the mechanical jnt_range.
        self._eff_low = np.maximum(self.low, self.hold - self.max_deflection)
        self._eff_high = np.minimum(self.high, self.hold + self.max_deflection)
        # If the hold itself sits outside jnt_range (should never happen) the
        # window could invert; guard so clamp stays well-defined.
        self._eff_high = np.maximum(self._eff_high, self._eff_low)
        self._eff_low.setflags(write=False)
        self._eff_high.setflags(write=False)
        self.hold.setflags(write=False)
        self.max_deflection.setflags(write=False)
        self.low.setflags(write=False)
        self.high.setflags(write=False)

    def derated(self, factor: float = DOCK_LEG_DERATING) -> "LegDockEnvelope":
        """Return a copy with the per-channel deflections scaled by ``factor``.

        Use ``env.derated()`` for the first hardware trials (see
        :data:`DOCK_LEG_DERATING`). Only the deflection window shrinks; the
        mechanical ``jnt_range`` is unchanged (it is a hard limit, not a
        derating).
        """
        if not 0 < factor <= 1:
            raise ValueError("factor must be in (0, 1]")
        return LegDockEnvelope(
            hold=self.hold.copy(),
            max_deflection=self.max_deflection * factor,
            low=self.low.copy(),
            high=self.high.copy(),
        )

    @property
    def eff_low(self) -> np.ndarray:
        return self._eff_low

    @property
    def eff_high(self) -> np.ndarray:
        return self._eff_high

    def clamp(self, leg_targets: ArrayLike, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Clamp length-10 leg targets into the effective safe window."""
        t = np.asarray(leg_targets, dtype=np.float64)
        if t.shape[-1] != N_LEG:
            raise ValueError("leg_targets last axis must be %d, got %r" % (N_LEG, t.shape))
        # Sanitise non-finite to the hold (last line of defence, mirrors the head
        # envelope): a NaN leg target must never reach a servo.
        if not np.all(np.isfinite(t)):
            t = np.where(np.isfinite(t), t, self.hold)
        return np.clip(t, self._eff_low, self._eff_high, out=out)


# Module-level defaults built from the constants above.
DEFAULT_LEG_ENVELOPE = LegDockEnvelope()
DERATED_LEG_ENVELOPE = DEFAULT_LEG_ENVELOPE.derated(DOCK_LEG_DERATING)
