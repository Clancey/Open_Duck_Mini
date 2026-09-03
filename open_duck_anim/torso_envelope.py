"""Safe torso-posture command envelope for the STAND full-body-emotion path.

The torso-command *standing* policy (Disney BD-X arXiv:2501.05204 Eq.5;
``Open_Duck_Playground .../standing.py``) takes torso **height** and torso
**orientation** as commands, in addition to the head passthrough. This module
owns and enforces the safe operating envelope for that torso command — the
posture analogue of :mod:`open_duck_anim.envelope` (the head) and
:mod:`open_duck_anim.leg_envelope` (the dock legs).

The three posture command channels (:data:`POSTURE_COMMAND_CHANNELS`) are the
engine's *offsets from neutral standing*:

* ``torso_height_delta`` — metres added to the nominal standing torso height
  (``+`` taller/puff, ``-`` lower/sag). Neutral = 0.
* ``torso_grav_x`` — target x of the IMU up-vector's projected gravity
  (``~ sin(pitch)``; ``+`` = torso pitched forward / nose-down). Neutral = 0.
* ``torso_grav_y`` — target y of the projected gravity (``~ -sin(roll)``).
  Neutral = 0.

Guards (identical structure to :class:`open_duck_anim.envelope.HeadEnvelope`),
applied in order: (0) finiteness repair, (1) per-channel deflection clamp,
(2) a combined multi-axis L2 budget, (3) a per-channel slew-rate cap LAST so the
returned command satisfies all three invariants simultaneously.

.. note::

   **HEIGHT is now MEASURED; ORIENTATION is still provisional.**

   The **height** limits below were swept against a real trained standing
   checkpoint (:data:`MEASURED_CHECKPOINT`, the Phase-3 torso-posture policy) with
   the plan §6.5 methodology used for the head: a fine command sweep, ≥5 s holds
   (instability proved non-monotonic and time-dependent), pushes and observation
   noise disabled. Result: **across the entire commanded height sweep
   (0.130–0.190 m about a ~0.159 m measured nominal) the duck NEVER fell** — the
   binding limit is *tracking saturation*, not topple. Achieved base height floors
   at ~0.140 m (deepest sag) and ceilings at ~0.177 m (tallest), and tracks the
   command ~1:1 in between. The shipped ``torso_height_delta`` box is the
   faithfully-tracked, zero-fall sub-band of that reach.

   The two **orientation** channels (``grav_x`` pitch, ``grav_y`` roll) are NOT
   usable and, on current evidence, **not achievable on this hardware**. TWO
   independently-trained standing policies keep the torso upright regardless of
   the commanded lean: the v1 checkpoint pinned here (orientation weight -1.0) and
   a second run v2 with **4x the orientation weight, a tighter error scale and
   holdable-only ranges** BOTH tracked achieved projected-gravity ≈0 for every
   commanded pitch/roll (v2 additionally regressed the height sag, so v1 is the
   shipped policy). The duck's small feet and head-heavy mass mean a *sustained
   static* torso lean is not balance-holdable, so the policy correctly refuses it.
   The orientation limits below are therefore kept only as an inert conservative
   box; callers should command ``grav_x = grav_y = 0`` and express "hunch"/"droop"
   via the HEAD channels instead (see the report). Re-open only if a future policy
   (e.g. a dynamic/transient lean rather than a static hold) demonstrates tracking
   in a sweep. Keep :data:`HARDWARE_DERATING` applied.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

# Posture command channel order within the length-3 torso command.
POSTURE_COMMAND_CHANNELS: Tuple[str, str, str] = (
    "torso_height_delta",  # metres from nominal standing height
    "torso_grav_x",        # target projected-gravity x (~ sin pitch)
    "torso_grav_y",        # target projected-gravity y (~ -sin roll)
)

# Control tick used to derive the per-tick slew step (plan §6.6: ctrl_dt = 0.02).
CTRL_DT: float = 0.02

# The trained standing checkpoint the HEIGHT limits below were swept against.
# (Phase-3 torso-posture policy, 300M steps; obs [1,88] -> continuous_actions
# [1,14]. This is a NEW, SEPARATE policy — it does NOT replace the deployed
# 101-wide walking passthrough.) Pin the envelope to the checkpoint that produced
# it, exactly as open_duck_anim.envelope pins the head numbers.
MEASURED_CHECKPOINT: str = "standing_torso_20260903_011541/2026_09_03_023626_300482560.onnx"
MEASURED_NOMINAL_HEIGHT: float = 0.159  # m, measured neutral-command base height

# Conservative fraction of a measured failure onset that would be adopted as the
# safe limit ONCE a sweep exists (see :mod:`open_duck_anim.envelope`). No onset
# has been measured yet, so the defaults below are NOT `SAFETY_FRACTION * onset`;
# they are the raw conservative kinematic reach. Kept here so a future sweep uses
# the same 2x margin the head uses.
SAFETY_FRACTION: float = 0.5

# ---------------------------------------------------------------------------
# Per-channel deflection limits, as (low, high) offsets from neutral standing.
# ---------------------------------------------------------------------------
#   height  : MEASURED against MEASURED_CHECKPOINT. The full commanded sweep
#             0.130..0.190 m held ≥5 s with ZERO falls; achieved base height
#             floored at ~0.140 m and ceilinged at ~0.177 m about a ~0.159 m
#             measured nominal, tracking ~1:1 in between. Shipped box is the
#             faithfully-tracked, zero-fall sub-band: (-0.020, +0.016) m
#             (achieves ~0.140 m sag .. ~0.172 m puff). NOTE the limit here is
#             tracking saturation, not topple — nothing fell — so this is not
#             halved by SAFETY_FRACTION; the 2x hardware margin is applied
#             separately via HARDWARE_DERATING.
#   grav_x  : NOT usable / not achievable. Two policies (v1 weight -1.0, v2 weight
#             -4.0) both refuse to lean (achieved pitch ≈0 for every command;
#             never fell). Kept as an inert conservative box (+/-0.12); command 0.
#   grav_y  : NOT usable / not achievable, as grav_x. Narrow lateral base -> roll
#             is the riskiest axis; kept tighter at +/-0.06. Command 0.
DEFLECTION_LIMITS: Dict[str, Tuple[float, float]] = {
    "torso_height_delta": (-0.020, 0.016),
    "torso_grav_x": (-0.12, 0.12),
    "torso_grav_y": (-0.06, 0.06),
}

DEFLECTION_LOW: np.ndarray = np.array(
    [DEFLECTION_LIMITS[c][0] for c in POSTURE_COMMAND_CHANNELS], dtype=np.float64
)
DEFLECTION_HIGH: np.ndarray = np.array(
    [DEFLECTION_LIMITS[c][1] for c in POSTURE_COMMAND_CHANNELS], dtype=np.float64
)
DEFLECTION_LOW.setflags(write=False)
DEFLECTION_HIGH.setflags(write=False)

# ---------------------------------------------------------------------------
# Per-channel command slew-rate limits (units/s). Per-channel (unlike the head's
# single scalar) because the channels carry different units (m vs dimensionless
# projected-gravity). Defence-in-depth: a posture change should ease in over the
# T_alpha (~0.35 s) body blend, never step. A 2 cm sag over 0.35 s is ~0.057 m/s,
# so 0.10 m/s leaves margin; the grav channels are capped so a full-range swing
# takes >~0.2 s. PROVISIONAL — re-derive alongside the deflection sweep.
SLEW_LIMITS: Dict[str, float] = {
    "torso_height_delta": 0.10,  # m/s
    "torso_grav_x": 1.0,         # 1/s
    "torso_grav_y": 0.8,         # 1/s
}
SLEW_LIMIT_VEC: np.ndarray = np.array(
    [SLEW_LIMITS[c] for c in POSTURE_COMMAND_CHANNELS], dtype=np.float64
)
SLEW_LIMIT_VEC.setflags(write=False)

# Combined multi-axis L2 budget (dimensionless, in per-axis-normalised units).
# UNCALIBRATED coupling guard. The per-channel box (step 1) is the hard cap; this
# only scales DOWN when several axes are large at once. It must be >= the worst
# single-axis normalised edge so that ANY one channel at its own box edge passes
# unscaled -- the height box is asymmetric (sag -0.020 vs puff +0.016) and the
# normaliser uses L = min(|low|, high) = 0.016, so the sag edge normalises to
# 0.020/0.016 = 1.25. Set the budget to 1.25 so the full measured -0.020 m sag
# (the owner's primary deliverable) is never clipped by coupling; all three axes
# at their edges (norm ~1.73) are still scaled back. Re-measure with an
# adversarial coupled sweep before trusting simultaneous multi-axis posture.
COMBINED_L2_BUDGET: float = 1.25

# Additional derating for first hardware trials (multiplies the deflection box).
# Held at 0.5 and, unlike the head, NOT relaxable until a first sweep exists at
# all: this envelope has never been validated against a policy.
HARDWARE_DERATING: float = 0.5

ArrayLike = Union[np.ndarray, Sequence[float]]


def _as_posture_vec(x: ArrayLike, name: str) -> np.ndarray:
    """Coerce to a 1-D length-3 float64 posture command vector."""
    arr = np.asarray(x)
    if arr.dtype == object or not np.issubdtype(arr.dtype, np.number):
        raise TypeError("%s must be a numeric length-3 torso-command array" % name)
    arr = arr.astype(np.float64, copy=False)
    if arr.shape != (3,):
        raise ValueError(
            "%s must be a 1-D length-3 torso vector (got shape %r)" % (name, arr.shape)
        )
    return arr


def _sanitize(c: np.ndarray, prev: Optional[np.ndarray]) -> Tuple[np.ndarray, bool]:
    """Replace non-finite channels with a safe fallback; report a fault.

    Mirrors :func:`open_duck_anim.envelope._sanitize_head`: NaN/Inf must never
    reach the clamp math (``np.clip`` and the L2 scale both propagate NaN, which
    would silently disable the guard). Substitute each non-finite channel with
    the previous *finite* enforced value (hold-last-good) or ``0.0`` (neutral),
    and flag ``fault=True``. ``c`` is mutated in place (already a private copy).
    """
    bad = ~np.isfinite(c)
    if not bad.any():
        return c, False
    fallback = (
        np.where(np.isfinite(prev), prev, 0.0)
        if prev is not None
        else np.zeros(3, dtype=np.float64)
    )
    c[bad] = fallback[bad]
    return c, True


@dataclass
class TorsoEnvelope:
    """Enforced safe torso-posture envelope: deflection + budget + per-axis slew.

    Structurally identical to :class:`open_duck_anim.envelope.HeadEnvelope`, with
    a length-3 command and a per-channel slew vector (the channels carry mixed
    units). See the module warning: the default limits are UNSWEPT kinematic
    placeholders and MUST be re-derived against a trained standing checkpoint
    before hardware. All operations are stateless (the caller owns ``prev``).
    """

    low: np.ndarray = field(default_factory=lambda: DEFLECTION_LOW.copy())
    high: np.ndarray = field(default_factory=lambda: DEFLECTION_HIGH.copy())
    slew_limit: np.ndarray = field(default_factory=lambda: SLEW_LIMIT_VEC.copy())
    l2_budget: float = COMBINED_L2_BUDGET
    bypass: bool = False

    def __post_init__(self) -> None:
        self.low = np.array(self.low, dtype=np.float64)
        self.high = np.array(self.high, dtype=np.float64)
        self.slew_limit = np.array(self.slew_limit, dtype=np.float64)
        if self.low.shape != (3,) or self.high.shape != (3,):
            raise ValueError("low/high must be length-3 torso vectors")
        if self.slew_limit.shape != (3,):
            raise ValueError("slew_limit must be a length-3 torso vector")
        if np.any(self.high < self.low):
            raise ValueError("every high must be >= its low")
        if not self.bypass:
            if np.any(self.low > 0) or np.any(self.high < 0):
                raise ValueError("deflection limits must bracket 0 (offsets from neutral)")
            if np.any(self.slew_limit <= 0):
                raise ValueError("every slew_limit must be > 0")
            if not (self.l2_budget > 0):
                raise ValueError("l2_budget must be > 0")
        self.low.setflags(write=False)
        self.high.setflags(write=False)
        self.slew_limit.setflags(write=False)
        L = np.minimum(np.abs(self.low), self.high)
        self._L = np.where(L > 1e-12, L, 1e-12)
        self._L.setflags(write=False)

    @classmethod
    def unbounded(cls) -> "TorsoEnvelope":
        """Explicit, greppable escape hatch that disables ALL enforcement."""
        inf = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
        return cls(low=-inf, high=inf, slew_limit=inf, l2_budget=np.inf, bypass=True)

    def derated(self, factor: float = HARDWARE_DERATING) -> "TorsoEnvelope":
        """Return a copy with the deflection limits scaled by ``factor``.

        Only ``low``/``high`` are scaled; the L2 budget is expressed in units
        normalised by ``L = min(|low|, high)`` so scaling the box already derates
        the combined constraint (see the head envelope's ``derated`` note). Slew
        is a rate cap already inside the safe region and is left unchanged.
        """
        if not 0 < factor <= 1:
            raise ValueError("factor must be in (0, 1]")
        return TorsoEnvelope(
            low=self.low * factor,
            high=self.high * factor,
            slew_limit=self.slew_limit,
            l2_budget=self.l2_budget,
        )

    def _combined_scale(self, c: np.ndarray) -> float:
        norm = float(np.sqrt(np.sum((c / self._L) ** 2)))
        if norm <= self.l2_budget or norm == 0.0:
            return 1.0
        return self.l2_budget / norm

    def clamp(
        self,
        command_torso: ArrayLike,
        prev_command_torso: Optional[ArrayLike] = None,
        dt: float = CTRL_DT,
        out: Optional[np.ndarray] = None,
        return_fault: bool = False,
    ):
        """Enforce the envelope on a length-3 torso posture command offset.

        Args mirror :meth:`HeadEnvelope.clamp`: ``prev_command_torso`` is the
        previous *enforced* command (the slew reference; ``None`` skips slew on
        the first tick), ``dt`` the tick duration, ``out`` an optional buffer,
        ``return_fault`` to also return whether a non-finite channel was repaired.
        """
        c = _as_posture_vec(command_torso, "command_torso").copy()

        if self.bypass:
            return (self._finish(c, out), False) if return_fault else self._finish(c, out)

        p = None
        if prev_command_torso is not None:
            p = _as_posture_vec(prev_command_torso, "prev_command_torso").copy()
            p_bad = ~np.isfinite(p)
            if p_bad.any():
                p[p_bad] = 0.0

        # (0) finiteness repair.
        c, fault = _sanitize(c, p)
        # (1) per-channel deflection clamp.
        np.clip(c, self.low, self.high, out=c)
        # (2) combined multi-axis L2 budget.
        scale = self._combined_scale(c)
        if scale < 1.0:
            c *= scale
        # (3) per-channel slew guard LAST.
        if p is not None:
            if not (dt > 0):
                raise ValueError("dt must be > 0")
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


# Module-level default (enforcing) and derated envelopes.
DEFAULT_TORSO_ENVELOPE = TorsoEnvelope()
DERATED_TORSO_ENVELOPE = DEFAULT_TORSO_ENVELOPE.derated()


def posture_to_command_offsets(posture: ArrayLike) -> np.ndarray:
    """Convert an authoring posture triple to standing-policy command offsets.

    Authoring posture is ``[torso_height_delta_m, torso_pitch_rad,
    torso_roll_rad]`` (intuitive units; see
    :data:`open_duck_anim.clip.POSTURE_CHANNELS`). The standing policy commands
    torso *orientation* as the target of the IMU up-vector's projected gravity,
    so pitch/roll are mapped through ``sin``:

    * ``grav_x = sin(pitch)``  (torso pitched forward → +x)
    * ``grav_y = -sin(roll)``  (roll right → -y), matching the sensor convention
      documented in ``standing.py``.

    Height passes through unchanged (already an offset in metres). Returns a
    length-3 ``[torso_height_delta, torso_grav_x, torso_grav_y]`` — the raw,
    UNCLAMPED offset; the caller clamps it through :meth:`TorsoEnvelope.clamp`.
    """
    p = np.asarray(posture, dtype=np.float64)
    if p.shape != (3,):
        raise ValueError("posture must be length-3, got shape %r" % (p.shape,))
    return np.array([p[0], np.sin(p[1]), -np.sin(p[2])], dtype=np.float64)
