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

.. warning::

   **THESE LIMITS ARE UNSWEPT PLACEHOLDERS — NOT A MEASURED SAFE ENVELOPE.**

   Unlike :mod:`open_duck_anim.envelope`, whose numbers were swept against a real
   trained checkpoint, **no torso-command policy has been trained yet**, so there
   is NO measured *holdable* range and NO measured topple onset. The defaults
   below are the conservative **kinematic reach** of the MJCF (how far the duck
   *could* move its torso if it stayed balanced), NOT the balance-limited range
   an RL policy can actually hold. Torso posture directly affects balance, so
   before ANY hardware use these MUST be re-derived by sweeping the trained
   standing checkpoint with the plan §6.5 methodology used for the head
   (``experiments/animation/envelope_sweep.py``): a genuinely-fine first-onset
   outward sweep, ≥5 s holds (head instability proved non-monotonic and
   time-dependent), across the full command grid. Until then treat this envelope
   as *sim-only, provisional* and keep :data:`HARDWARE_DERATING` applied.
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

# Conservative fraction of a measured failure onset that would be adopted as the
# safe limit ONCE a sweep exists (see :mod:`open_duck_anim.envelope`). No onset
# has been measured yet, so the defaults below are NOT `SAFETY_FRACTION * onset`;
# they are the raw conservative kinematic reach. Kept here so a future sweep uses
# the same 2x margin the head uses.
SAFETY_FRACTION: float = 0.5

# ---------------------------------------------------------------------------
# Per-channel deflection limits, as (low, high) offsets from neutral standing.
# ---------------------------------------------------------------------------
# PROVISIONAL / KINEMATIC — NOT policy-measured (see module warning).
#
#   height  : MJCF kinematic reach is ~0.127..0.195 m about a ~0.16 m nominal
#             (roughly -0.033 / +0.035 m). Shipped TIGHTER than the reach because
#             the *balance-holdable* range is unknown and is expected to be a
#             subset of the reach: (-0.020, +0.020) m.
#   grav_x  : training range is +/-0.20 (~11.5 deg pitch). Shipped tighter until
#             swept: (-0.12, +0.12).
#   grav_y  : narrow lateral base -> roll is the riskiest axis. training +/-0.10;
#             shipped tighter: (-0.06, +0.06).
DEFLECTION_LIMITS: Dict[str, Tuple[float, float]] = {
    "torso_height_delta": (-0.020, 0.020),
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
# UNCALIBRATED. Without a coupled-axis sweep we cannot measure the resonance
# budget the head has (0.70); the per-channel box is the real guard for now, so
# we set a mild coupling penalty of 1.0 (a single channel at its box edge is
# allowed; all three simultaneously at their edges are scaled back to ||n||=1).
# Re-measure with an adversarial coupled sweep before trusting multi-axis posture.
COMBINED_L2_BUDGET: float = 1.0

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
