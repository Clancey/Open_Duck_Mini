"""Safety utilities: joint clamping, rate limiting, antenna slew (plan §6.4, §6.5).

Three distinct, deliberately non-interchangeable limiters:

1. :class:`JointLimiter` — per-joint position clamp against a configurable
   ``jnt_range`` table (plan §6.4: "All joint outputs are clamped to MJCF
   ``jnt_range``").
2. :class:`JointRateLimiter` — velocity clamp at ``max_motor_velocity =
   5.24 rad/s`` applied given a ``dt``. **Intended for the FINAL 14-DOF bus
   targets, after policy-vs-direct-mode selection (plan §6.4 / S5) — NOT for
   animation commands.** Limiting an animation command does not constrain what
   the policy emits.
3. :class:`AntennaSlewLimiter` — a *separate* normalised slew limit on the
   ``[-1,1]`` antenna track (plan §6.4 / Appendix A: hobby servos held against a
   stop overheat, so antenna motion must be slew-limited).

Discrete ``events`` (eyes/sounds/projector) are **never** rate-limited as joint
angles (plan §6.4). This is made structurally impossible: the rate limiters only
accept numeric float arrays and raise :class:`TypeError` on anything else, and
there is no code path that feeds an event object into a rate limiter. Events are
carried as :class:`DiscreteEvent` objects (see :mod:`open_duck_anim.clip`) which
expose no numeric interface a limiter could consume.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np

# max_motor_velocity (joystick.py:49-60, plan §6.4 / Appendix C / D7).
MAX_MOTOR_VELOCITY = 5.24  # rad/s

# Default normalised antenna slew rate (units of the [-1,1] track per second).
# The plan mandates a separate antenna slew limit but gives no numeric constant.
# This value was LOWERED from an earlier arbitrary 8.0 (full [-1,1] span in
# ~0.25 s) to 4.0 (full span in ~0.5 s) in response to real-hardware feedback:
# the robot's owner reported the open-loop 9g hobby antenna servos (GPIO D13/D12)
# were audibly noisy — PWM hobby servos buzz and chatter in proportion to how
# fast and how often they are driven. A lower global cap is defence-in-depth so
# that *any* clip, including ones authored in the future, cannot drive the
# antennas harshly regardless of what its tracks request. The shipped clip
# library is authored to stay within this cap (peak authored slew ~3.8 units/s),
# so at runtime this limiter is a no-op on the current clips — it only bites on
# pathological or future over-driven motion. Still tunable per-deployment.
DEFAULT_ANTENNA_SLEW = 4.0  # normalised units / s

ArrayLike = Union[np.ndarray, Sequence[float]]


def _as_float_array(x: ArrayLike, name: str) -> np.ndarray:
    """Coerce to a float ndarray, rejecting non-numeric inputs.

    This is the structural guard that prevents discrete events (or any
    non-joint-angle payload) from being rate-limited as joint angles.
    """
    arr = np.asarray(x)
    if arr.dtype == object or not np.issubdtype(arr.dtype, np.number):
        raise TypeError(
            "%s must be a numeric float array of joint angles; discrete events "
            "are never rate-limited as joint angles (plan §6.4)" % name
        )
    return arr.astype(np.float64, copy=False)


@dataclass
class JointLimiter:
    """Per-joint position clamp against a configurable range table (plan §6.4).

    ``low`` and ``high`` are arrays of equal length (one entry per joint). The
    clamp is stateless and allocation-light (supports an ``out`` buffer).
    """

    low: np.ndarray
    high: np.ndarray

    def __post_init__(self) -> None:
        self.low = np.asarray(self.low, dtype=np.float64)
        self.high = np.asarray(self.high, dtype=np.float64)
        if self.low.shape != self.high.shape:
            raise ValueError("low/high must have the same shape")
        if np.any(self.high < self.low):
            raise ValueError("every high must be >= its low")

    def clamp(self, targets: ArrayLike, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Clamp ``targets`` element-wise into ``[low, high]``."""
        t = _as_float_array(targets, "targets")
        if t.shape[-1] != self.low.shape[-1]:
            raise ValueError(
                "targets last axis %d != limits length %d"
                % (t.shape[-1], self.low.shape[-1])
            )
        return np.clip(t, self.low, self.high, out=out)


@dataclass
class JointRateLimiter:
    """Velocity clamp for the FINAL 14-DOF bus targets (plan §6.4 / S5, D7).

    Applies ``max_velocity`` rad/s given a ``dt``: the per-tick step is clamped
    to ``±max_velocity * dt`` around the previous target.

    IMPORTANT: apply this to the actual bus targets after policy/direct-mode
    selection, never to animation commands (limiting an animation command does
    not constrain the policy output). See plan §6.4.
    """

    max_velocity: float = MAX_MOTOR_VELOCITY

    def __post_init__(self) -> None:
        if self.max_velocity <= 0:
            raise ValueError("max_velocity must be > 0")

    def limit(
        self,
        prev: ArrayLike,
        target: ArrayLike,
        dt: float,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return ``target`` moved from ``prev`` by at most ``max_velocity*dt``."""
        if dt <= 0:
            raise ValueError("dt must be > 0")
        p = _as_float_array(prev, "prev")
        t = _as_float_array(target, "target")
        max_step = self.max_velocity * dt
        step = np.clip(t - p, -max_step, max_step)
        result = np.add(p, step, out=out)
        return result


@dataclass
class AntennaSlewLimiter:
    """Separate normalised slew limiter for antennas (plan §6.4, Appendix A).

    Operates on the normalised ``[-1,1]`` antenna track; ``max_slew`` is in
    normalised units per second. This is intentionally distinct from
    :class:`JointRateLimiter` (different units, different limits) so antenna slew
    can be tuned to avoid stall-holds without touching joint velocity limits.
    Values are also clamped into ``[-1, 1]`` (the physical track bound).
    """

    max_slew: float = DEFAULT_ANTENNA_SLEW

    def __post_init__(self) -> None:
        if self.max_slew <= 0:
            raise ValueError("max_slew must be > 0")

    def limit(
        self,
        prev: ArrayLike,
        target: ArrayLike,
        dt: float,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Slew-limit then clamp normalised antenna targets into ``[-1, 1]``."""
        if dt <= 0:
            raise ValueError("dt must be > 0")
        p = _as_float_array(prev, "prev")
        t = _as_float_array(target, "target")
        max_step = self.max_slew * dt
        step = np.clip(t - p, -max_step, max_step)
        slewed = p + step
        return np.clip(slewed, -1.0, 1.0, out=out)
