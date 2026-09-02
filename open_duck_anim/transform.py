"""Absolute authored head pose → relative command offset transform (plan §6.3).

Authored clips store **absolute** head joint angles; the RL policy consumes
**relative** command offsets on ``commands[3:7]``. The transform is:

    animation_delta = authored_head_pose - authored_nominal_pose      # per channel
    command[3:7]    = base_command[3:7] + animation_delta + joystick  # additive
    command[3:7]    = clamp(command[3:7], training_range)             # then clamp

The four head channels, in order, are
``[neck_pitch, head_pitch, head_yaw, head_roll]`` (plan §3.3 command layout,
indices 3..6 of the length-7 command vector).

Clamping: the plan says clamp to the **training command ranges** first (because
commands outside the trained range produce undefined policy behaviour), then to
physical joint limits, "the tighter of the two governs". This module owns the
training-range clamp; physical ``jnt_range`` clamping is applied separately by
:mod:`open_duck_anim.limits` on the final targets. Applying the training clamp
here is safe because it is the tighter bound for the command channel.
"""

from typing import Optional, Tuple

import numpy as np

from .envelope import HeadEnvelope, DEFAULT_ENVELOPE

# Head channel order within commands[3:7] (plan §3.3 / §6.3).
HEAD_CHANNELS: Tuple[str, str, str, str] = (
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
)

# Training command ranges (joystick.py:94-101, plan §6.3 / Appendix C).
TRAINING_RANGES: Tuple[Tuple[float, float], ...] = (
    (-0.34, 1.1),   # neck_pitch
    (-0.78, 0.78),  # head_pitch
    (-1.5, 1.5),    # head_yaw
    (-0.5, 0.5),    # head_roll
)

TRAINING_LOW: np.ndarray = np.array([r[0] for r in TRAINING_RANGES], dtype=np.float64)
TRAINING_HIGH: np.ndarray = np.array([r[1] for r in TRAINING_RANGES], dtype=np.float64)
TRAINING_LOW.setflags(write=False)
TRAINING_HIGH.setflags(write=False)

# Default authored nominal head pose is all-zero head channels (plan §6.3:
# "authored_nominal_pose is the clip's neutral head pose, typically all-zero").
NOMINAL_HEAD_POSE: np.ndarray = np.zeros(4, dtype=np.float64)
NOMINAL_HEAD_POSE.setflags(write=False)


def animation_delta(
    authored_head_pose: np.ndarray,
    authored_nominal_pose: np.ndarray = NOMINAL_HEAD_POSE,
) -> np.ndarray:
    """Return ``authored_head_pose - authored_nominal_pose`` (plan §6.3).

    Both arguments are length-4 head-channel vectors in :data:`HEAD_CHANNELS`
    order. Returns a new length-4 ``float64`` array.
    """
    pose = np.asarray(authored_head_pose, dtype=np.float64)
    nominal = np.asarray(authored_nominal_pose, dtype=np.float64)
    if pose.shape[-1] != 4 or nominal.shape[-1] != 4:
        raise ValueError("head pose vectors must have last axis == 4")
    return pose - nominal


def clamp_training_range(command_head: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
    """Clamp head command channels to the training ranges (plan §6.3).

    ``command_head`` is a length-4 vector in :data:`HEAD_CHANNELS` order.
    """
    c = np.asarray(command_head, dtype=np.float64)
    if c.shape[-1] != 4:
        raise ValueError("command_head must have last axis == 4")
    return np.clip(c, TRAINING_LOW, TRAINING_HIGH, out=out)


def pose_to_command(
    authored_head_pose: np.ndarray,
    base_command: np.ndarray = NOMINAL_HEAD_POSE,
    joystick_offset: Optional[np.ndarray] = None,
    authored_nominal_pose: np.ndarray = NOMINAL_HEAD_POSE,
    head_envelope: "HeadEnvelope" = DEFAULT_ENVELOPE,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Full plan §6.3 transform: absolute authored head pose → safe command.

    SAFETY (reviewer E3, plan §6.5/D13): the training-range clamp alone is NOT
    balance-safe — it permits ``neck_pitch`` to +1.1 and ``head_yaw`` to -1.5,
    well past the measured topple onsets. So this transform routes its output
    through the D13 safety envelope by DEFAULT (``head_envelope=DEFAULT_ENVELOPE``):
    per-channel deflection clamp + combined L2 budget. Enforcement is OPT-OUT —
    to obtain the raw training-range command you must pass the explicit, greppable
    ``head_envelope=HeadEnvelope.unbounded()`` sentinel, so the decision is
    auditable. NOTE: this static path applies deflection + combined limits but
    NOT the slew/rate guard (that needs per-tick state); drive dynamic/authored
    motion through :class:`open_duck_anim.blend.Engine`, which owns the clock and
    applies the full envelope including slew.

    Args:
        authored_head_pose: length-4 absolute authored head angles.
        base_command: length-4 base head command (the ``commands[3:7]`` the
            policy would otherwise see; defaults to nominal/zero).
        joystick_offset: optional length-4 additive gaze offset (plan §6.3:
            "joystick composes additively"). ``None`` means no offset.
        authored_nominal_pose: the clip's neutral head pose (default zero).
        head_envelope: the D13 safety envelope to enforce (default
            :data:`~open_duck_anim.envelope.DEFAULT_ENVELOPE`). Pass
            ``HeadEnvelope.unbounded()`` to deliberately disable it.
        out: optional preallocated length-4 output buffer (hot-path friendly).

    Returns the length-4 head command: clamped to the training ranges, then
    enforced through ``head_envelope`` (deflection + combined budget).
    """
    delta = animation_delta(authored_head_pose, authored_nominal_pose)
    base = np.asarray(base_command, dtype=np.float64)
    command = base + delta
    if joystick_offset is not None:
        command = command + np.asarray(joystick_offset, dtype=np.float64)
    command = clamp_training_range(command)
    # D13/R16 balance-safety envelope (deflection + combined budget). The
    # command channels are deflections from the (zero) nominal head pose, i.e.
    # exactly the quantity the S0.1 sweep drove, so the envelope applies directly.
    return head_envelope.clamp(command, out=out)
