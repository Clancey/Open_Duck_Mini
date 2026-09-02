"""Canonical joint ordering — the single source of truth (plan §4.1-C, Appendix A).

Two orderings exist in the ecosystem and they differ **only** by the two antenna
channels:

* ``JOINT_ORDER_16`` — the reference / authoring order used by the 59-float
  reference JSON and ``.duckanim`` joint arrays (``poly_reference_motion.py``).
  Antennas live at indices 9 (``left_antenna``) and 10 (``right_antenna``).
* ``HW_ORDER_14`` — the Feetech STS3215 bus order used by the runtime action
  vector (``rustypot_position_hwi.py``). It has no antennas.

Conversion is a pure index drop (16→14, remove 9,10) / insert (14→16, add
antennas). Per plan §3.3 / Appendix A this must be an explicit, tested module,
never an ad-hoc slice.

Design choices where the plan is silent:

* ``to_hw14`` / ``to_ref16`` accept either a single frame ``(16,)`` / ``(14,)``
  or a batch ``(N,16)`` / ``(N,14)`` and always operate on the **last** axis, so
  they compose with arbitrary leading batch dimensions.
* Antennas default to ``0.0`` (neutral) when re-inflating a 14→16 vector, since
  the hardware carries no antenna feedback and the runtime never reads antenna
  values from the joint array anyway (plan §5.2 precedence rule).
"""

from typing import List, Sequence, Union

import numpy as np

# --- Canonical 16-joint reference/authoring order (Appendix A) ----------------
JOINT_ORDER_16: List[str] = [
    "left_hip_yaw",   # 0
    "left_hip_roll",  # 1
    "left_hip_pitch", # 2
    "left_knee",      # 3
    "left_ankle",     # 4
    "neck_pitch",     # 5
    "head_pitch",     # 6
    "head_yaw",       # 7
    "head_roll",      # 8
    "left_antenna",   # 9
    "right_antenna",  # 10
    "right_hip_yaw",  # 11
    "right_hip_roll", # 12
    "right_hip_pitch",# 13
    "right_knee",     # 14
    "right_ankle",    # 15
]

# --- Hardware 14-DOF Feetech bus order (Appendix A) ---------------------------
# This is JOINT_ORDER_16 with the antenna entries (9, 10) removed.
HW_ORDER_14: List[str] = [
    "left_hip_yaw",   # 0
    "left_hip_roll",  # 1
    "left_hip_pitch", # 2
    "left_knee",      # 3
    "left_ankle",     # 4
    "neck_pitch",     # 5
    "head_pitch",     # 6
    "head_yaw",       # 7
    "head_roll",      # 8
    "right_hip_yaw",  # 9
    "right_hip_roll", # 10
    "right_hip_pitch",# 11
    "right_knee",     # 12
    "right_ankle",    # 13
]

N_JOINTS_16 = 16
N_JOINTS_14 = 14

# Indices of the antenna channels within the 16-order (the only difference).
ANTENNA_INDICES_16 = (9, 10)
LEFT_ANTENNA_INDEX_16 = 9
RIGHT_ANTENNA_INDEX_16 = 10

# --- Head block named index constants -----------------------------------------
# The head/neck block (neck_pitch, head_pitch, head_yaw, head_roll) occupies
# indices 5..8 in BOTH orders. We define the constants once against each order's
# list (derived, not hardcoded twice) and assert the coincidence in the tests
# rather than relying on it silently here.
NECK_PITCH_16 = JOINT_ORDER_16.index("neck_pitch")
HEAD_PITCH_16 = JOINT_ORDER_16.index("head_pitch")
HEAD_YAW_16 = JOINT_ORDER_16.index("head_yaw")
HEAD_ROLL_16 = JOINT_ORDER_16.index("head_roll")

NECK_PITCH_14 = HW_ORDER_14.index("neck_pitch")
HEAD_PITCH_14 = HW_ORDER_14.index("head_pitch")
HEAD_YAW_14 = HW_ORDER_14.index("head_yaw")
HEAD_ROLL_14 = HW_ORDER_14.index("head_roll")

# The 4 head channels as contiguous slices (5..9 exclusive) in both orders.
HEAD_SLICE_16 = slice(NECK_PITCH_16, HEAD_ROLL_16 + 1)
HEAD_SLICE_14 = slice(NECK_PITCH_14, HEAD_ROLL_14 + 1)

# Leg channels in the 16-order = everything that is neither head nor antenna.
_HEAD_SET = {NECK_PITCH_16, HEAD_PITCH_16, HEAD_YAW_16, HEAD_ROLL_16}
LEG_INDICES_16 = tuple(
    i for i in range(N_JOINTS_16)
    if i not in _HEAD_SET and i not in ANTENNA_INDICES_16
)

# --- Nominal pose (Appendix A, init_pos in rad, 14-DOF bus order) --------------
INIT_POS_14: np.ndarray = np.array(
    [
        0.002,   # left_hip_yaw
        0.053,   # left_hip_roll
        -0.63,   # left_hip_pitch
        1.368,   # left_knee
        -0.784,  # left_ankle
        0.0,     # neck_pitch
        0.0,     # head_pitch
        0.0,     # head_yaw
        0.0,     # head_roll
        -0.003,  # right_hip_yaw
        -0.065,  # right_hip_roll
        0.635,   # right_hip_pitch
        1.379,   # right_knee
        -0.796,  # right_ankle
    ],
    dtype=np.float64,
)
INIT_POS_14.setflags(write=False)

# All-zeros reference pose (Appendix A "zero_pos").
ZERO_POS_14: np.ndarray = np.zeros(N_JOINTS_14, dtype=np.float64)
ZERO_POS_14.setflags(write=False)

ArrayLike = Union[np.ndarray, Sequence[float]]


def to_hw14(arr16: ArrayLike) -> np.ndarray:
    """Convert a 16-joint reference vector to the 14-DOF hardware bus order.

    Drops the antenna channels at indices 9, 10 (plan Appendix A "index drop").
    Operates on the **last** axis, so both a single frame ``(16,)`` and a batch
    ``(..., 16)`` are supported.

    Returns a new ``float64`` array; the input is never mutated.
    """
    a = np.asarray(arr16, dtype=np.float64)
    if a.shape[-1] != N_JOINTS_16:
        raise ValueError(
            "to_hw14 expects last axis == 16, got shape %r" % (a.shape,)
        )
    return np.delete(a, ANTENNA_INDICES_16, axis=-1)


def to_ref16(arr14: ArrayLike, antennas: ArrayLike = (0.0, 0.0)) -> np.ndarray:
    """Convert a 14-DOF hardware vector back to the 16-joint reference order.

    Inserts ``antennas = (left_antenna, right_antenna)`` at indices 9, 10 (plan
    Appendix A "index insert"). Operates on the last axis and supports a single
    frame ``(14,)`` or a batch ``(..., 14)``.

    ``antennas`` may be a length-2 pair (broadcast across a batch) or, for a
    batch of ``N`` frames, an ``(N, 2)`` array of per-frame antenna values.

    Note (plan §5.2 precedence rule): the runtime never reads antenna values
    from the joint array; this parameter exists only for training-parity /
    round-trip completeness. It defaults to neutral ``(0.0, 0.0)``.
    """
    a = np.asarray(arr14, dtype=np.float64)
    if a.shape[-1] != N_JOINTS_14:
        raise ValueError(
            "to_ref16 expects last axis == 14, got shape %r" % (a.shape,)
        )
    ant = np.asarray(antennas, dtype=np.float64)
    if ant.shape[-1] != 2:
        raise ValueError(
            "antennas must have last axis == 2 (left, right), got %r" % (ant.shape,)
        )

    left = a[..., :LEFT_ANTENNA_INDEX_16]           # indices 0..8 (head incl.)
    right = a[..., LEFT_ANTENNA_INDEX_16:]          # indices 9.. (leg block)

    # Broadcast antennas to the batch shape of ``a`` (excluding the joint axis).
    # Keep the last axis via 0:1 slicing so (2,) and (N,2) both broadcast to
    # batch_shape + (1,).
    batch_shape = a.shape[:-1]
    left_ant = np.broadcast_to(ant[..., 0:1], batch_shape + (1,))
    right_ant = np.broadcast_to(ant[..., 1:2], batch_shape + (1,))

    return np.concatenate([left, left_ant, right_ant, right], axis=-1)
