"""open_duck_anim — numpy-only animation blending core for Open Duck Mini v2.

Phase 1 of the plan in ``docs/animation_system_plan.md``: joint-order conversion,
``.duckanim`` clip IO + validation, the one-way 59-float→``.duckanim`` compiler,
the three-layer blend engine, the absolute→relative head transform, and safety
limiters. numpy-only at runtime (installs on a Raspberry Pi Zero 2W).
"""

from .version import __version__

# Joint ordering (plan §4.1-C, Appendix A)
from .joint_order import (
    JOINT_ORDER_16,
    HW_ORDER_14,
    INIT_POS_14,
    ZERO_POS_14,
    to_hw14,
    to_ref16,
    NECK_PITCH_16,
    HEAD_PITCH_16,
    HEAD_YAW_16,
    HEAD_ROLL_16,
    NECK_PITCH_14,
    HEAD_PITCH_14,
    HEAD_YAW_14,
    HEAD_ROLL_14,
    HEAD_SLICE_16,
    HEAD_SLICE_14,
    LEG_INDICES_16,
    ANTENNA_INDICES_16,
)

# Clip format (plan §5)
from .clip import (
    DuckAnimClip,
    DiscreteEvent,
    ShowOutput,
    AntennaCalibration,
    ClipValidationError,
    load_clip,
    load_clip_json,
    load_clip_npz,
    save_clip_npz,
    clip_from_dict,
    validate_clip_dict,
    LOOP_MODES,
    LAYER_MASKS,
    REQUIRES_MODES,
)

# Compiler (plan §4.1-C, §5)
from .compiler import (
    compile_to_dict,
    compile_to_json_bytes,
    compile_file,
    CompileError,
    COMPILER_VERSION,
)

# Engine (plan §6)
from .blend import (
    Engine,
    Triggers,
    EngineOutput,
    TickShow,
    clamp_blend_times,
    MODE_DOCK,
    MODE_STAND,
    MODE_WALK,
    CTRL_DT,
    T_ALPHA,
    T_BETA,
)

# Transform (plan §6.3). NOTE: ``pose_to_command`` enforces the D13/R16 balance
# envelope by DEFAULT (reviewer E3); ``clamp_training_range`` does NOT — it only
# clamps to the (unsafe) training ranges and must never drive hardware directly
# without the envelope. See ``transform.pose_to_command`` and ``envelope``.
from .transform import (
    pose_to_command,
    animation_delta,
    clamp_training_range,
    HEAD_CHANNELS,
    TRAINING_RANGES,
    NOMINAL_HEAD_POSE,
)

# Limits (plan §6.4, §6.5)
from .limits import (
    JointLimiter,
    JointRateLimiter,
    AntennaSlewLimiter,
    MAX_MOTOR_VELOCITY,
    DEFAULT_ANTENNA_SLEW,
)

# Head safety envelope (plan §6.5, defect D13 / risk R16)
from .envelope import (
    HeadEnvelope,
    clamp_head_envelope,
    DEFAULT_ENVELOPE,
    DEFLECTION_LIMITS,
    DEFLECTION_LOW,
    DEFLECTION_HIGH,
    SLEW_LIMIT,
    COMBINED_L2_BUDGET,
    SAFETY_FRACTION,
    HARDWARE_DERATING,
)

# Leg dock safety envelope (plan §6.2 dock full-body capability)
from .leg_envelope import (
    LegDockEnvelope,
    DEFAULT_LEG_ENVELOPE,
    DERATED_LEG_ENVELOPE,
    DOCK_LEG_HOLD,
    DOCK_LEG_MAX_DEFLECTION,
    DOCK_LEG_DERATING,
    LEG_JNT_LOW,
    LEG_JNT_HIGH,
    LEG_HW_INDICES,
    LEG_NAMES,
)

# Torso posture safety envelope (STAND full-body emotion; UNSWEPT placeholder)
from .torso_envelope import (
    TorsoEnvelope,
    DEFAULT_TORSO_ENVELOPE,
    DERATED_TORSO_ENVELOPE,
    posture_to_command_offsets,
    POSTURE_COMMAND_CHANNELS,
)

# Clip posture channel (STAND full-body emotion)
from .clip import (
    POSTURE_CHANNELS,
    NEUTRAL_POSTURE,
    POSTURE_AUTHORING_BOUNDS,
)

__all__ = [
    "__version__",
    # joint_order
    "JOINT_ORDER_16", "HW_ORDER_14", "INIT_POS_14", "ZERO_POS_14",
    "to_hw14", "to_ref16",
    "NECK_PITCH_16", "HEAD_PITCH_16", "HEAD_YAW_16", "HEAD_ROLL_16",
    "NECK_PITCH_14", "HEAD_PITCH_14", "HEAD_YAW_14", "HEAD_ROLL_14",
    "HEAD_SLICE_16", "HEAD_SLICE_14", "LEG_INDICES_16", "ANTENNA_INDICES_16",
    # clip
    "DuckAnimClip", "DiscreteEvent", "ShowOutput", "AntennaCalibration",
    "ClipValidationError", "load_clip", "load_clip_json", "load_clip_npz", "save_clip_npz",
    "clip_from_dict", "validate_clip_dict",
    "LOOP_MODES", "LAYER_MASKS", "REQUIRES_MODES",
    # compiler
    "compile_to_dict", "compile_to_json_bytes", "compile_file",
    "CompileError", "COMPILER_VERSION",
    # blend
    "Engine", "Triggers", "EngineOutput", "TickShow", "clamp_blend_times",
    "MODE_DOCK", "MODE_STAND", "MODE_WALK", "CTRL_DT", "T_ALPHA", "T_BETA",
    # transform
    "pose_to_command", "animation_delta", "clamp_training_range",
    "HEAD_CHANNELS", "TRAINING_RANGES", "NOMINAL_HEAD_POSE",
    # limits
    "JointLimiter", "JointRateLimiter", "AntennaSlewLimiter",
    "MAX_MOTOR_VELOCITY", "DEFAULT_ANTENNA_SLEW",
    # envelope (D13/R16)
    "HeadEnvelope", "clamp_head_envelope", "DEFAULT_ENVELOPE",
    "DEFLECTION_LIMITS", "DEFLECTION_LOW", "DEFLECTION_HIGH",
    "SLEW_LIMIT", "COMBINED_L2_BUDGET", "SAFETY_FRACTION", "HARDWARE_DERATING",
    # leg dock envelope (§6.2)
    "LegDockEnvelope", "DEFAULT_LEG_ENVELOPE", "DERATED_LEG_ENVELOPE",
    "DOCK_LEG_HOLD", "DOCK_LEG_MAX_DEFLECTION", "DOCK_LEG_DERATING",
    "LEG_JNT_LOW", "LEG_JNT_HIGH", "LEG_HW_INDICES", "LEG_NAMES",
    # torso posture envelope + clip posture channel (STAND full-body emotion)
    "TorsoEnvelope", "DEFAULT_TORSO_ENVELOPE", "DERATED_TORSO_ENVELOPE",
    "posture_to_command_offsets", "POSTURE_COMMAND_CHANNELS",
    "POSTURE_CHANNELS", "NEUTRAL_POSTURE", "POSTURE_AUTHORING_BOUNDS",
]
