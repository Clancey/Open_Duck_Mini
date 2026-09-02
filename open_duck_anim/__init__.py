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

# Transform (plan §6.3)
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
]
