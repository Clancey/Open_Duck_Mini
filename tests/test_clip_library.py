"""Library guard: every shipped ``.duckanim`` clip must be safe to ship.

This test loads **every** clip in ``experiments/animation/clips/`` and asserts,
for each one, that it:

  * parses and passes the full ``.duckanim`` schema validation (plan §5/§6.2),
  * is ``layer_mask="head"`` (the architecture: the RL policy owns the legs, a
    head-masked clip is safe in DOCK_DEMO and while walking),
  * genuinely holds the legs constant (all ten leg joints are flat), and
  * lies inside the ×0.5 hardware-derated safety envelope — i.e. the runtime
    envelope would NOT have to clamp it (peak ``||c/L||_2`` under budget and a
    per-frame clamp delta of ~0, allowing only float round-off).

Its purpose is to stop a future broken or over-amplitude clip from silently
shipping. New clips authored via ``experiments/animation/author_clips.py`` land
in the same directory and are covered automatically.
"""

import glob
import json
import os

import numpy as np
import pytest

from open_duck_anim import load_clip
from open_duck_anim.clip import HEAD_SLICE_16, validate_clip_dict
from open_duck_anim.envelope import DEFAULT_ENVELOPE, HARDWARE_DERATING

_HERE = os.path.dirname(__file__)
_CLIPS_DIR = os.path.abspath(
    os.path.join(_HERE, "..", "experiments", "animation", "clips")
)

# JOINT_ORDER_16 leg indices (everything that is NOT head[5:9] or antenna[9:10]).
_LEG_IDX = [0, 1, 2, 3, 4, 11, 12, 13, 14, 15]

# Design-time gate tolerances. The authoring script keeps every clip well inside
# the derated envelope, so the enforced clamp should be a no-op; we allow a
# sliver for float32 round-off from the Blender-recorded clips (~1e-7 rad).
_CLAMP_TOL_RAD = 1e-3
_LEG_HOLD_TOL_RAD = 1e-6


def _clip_paths():
    paths = sorted(glob.glob(os.path.join(_CLIPS_DIR, "*.duckanim")))
    assert paths, "no .duckanim clips found in %s" % _CLIPS_DIR
    return paths


def _clip_id(path):
    return os.path.basename(path)[: -len(".duckanim")]


CLIP_PATHS = _clip_paths()


@pytest.mark.parametrize("path", CLIP_PATHS, ids=[_clip_id(p) for p in CLIP_PATHS])
def test_clip_schema_valid(path):
    """The raw JSON validates against the full clip schema, and loads."""
    with open(path) as f:
        raw = json.load(f)
    validate_clip_dict(raw)          # raises ClipValidationError on any violation
    clip = load_clip(path)           # loader re-validates + builds the runtime obj
    assert clip.frame_count == len(clip.joints)
    assert clip.blend_in_s + clip.blend_out_s <= clip.duration_s + 1e-9


@pytest.mark.parametrize("path", CLIP_PATHS, ids=[_clip_id(p) for p in CLIP_PATHS])
def test_clip_is_head_masked(path):
    """Head-only: mask is ``head`` and every leg joint is held constant."""
    clip = load_clip(path)
    assert clip.layer_mask == "head", (
        "%s is not head-masked (mask=%r)" % (clip.name, clip.layer_mask)
    )
    leg = clip.joints[:, _LEG_IDX]
    leg_ptp = float(np.max(np.ptp(leg, axis=0))) if len(leg) else 0.0
    assert leg_ptp <= _LEG_HOLD_TOL_RAD, (
        "%s moves the legs (max leg ptp %.3e rad); clips must hold the legs"
        % (clip.name, leg_ptp)
    )


@pytest.mark.parametrize("path", CLIP_PATHS, ids=[_clip_id(p) for p in CLIP_PATHS])
def test_clip_within_derated_envelope(path):
    """The head motion sits inside the ×0.5 hardware-derated safety envelope, so
    the runtime clamp would be a no-op (nothing shipped clamped)."""
    clip = load_clip(path)
    env = DEFAULT_ENVELOPE.derated(HARDWARE_DERATING)
    head = clip.joints[:, HEAD_SLICE_16]
    prev = None
    max_clamp_delta = 0.0
    max_l2 = 0.0
    dt = 1.0 / clip.fps
    for row in head:
        clamped = env.clamp(np.asarray(row, dtype=np.float64),
                            prev_command_head=prev, dt=dt)
        max_clamp_delta = max(max_clamp_delta,
                              float(np.max(np.abs(clamped - row))))
        max_l2 = max(max_l2, float(np.sqrt(np.sum((row / env._L) ** 2))))
        prev = clamped
    assert max_l2 <= env.l2_budget + 1e-9, (
        "%s exceeds the derated combined budget: ||c/L||=%.3f > %.3f"
        % (clip.name, max_l2, env.l2_budget)
    )
    assert max_clamp_delta <= _CLAMP_TOL_RAD, (
        "%s would be clamped by the derated envelope (max delta %.3e rad); "
        "re-author to smaller amplitude rather than shipping it clamped"
        % (clip.name, max_clamp_delta)
    )
