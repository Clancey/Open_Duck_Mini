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


# The runtime reads antenna values ONLY from show_functions (plan §5.2), so a
# looping/idle clip is silent on the antennas iff these two tracks never change.
# Rest is the neutral normalised value 0.0 (to_normalized(0 rad); see the clip
# calibration), so an idle antenna track must be a flat constant at exactly 0.0.
_ANTENNA_REST_TOL = 1e-9


def _is_idle_or_looping(clip):
    """A clip is a looping / background-idle layer if it loops (``wrap``) or acts
    as a background layer (``priority == 0``). These may run for many minutes
    continuously on a dock, so any antenna motion in one is effectively
    continuous noise — see experiments/animation/clips/README.md."""
    return clip.loop_mode == "wrap" or clip.priority == 0


@pytest.mark.parametrize("path", CLIP_PATHS, ids=[_clip_id(p) for p in CLIP_PATHS])
def test_idle_and_looping_clips_have_no_antenna_motion(path):
    """OWNER DECISION (measured hardware noise): looping / idle clips must NEVER
    move the antennas.

    The antennas are open-loop PWM hobby servos that audibly buzz whenever they
    are driven. A dock idle loop runs for many minutes on a desk, so *any*
    antenna motion in a loop is effectively continuous noise. The owner asked for
    the antennas to be unused in the idle animations, so every looping / idle
    clip (``loop_mode == 'wrap'`` or a background layer at ``priority == 0``) must
    hold both antenna tracks at a flat constant at the neutral rest value (0.0) —
    the runtime then issues no changing antenna command at all.

    Brief triggered reactions (``once`` clips) are intentionally NOT covered: a
    short antenna flick there is a momentary gesture, not sustained noise. Do not
    relax this test to re-enable idle antenna motion.
    """
    clip = load_clip(path)
    if not _is_idle_or_looping(clip):
        pytest.skip("%s is a triggered (non-looping) clip; antennas allowed" % clip.name)
    for side, track in (("antenna_left", clip.show.antenna_l),
                        ("antenna_right", clip.show.antenna_r)):
        arr = np.asarray(track, dtype=np.float64)
        ptp = float(np.ptp(arr)) if arr.size else 0.0
        assert ptp == 0.0, (
            "%s (loop_mode=%r, priority=%d) moves %s: peak-to-peak %.3e over the "
            "clip. Looping/idle clips must never move the antennas (owner decision, "
            "measured hardware noise) — hold the track flat at rest. See "
            "experiments/animation/clips/README.md."
            % (clip.name, clip.loop_mode, clip.priority, side, ptp)
        )
        assert abs(float(arr.flat[0])) <= _ANTENNA_REST_TOL, (
            "%s holds %s at a constant %.4f rather than the neutral rest value 0.0; "
            "idle antenna tracks must rest at 0.0 so the servo is never energised to "
            "an offset." % (clip.name, side, float(arr.flat[0]))
        )
