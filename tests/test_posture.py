"""Tests for the STAND full-body posture path (torso height/orientation command).

Covers:
* the clip-format ``posture`` channel: parse, default-neutral, validation
  (shape/finiteness/authoring-bounds), the requires_mode capability rule, and
  round-trip through the compiler and the npz container;
* :class:`TorsoEnvelope` enforcement (deflection clamp, per-channel slew LAST,
  combined L2 budget, non-finite repair, unbounded bypass, derating);
* the engine emitting ``posture_command_offsets`` ONLY in STAND, blending a
  mood's posture in over ``T_alpha`` and easing it back to neutral under a
  neutral clip, and freezing/dropping the slew reference outside STAND.
"""

import json
import os

import numpy as np
import pytest

import open_duck_anim as oda
from open_duck_anim import clip as clipmod
from open_duck_anim import compiler
from open_duck_anim.blend import Engine, Triggers, MODE_STAND, MODE_WALK, MODE_DOCK, CTRL_DT
from open_duck_anim.torso_envelope import (
    TorsoEnvelope,
    DEFAULT_TORSO_ENVELOPE,
    DEFLECTION_LOW,
    DEFLECTION_HIGH,
    SLEW_LIMIT_VEC,
    posture_to_command_offsets,
    POSTURE_COMMAND_CHANNELS,
)
from open_duck_anim.clip import (
    ClipValidationError,
    POSTURE_CHANNELS,
    POSTURE_AUTHORING_BOUNDS,
)

from _helpers import make_clip


# --- clip format: posture channel ------------------------------------------
def test_posture_defaults_to_neutral():
    c = make_clip()
    assert c.posture.shape == (3,)
    assert np.all(c.posture == 0.0)
    assert c.has_posture is False


def test_posture_parsed_and_flagged():
    c = make_clip(posture=[-0.02, 0.15, 0.0], requires_mode="stand")
    np.testing.assert_allclose(c.posture, [-0.02, 0.15, 0.0])
    assert c.has_posture is True


def test_posture_requires_stand_or_any():
    # non-neutral posture on a walk clip is rejected
    with pytest.raises(ClipValidationError, match="requires_mode in"):
        make_clip(posture=[-0.02, 0.0, 0.0], requires_mode="walk")
    # dock likewise (body animated via legs there)
    with pytest.raises(ClipValidationError, match="requires_mode in"):
        make_clip(posture=[-0.02, 0.0, 0.0], requires_mode="dock", layer_mask="head")
    # stand and any are fine
    make_clip(posture=[-0.02, 0.0, 0.0], requires_mode="stand")
    make_clip(posture=[-0.02, 0.0, 0.0], requires_mode="any")


def test_neutral_posture_allowed_in_any_mode():
    # an explicit all-zero posture must not trip the requires_mode rule
    make_clip(posture=[0.0, 0.0, 0.0], requires_mode="walk")


def test_posture_out_of_authoring_bounds_rejected():
    hi = POSTURE_AUTHORING_BOUNDS["torso_height_delta_m"][1]
    with pytest.raises(ClipValidationError, match="authoring bounds"):
        make_clip(posture=[hi + 0.01, 0.0, 0.0], requires_mode="stand")


def test_posture_bad_shape_rejected():
    with pytest.raises(ClipValidationError, match="length-3"):
        make_clip(posture=[0.0, 0.0], requires_mode="stand")


def test_posture_nonfinite_rejected_by_validator():
    d = compiler.compile_to_dict(None if False else _src(), _meta(posture=[0.0, 0.0, 0.0]))
    d["posture"] = [float("nan"), 0.0, 0.0]
    with pytest.raises(ClipValidationError, match="non-finite"):
        clipmod.validate_clip_dict(d)


def test_posture_roundtrip_npz(tmp_path):
    c_in = make_clip(posture=[-0.018, 0.1, -0.03], requires_mode="stand")
    d = compiler.compile_to_dict(_src(), _meta(posture=[-0.018, 0.1, -0.03], requires_mode="stand"))
    p = os.path.join(tmp_path, "m.npz")
    clipmod.save_clip_npz(p, d)
    c_out = clipmod.load_clip(p)
    np.testing.assert_allclose(c_out.posture, c_in.posture)


# --- posture_to_command_offsets --------------------------------------------
def test_posture_to_command_offsets_mapping():
    off = posture_to_command_offsets([0.02, 0.3, -0.1])
    assert off[0] == pytest.approx(0.02)              # height passthrough
    assert off[1] == pytest.approx(np.sin(0.3))       # grav_x = sin(pitch)
    assert off[2] == pytest.approx(-np.sin(-0.1))     # grav_y = -sin(roll)
    # neutral maps to neutral
    np.testing.assert_allclose(posture_to_command_offsets([0, 0, 0]), [0, 0, 0])


# --- TorsoEnvelope ----------------------------------------------------------
def test_torso_envelope_deflection_clamp():
    env = TorsoEnvelope()
    over = DEFLECTION_HIGH * 2.0
    out = env.clamp(over)
    assert np.all(out <= DEFLECTION_HIGH + 1e-12)
    under = DEFLECTION_LOW * 2.0
    out = env.clamp(under)
    assert np.all(out >= DEFLECTION_LOW - 1e-12)


def test_torso_envelope_within_passes_unchanged():
    env = TorsoEnvelope()
    c = np.array([0.01, 0.05, -0.02])
    out = env.clamp(c)  # no prev → no slew; inside box; inside budget
    np.testing.assert_allclose(out, c)


def test_torso_envelope_slew_last_per_channel():
    env = TorsoEnvelope()
    prev = np.zeros(3)
    target = DEFLECTION_HIGH.copy()
    out = env.clamp(target, prev_command_torso=prev, dt=CTRL_DT)
    step = np.abs(out - prev)
    assert np.all(step <= SLEW_LIMIT_VEC * CTRL_DT + 1e-12)


def test_torso_envelope_nonfinite_repaired_and_flagged():
    env = TorsoEnvelope()
    out, fault = env.clamp([np.nan, 0.05, np.inf], return_fault=True)
    assert fault is True
    assert np.all(np.isfinite(out))


def test_torso_envelope_unbounded_is_passthrough():
    env = TorsoEnvelope.unbounded()
    huge = np.array([100.0, -100.0, 50.0])
    np.testing.assert_allclose(env.clamp(huge), huge)


def test_torso_envelope_derated_scales_box():
    env = TorsoEnvelope().derated(0.5)
    np.testing.assert_allclose(env.low, DEFLECTION_LOW * 0.5)
    np.testing.assert_allclose(env.high, DEFLECTION_HIGH * 0.5)


def test_torso_envelope_combined_budget_binds():
    # a command inside every per-channel box but with ||c/L|| > budget is scaled.
    env = TorsoEnvelope(l2_budget=0.5)
    c = np.minimum(np.abs(DEFLECTION_LOW), DEFLECTION_HIGH) * 0.5  # ||c/L|| = 0.5*sqrt(3)
    out = env.clamp(c)
    norm = np.sqrt(np.sum((out / np.minimum(np.abs(DEFLECTION_LOW), DEFLECTION_HIGH)) ** 2))
    assert norm <= 0.5 + 1e-9


# --- engine wiring ----------------------------------------------------------
def _stand_mood(pitch=0.15, height=-0.018):
    """A sustained head-mask mood clip carrying a sag posture, playable in stand."""
    return make_clip(
        loop_mode="wrap", blend_in_s=0.0, blend_out_s=0.0,
        priority=5, requires_mode="stand",
        posture=[height, pitch, 0.0], move_head=False,
    )


def test_engine_emits_posture_only_in_stand():
    eng = Engine()
    mood = _stand_mood()
    out = eng.evaluate(0.0, MODE_STAND, Triggers(clips=[mood]))
    assert out.posture_command_offsets is not None
    assert out.posture_command_offsets.shape == (3,)
    # walk: deployed policy has no torso command → None
    eng2 = Engine()
    m2 = make_clip(loop_mode="wrap", requires_mode="any", posture=[0.0, 0.0, 0.0])
    out2 = eng2.evaluate(0.0, MODE_WALK, Triggers(clips=[m2]))
    assert out2.posture_command_offsets is None
    # dock: body via legs → None
    out3 = Engine().evaluate(0.0, MODE_DOCK)
    assert out3.posture_command_offsets is None


def test_engine_posture_blends_in_over_t_alpha():
    eng = Engine(torso_envelope=TorsoEnvelope.unbounded())
    mood = make_clip(
        loop_mode="wrap", blend_in_s=0.2, blend_out_s=0.0,
        priority=5, requires_mode="stand",
        posture=[-0.02, 0.2, 0.0], move_head=False,
    )
    # tick 0: just triggered, body weight ramps from 0 → ~neutral posture
    o0 = eng.evaluate(0.0, MODE_STAND, Triggers(clips=[mood]))
    assert abs(o0.posture_command_offsets[1]) < 1e-9  # w_in(0)=0 → still neutral
    # a tick partway through the blend-in: partial posture
    o_mid = eng.evaluate(0.1, MODE_STAND)
    # after well past the blend-in: full weight → posture near command target
    t = 0.1
    last = o_mid
    while t < 1.0:
        t += CTRL_DT
        last = eng.evaluate(t, MODE_STAND)
    target = posture_to_command_offsets([-0.02, 0.2, 0.0])
    np.testing.assert_allclose(last.posture_command_offsets, target, atol=1e-6)
    # grew monotonically toward target on grav_x (magnitude increased through blend)
    assert abs(last.posture_command_offsets[1]) > abs(o_mid.posture_command_offsets[1]) > 0.0


def test_engine_posture_eases_back_to_neutral_under_neutral_clip():
    eng = Engine(torso_envelope=TorsoEnvelope.unbounded())
    mood = _stand_mood(pitch=0.2, height=-0.02)
    t = 0.0
    eng.evaluate(t, MODE_STAND, Triggers(clips=[mood]))
    while t < 1.0:
        t += CTRL_DT
        eng.evaluate(t, MODE_STAND)
    # preempt with a neutral-posture head clip at higher priority
    neutral = make_clip(loop_mode="wrap", requires_mode="stand", priority=9,
                        posture=[0.0, 0.0, 0.0], move_head=False)
    eng.evaluate(t, MODE_STAND, Triggers(clips=[neutral]))
    while t < 2.0:
        t += CTRL_DT
        out = eng.evaluate(t, MODE_STAND)
    np.testing.assert_allclose(out.posture_command_offsets, [0.0, 0.0, 0.0], atol=1e-6)


def test_engine_posture_default_envelope_bounds_output():
    # with the enforcing default envelope, a large authored sag is clamped to box
    eng = Engine()  # DEFAULT_TORSO_ENVELOPE
    big = make_clip(loop_mode="wrap", requires_mode="stand",
                    posture=[-0.05, 0.35, 0.0], move_head=False)
    t = 0.0
    eng.evaluate(t, MODE_STAND, Triggers(clips=[big]))
    while t < 1.5:
        t += CTRL_DT
        out = eng.evaluate(t, MODE_STAND)
    assert np.all(out.posture_command_offsets >= DEFLECTION_LOW - 1e-9)
    assert np.all(out.posture_command_offsets <= DEFLECTION_HIGH + 1e-9)


# --- tiny local source/meta builders (avoid moving-head noise in posture math)
def _src():
    frames = []
    for _ in range(10):
        fr = [0.0] * 59
        jp = [0.0] * 16
        for k, v in {2: -0.63, 3: 1.368, 4: -0.784, 13: 0.635, 14: 1.379, 15: -0.796}.items():
            jp[k] = v
        fr[7:23] = jp
        frames.append(fr)
    return json.dumps({"FPS": 50, "Frames": frames})


def _meta(**kw):
    m = {
        "name": "m", "loop_mode": "wrap",
        "blend_in_s": 0.0, "blend_out_s": 0.0,
        "show_blend_in_s": 0.0, "show_blend_out_s": 0.0,
        "layer_mask": "head", "priority": 5, "requires_mode": "stand",
        "source_blend": "t.blend",
        "antenna_calibration": {
            "left": {"sign": 1, "rad_min": -0.6, "rad_max": 0.6},
            "right": {"sign": -1, "rad_min": -0.6, "rad_max": 0.6},
        },
    }
    m.update(kw)
    return m
