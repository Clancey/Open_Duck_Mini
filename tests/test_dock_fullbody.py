"""Dock-only full-body clip support (plan §6.2 dock capability).

Covers the architectural extension that lets a clip animate the legs, but ONLY
when docked:

* compile-time: ``layer_mask="full_body"`` is accepted with
  ``requires_mode="dock"`` and rejected with any / stand / walk;
* runtime: the engine refuses to start a full-body clip outside DOCK and
  controlled-aborts one if the mode leaves DOCK mid-clip (never a snap);
* the animated leg targets are clamped to the conservative dock leg envelope and
  rate-limited at ``max_motor_velocity``;
* existing head-masked clips are completely unaffected.
"""

import os
import sys
import json

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from _helpers import make_clip, make_meta, make_source_text  # noqa: E402

from open_duck_anim import compiler, clip as clipmod  # noqa: E402
from open_duck_anim.clip import ClipValidationError  # noqa: E402
from open_duck_anim.blend import Engine, Triggers, MODE_DOCK, MODE_STAND, MODE_WALK  # noqa: E402
from open_duck_anim.leg_envelope import (  # noqa: E402
    LegDockEnvelope,
    DERATED_LEG_ENVELOPE,
    DEFAULT_LEG_ENVELOPE,
    DOCK_LEG_HOLD,
    DOCK_LEG_MAX_DEFLECTION,
    LEG_NAMES,
)
from open_duck_anim.limits import MAX_MOTOR_VELOCITY  # noqa: E402

CTRL_DT = 0.02
REPO_CLIPS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "experiments", "animation", "clips"
)


# ---------------------------------------------------------------------------
# 1. Compile-time channel legality
# ---------------------------------------------------------------------------
def test_full_body_accepted_only_with_dock():
    c = make_clip(move_legs=True, layer_mask="full_body", requires_mode="dock")
    assert c.layer_mask == "full_body"
    assert c.requires_mode == "dock"


@pytest.mark.parametrize("mode", ["any", "stand", "walk"])
def test_full_body_rejected_outside_dock_at_compile(mode):
    with pytest.raises(ClipValidationError) as ei:
        make_clip(move_legs=True, layer_mask="full_body", requires_mode=mode)
    msg = str(ei.value)
    assert "dock" in msg
    assert "legs" in msg


def test_full_body_declared_but_static_legs_is_still_rejected_outside_dock():
    # Even a full-body clip whose legs happen not to move is rejected outside
    # dock: the capability is about the DECLARED channel, not the data (a static
    # clip could be swapped for a moving one under the same mask).
    with pytest.raises(ClipValidationError):
        make_clip(move_legs=False, layer_mask="full_body", requires_mode="stand")


def test_head_mask_with_leg_motion_still_rejected():
    # The pre-existing guarantee is untouched: a head-masked clip that moves the
    # legs is rejected (legs must be neutral), in every mode including dock.
    for mode in ("any", "dock", "stand", "walk"):
        with pytest.raises(ClipValidationError):
            make_clip(move_legs=True, layer_mask="head", requires_mode=mode)


def test_head_masked_clip_unchanged_compiles_all_modes():
    for mode in ("any", "dock", "stand", "walk"):
        c = make_clip(move_legs=False, layer_mask="head", requires_mode=mode)
        assert c.layer_mask == "head"


# ---------------------------------------------------------------------------
# 2. Runtime capability gate
# ---------------------------------------------------------------------------
def _dock_fullbody_clip(**kw):
    return make_clip(move_legs=True, layer_mask="full_body", requires_mode="dock", **kw)


def test_engine_plays_fullbody_in_dock():
    c = _dock_fullbody_clip()
    eng = Engine()
    out = eng.evaluate(0.0, MODE_DOCK, Triggers(clips=[c]))
    # advance to full weight
    for k in range(1, 12):
        out = eng.evaluate(k * CTRL_DT, MODE_DOCK)
    assert out.leg_targets is not None
    assert out.leg_targets.shape == (10,)
    assert out.head_targets is not None
    assert eng.dropped_fullbody_triggers == 0


@pytest.mark.parametrize("mode", [MODE_STAND, MODE_WALK])
def test_engine_refuses_fullbody_trigger_outside_dock(mode):
    c = _dock_fullbody_clip()
    eng = Engine()
    out = eng.evaluate(0.0, mode, Triggers(clips=[c]))
    # Not started: no leg targets (policy owns legs outside dock), counted.
    assert out.leg_targets is None
    assert eng.dropped_fullbody_triggers == 1
    # The clip never became an owner: subsequent ticks stay refused-idle.
    out = eng.evaluate(CTRL_DT, mode)
    assert out.leg_targets is None
    # And a head-command path that is just the (empty) background — no motion.
    assert np.allclose(out.head_command_offsets, 0.0, atol=1e-6)


def test_head_clip_not_treated_as_fullbody():
    c = make_clip(layer_mask="head", requires_mode="any")
    eng = Engine()
    eng.evaluate(0.0, MODE_STAND, Triggers(clips=[c]))
    for k in range(1, 6):
        eng.evaluate(k * CTRL_DT, MODE_STAND)
    assert eng.dropped_fullbody_triggers == 0
    assert eng.fullbody_mode_aborts == 0


# ---------------------------------------------------------------------------
# 3. Mode-transition-mid-clip degrades safely (controlled abort, no snap)
# ---------------------------------------------------------------------------
def test_mode_transition_mid_clip_controlled_abort():
    c = _dock_fullbody_clip(n_frames=100)
    eng = Engine()
    eng.evaluate(0.0, MODE_DOCK, Triggers(clips=[c]))
    for k in range(1, 10):
        out = eng.evaluate(k * CTRL_DT, MODE_DOCK)
    assert out.leg_targets is not None  # playing legs while docked

    # Mode leaves DOCK while the clip is still playing.
    prev_legs = out.leg_targets.copy()
    out2 = eng.evaluate(10 * CTRL_DT, MODE_STAND)
    # Legs immediately stop being emitted (returned to the policy) — no snap
    # because the engine emits None, not a jumped target.
    assert out2.leg_targets is None
    assert eng.fullbody_mode_aborts == 1
    # The clip is now releasing; the head path is bounded by the envelope (no
    # exception, finite output).
    assert np.all(np.isfinite(out2.head_command_offsets))

    # Continue in STAND: it must not re-arm the legs and must not re-abort.
    for k in range(11, 40):
        o = eng.evaluate(k * CTRL_DT, MODE_STAND)
        assert o.leg_targets is None
    assert eng.fullbody_mode_aborts == 1
    _ = prev_legs  # (captured to document there was leg motion before the abort)


def test_mode_transition_then_back_to_dock_no_snap():
    c = _dock_fullbody_clip(n_frames=100)
    eng = Engine()
    eng.evaluate(0.0, MODE_DOCK, Triggers(clips=[c]))
    for k in range(1, 8):
        eng.evaluate(k * CTRL_DT, MODE_STAND if False else MODE_DOCK)
    # leave dock
    eng.evaluate(8 * CTRL_DT, MODE_STAND)
    # come back to dock: legs re-seed from the hold (first emitted target within
    # one rate step of the hold), never a jump to a stale target.
    out = eng.evaluate(9 * CTRL_DT, MODE_DOCK)
    assert out.leg_targets is not None
    step = np.max(np.abs(out.leg_targets - DOCK_LEG_HOLD))
    assert step <= MAX_MOTOR_VELOCITY * CTRL_DT + 1e-9


# ---------------------------------------------------------------------------
# 4. Leg clamping and rate limiting
# ---------------------------------------------------------------------------
def _source_with_leg_step(n_frames, joint_idx, before, after, step_frame):
    """Synthetic source: one leg joint holds ``before`` then jumps to ``after``."""
    frames = []
    for i in range(n_frames):
        fr = [0.0] * 59
        jp = [0.0] * 16
        # keep other legs at their hold so only one channel is exercised
        for name, idx16 in zip(
            LEG_NAMES,
            (0, 1, 2, 3, 4, 11, 12, 13, 14, 15),
        ):
            jp[idx16] = float(DOCK_LEG_HOLD[LEG_NAMES.index(name)])
        jp[joint_idx] = before if i < step_frame else after
        fr[7:23] = jp
        frames.append(fr)
    return json.dumps({"FPS": 50, "Frames": frames})


def test_leg_targets_clamped_to_derated_envelope():
    # A full-body clip that drives left_knee (idx 3) far past its derated
    # deflection cap must be clamped by the engine's default (derated) envelope.
    hold_knee = float(DOCK_LEG_HOLD[LEG_NAMES.index("left_knee")])
    src = _source_with_leg_step(30, 3, hold_knee, hold_knee + 0.6, step_frame=0)
    meta = make_meta(layer_mask="full_body", requires_mode="dock", blend_in_s=0.0)
    c = clipmod.clip_from_dict(compiler.compile_to_dict(src, meta))
    eng = Engine()  # default DERATED_LEG_ENVELOPE
    out = eng.evaluate(0.0, MODE_DOCK, Triggers(clips=[c]))
    for k in range(1, 40):
        out = eng.evaluate(k * CTRL_DT, MODE_DOCK)
    lo = DERATED_LEG_ENVELOPE.eff_low
    hi = DERATED_LEG_ENVELOPE.eff_high
    assert np.all(out.leg_targets >= lo - 1e-9)
    assert np.all(out.leg_targets <= hi + 1e-9)
    # The knee is pinned at its clamped ceiling, not the authored 0.6 excursion.
    knee = out.leg_targets[LEG_NAMES.index("left_knee")]
    assert knee <= hi[LEG_NAMES.index("left_knee")] + 1e-9
    assert knee < hold_knee + 0.6 - 0.1  # clearly clamped


def test_leg_targets_rate_limited():
    # With a permissive envelope (so the clamp does not hide the effect), a sharp
    # leg jump is limited to max_motor_velocity * dt per tick.
    hold_yaw = float(DOCK_LEG_HOLD[LEG_NAMES.index("left_hip_yaw")])
    src = _source_with_leg_step(30, 0, hold_yaw, hold_yaw + 0.4, step_frame=3)
    meta = make_meta(layer_mask="full_body", requires_mode="dock", blend_in_s=0.0)
    c = clipmod.clip_from_dict(compiler.compile_to_dict(src, meta))
    big = LegDockEnvelope(max_deflection=np.full(10, 1.5))  # clamp won't bite
    eng = Engine(leg_envelope=big)
    prev = None
    budget = MAX_MOTOR_VELOCITY * CTRL_DT + 1e-9
    for k in range(0, 30):
        trig = Triggers(clips=[c]) if k == 0 else None
        out = eng.evaluate(k * CTRL_DT, MODE_DOCK, trig)
        if prev is not None:
            step = np.max(np.abs(out.leg_targets - prev))
            assert step <= budget, (k, step, budget)
        prev = out.leg_targets.copy()


# ---------------------------------------------------------------------------
# 5. Head-masked clips completely unaffected in DOCK
# ---------------------------------------------------------------------------
def test_head_masked_clip_holds_legs_in_dock():
    c = make_clip(layer_mask="head", requires_mode="any")
    eng = Engine()
    out = eng.evaluate(0.0, MODE_DOCK, Triggers(clips=[c]))
    for k in range(1, 10):
        out = eng.evaluate(k * CTRL_DT, MODE_DOCK)
    # A head clip never moves the legs: the dock leg targets equal the hold.
    assert np.allclose(out.leg_targets, DOCK_LEG_HOLD, atol=1e-9)
    assert out.head_targets is not None


def test_reset_clears_leg_state():
    c = _dock_fullbody_clip()
    eng = Engine()
    eng.evaluate(0.0, MODE_DOCK, Triggers(clips=[c]))
    eng.evaluate(CTRL_DT, MODE_DOCK)
    assert eng._prev_leg_targets is not None
    eng.reset()
    assert eng._prev_leg_targets is None
    assert eng._last_leg_t is None


# ---------------------------------------------------------------------------
# 6. The shipped deliverable: dock_wiggle.duckanim
# ---------------------------------------------------------------------------
def test_shipped_dock_wiggle_is_dock_only_full_body():
    path = os.path.join(REPO_CLIPS_DIR, "dock_wiggle.duckanim")
    if not os.path.exists(path):
        pytest.skip("dock_wiggle.duckanim not built")
    c = clipmod.load_clip(path)
    assert c.layer_mask == "full_body"
    assert c.requires_mode == "dock"
    assert c.loop_mode == "once"  # never a loop (not idle-safe)
    assert c.priority > 0


def test_shipped_dock_wiggle_stays_inside_derated_envelope_in_dock():
    path = os.path.join(REPO_CLIPS_DIR, "dock_wiggle.duckanim")
    if not os.path.exists(path):
        pytest.skip("dock_wiggle.duckanim not built")
    c = clipmod.load_clip(path)
    eng = Engine()  # default derated envelopes
    lo, hi = DERATED_LEG_ENVELOPE.eff_low, DERATED_LEG_ENVELOPE.eff_high
    prev = None
    budget = MAX_MOTOR_VELOCITY * CTRL_DT + 1e-9
    n = c.frame_count + 30
    for k in range(n):
        trig = Triggers(clips=[c]) if k == 0 else None
        out = eng.evaluate(k * CTRL_DT, MODE_DOCK, trig)
        assert out.leg_targets is not None
        # authored inside the derated box → clamp is a no-op (nothing ships clamped)
        assert np.all(out.leg_targets >= lo - 1e-9)
        assert np.all(out.leg_targets <= hi + 1e-9)
        if prev is not None:
            assert np.max(np.abs(out.leg_targets - prev)) <= budget
        prev = out.leg_targets.copy()


def test_shipped_dock_wiggle_refused_outside_dock():
    path = os.path.join(REPO_CLIPS_DIR, "dock_wiggle.duckanim")
    if not os.path.exists(path):
        pytest.skip("dock_wiggle.duckanim not built")
    c = clipmod.load_clip(path)
    for mode in (MODE_STAND, MODE_WALK):
        eng = Engine()
        out = eng.evaluate(0.0, mode, Triggers(clips=[c]))
        assert out.leg_targets is None
        assert eng.dropped_fullbody_triggers == 1
