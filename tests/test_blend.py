"""Tests for blend.py engine math (plan §6.1-§6.4)."""

import numpy as np
import pytest

import open_duck_anim as oda
from open_duck_anim import blend
from open_duck_anim.blend import Engine, Triggers, clamp_blend_times

from _helpers import make_clip

HEAD_YAW = 2  # index within the 4 head channels (neck, pitch, yaw, roll)


def const_head_clip(value=0.5, loop_mode="wrap", **kw):
    return make_clip(head_const=value, antenna_const_rad=0.3, loop_mode=loop_mode, **kw)


def test_ramp_in_shape_linear():
    # head held constant at 0.5; blend_in 0.1s → offset ramps linearly 0→0.5.
    c = const_head_clip(0.5, loop_mode="wrap", blend_in_s=0.1, blend_out_s=0.1)
    eng = Engine()
    eng.evaluate(0.0, "stand", Triggers(clips=[c]))
    v0 = eng.evaluate(0.0, "stand").head_command_offsets[HEAD_YAW]
    v25 = eng.evaluate(0.025, "stand").head_command_offsets[HEAD_YAW]
    v50 = eng.evaluate(0.05, "stand").head_command_offsets[HEAD_YAW]
    v100 = eng.evaluate(0.10, "stand").head_command_offsets[HEAD_YAW]
    assert v0 == pytest.approx(0.0, abs=1e-9)
    assert v25 == pytest.approx(0.125, abs=1e-6)
    assert v50 == pytest.approx(0.25, abs=1e-6)
    assert v100 == pytest.approx(0.5, abs=1e-6)


def test_ramp_out_once():
    # once clip, duration 0.4, blend_out 0.1 → ramps down to 0 at the end.
    c = const_head_clip(0.5, loop_mode="once", blend_in_s=0.1, blend_out_s=0.1)
    eng = Engine()
    eng.evaluate(0.0, "stand", Triggers(clips=[c]))
    mid = eng.evaluate(0.2, "stand").head_command_offsets[HEAD_YAW]
    near_end = eng.evaluate(0.35, "stand").head_command_offsets[HEAD_YAW]
    assert mid == pytest.approx(0.5, abs=1e-6)
    assert 0.0 < near_end < 0.5


def test_blend_overlap_clamped():
    # blend_in + blend_out > duration → clamp so full weight holds ≥ 1 frame.
    bi, bo = clamp_blend_times(0.3, 0.3, 0.4, 50)
    assert bi + bo <= 0.4 - 1.0 / 50 + 1e-12
    assert bi == pytest.approx(bo)  # symmetric scaling
    # non-overlapping is unchanged
    bi2, bo2 = clamp_blend_times(0.1, 0.1, 0.4, 50)
    assert (bi2, bo2) == (0.1, 0.1)


def test_loop_wrap_repeats():
    c = make_clip(loop_mode="wrap", head_yaw_end=0.5, blend_in_s=0.0, blend_out_s=0.0)
    eng = Engine()
    eng.evaluate(0.0, "stand", Triggers(clips=[c]))
    # duration 0.4; at t=0.0 and t=0.4 (one full loop) phase is identical.
    v_start = eng.evaluate(0.0, "stand").head_command_offsets[HEAD_YAW]
    v_loop = eng.evaluate(0.4, "stand").head_command_offsets[HEAD_YAW]
    assert v_start == pytest.approx(v_loop, abs=1e-6)


def test_loop_clamp_holds_last_frame():
    c = make_clip(loop_mode="clamp", head_yaw_end=0.5, blend_in_s=0.0, blend_out_s=0.0)
    eng = Engine()
    eng.evaluate(0.0, "stand", Triggers(clips=[c]))
    # well past the end → holds the last frame value (~0.5).
    v = eng.evaluate(2.0, "stand").head_command_offsets[HEAD_YAW]
    assert v == pytest.approx(0.5, abs=1e-6)


def test_loop_once_returns_to_background():
    c = make_clip(loop_mode="once", head_yaw_end=0.5, blend_in_s=0.05, blend_out_s=0.05)
    eng = Engine()
    eng.evaluate(0.0, "stand", Triggers(clips=[c]))
    v = eng.evaluate(2.0, "stand").head_command_offsets[HEAD_YAW]
    assert v == pytest.approx(0.0, abs=1e-6)  # background (none) → 0


def test_priority_higher_preempts():
    low = const_head_clip(0.3, loop_mode="wrap", priority=1, blend_in_s=0.0)
    high = const_head_clip(0.8, loop_mode="wrap", priority=10, blend_in_s=0.0)
    eng = Engine()
    eng.evaluate(0.0, "stand", Triggers(clips=[low]))
    v_low = eng.evaluate(0.05, "stand").head_command_offsets[HEAD_YAW]
    assert v_low == pytest.approx(0.3, abs=1e-6)
    # high priority preempts; after its (instant) blend it dominates.
    eng.evaluate(0.05, "stand", Triggers(clips=[high]))
    v_high = eng.evaluate(0.3, "stand").head_command_offsets[HEAD_YAW]
    assert v_high == pytest.approx(0.8, abs=1e-6)


def test_lower_priority_ignored():
    high = const_head_clip(0.8, loop_mode="wrap", priority=10, blend_in_s=0.0)
    low = const_head_clip(0.2, loop_mode="wrap", priority=1, blend_in_s=0.0)
    eng = Engine()
    eng.evaluate(0.0, "stand", Triggers(clips=[high]))
    eng.evaluate(0.1, "stand")  # settle to 0.8
    # lower priority trigger should NOT take ownership
    eng.evaluate(0.1, "stand", Triggers(clips=[low]))
    v = eng.evaluate(0.3, "stand").head_command_offsets[HEAD_YAW]
    assert v == pytest.approx(0.8, abs=1e-6)


def test_preemption_starts_from_current_output_no_snap():
    # A preempting clip must blend from the CURRENT output, not snap to background.
    a = const_head_clip(0.6, loop_mode="wrap", priority=1, blend_in_s=0.1, blend_out_s=0.1)
    b = const_head_clip(-0.4, loop_mode="wrap", priority=5, blend_in_s=0.2, blend_out_s=0.1)
    eng = Engine()
    eng.evaluate(0.0, "stand", Triggers(clips=[a]))
    v_before = eng.evaluate(0.2, "stand").head_command_offsets[HEAD_YAW]
    assert v_before == pytest.approx(0.6, abs=1e-6)
    # preempt at t=0.2 with b (blend_in 0.2s). At the instant of preemption the
    # output must equal the pre-preempt value (no snap toward background/b).
    v_at = eng.evaluate(0.2, "stand", Triggers(clips=[b])).head_command_offsets[HEAD_YAW]
    assert v_at == pytest.approx(0.6, abs=1e-3)
    # Small step later it should move gradually toward b, not jump.
    v_mid = eng.evaluate(0.21, "stand").head_command_offsets[HEAD_YAW]
    assert v_mid < v_before  # heading toward -0.4
    assert v_mid > -0.4      # but not there yet (no snap)
    # After b fully blends in, output reaches b.
    v_after = eng.evaluate(0.6, "stand").head_command_offsets[HEAD_YAW]
    assert v_after == pytest.approx(-0.4, abs=1e-3)


def test_cancel_blends_back_to_background():
    # H1: a cancelled clip fades over at least T_ALPHA (0.35s) even though the
    # authored blend_out_s is short (0.1s) — it must not snap to background.
    c = const_head_clip(0.5, loop_mode="wrap", blend_in_s=0.0, blend_out_s=0.1)
    eng = Engine()
    eng.evaluate(0.0, "stand", Triggers(clips=[c]))
    eng.evaluate(0.1, "stand")  # at 0.5
    eng.evaluate(0.1, "stand", Triggers(cancel=True))
    # 0.15s after cancel the floored 0.35s release is only partway down: still
    # well above background, proving no snap.
    v_mid = eng.evaluate(0.25, "stand").head_command_offsets[HEAD_YAW]
    assert 0.0 < v_mid < 0.5
    # once the full floored release elapses it returns to background (0).
    v = eng.evaluate(0.1 + blend.T_ALPHA + 0.02, "stand").head_command_offsets[HEAD_YAW]
    assert v == pytest.approx(0.0, abs=1e-6)


def test_preempt_zero_blend_out_no_snap():
    # H1: outgoing clip has blend_out_s == 0 (accepted by validation). Preemption
    # must NOT collapse the composite to background in a single tick; the floored
    # release keeps the outgoing contribution alive while the incoming ramps in.
    a = const_head_clip(0.6, loop_mode="wrap", priority=1, blend_in_s=0.0, blend_out_s=0.0)
    b = const_head_clip(-0.4, loop_mode="wrap", priority=5, blend_in_s=0.2, blend_out_s=0.1)
    eng = Engine()
    eng.evaluate(0.0, "stand", Triggers(clips=[a]))
    v_before = eng.evaluate(0.1, "stand").head_command_offsets[HEAD_YAW]
    assert v_before == pytest.approx(0.6, abs=1e-6)
    # preempt: at the instant of preemption output must stay near 0.6 (no snap
    # to 0.0 background, which would be a ~30 rad/s step).
    v_at = eng.evaluate(0.1, "stand", Triggers(clips=[b])).head_command_offsets[HEAD_YAW]
    assert v_at == pytest.approx(0.6, abs=1e-2)
    # one tick later still nowhere near background: bounded step.
    v_next = eng.evaluate(0.12, "stand").head_command_offsets[HEAD_YAW]
    assert abs(v_next - v_before) < 0.2  # << the ~0.6 snap it used to make



def test_dock_outputs_legs_and_head():
    c = const_head_clip(0.4, loop_mode="wrap", blend_in_s=0.0)
    eng = Engine()
    eng.evaluate(0.0, "dock", Triggers(clips=[c]))
    out = eng.evaluate(0.1, "dock")
    assert out.leg_targets is not None and out.leg_targets.shape == (10,)
    assert out.head_targets is not None and out.head_targets.shape == (4,)
    assert out.head_targets[HEAD_YAW] == pytest.approx(0.4, abs=1e-6)
    # legs are held at the nominal dock posture (no motion).
    assert np.allclose(out.leg_targets, blend._DOCK_LEG_HOLD)


def test_stand_mode_no_direct_targets():
    c = const_head_clip(0.4, loop_mode="wrap", blend_in_s=0.0)
    eng = Engine()
    eng.evaluate(0.0, "stand", Triggers(clips=[c]))
    out = eng.evaluate(0.1, "stand")
    assert out.leg_targets is None
    assert out.head_targets is None


def test_bad_mode_raises():
    eng = Engine()
    with pytest.raises(ValueError):
        eng.evaluate(0.0, "flying")


def test_head_offsets_are_unclamped_delta():
    # H2: evaluate() returns an UNCLAMPED delta (added to commands[3:7]
    # downstream). Clamping to the training range is the caller's job via
    # transform.pose_to_command, NOT evaluate()'s.
    from open_duck_anim import transform
    c = const_head_clip(5.0, loop_mode="wrap", blend_in_s=0.0)
    eng = Engine()
    eng.evaluate(0.0, "stand", Triggers(clips=[c]))
    out = eng.evaluate(0.1, "stand")
    # raw delta is passed through unclamped
    assert out.head_command_offsets[HEAD_YAW] == pytest.approx(5.0)
    # the downstream §6.3 clamp is what bounds the ABSOLUTE command (base+delta).
    cmd = transform.clamp_training_range(np.zeros(4) + out.head_command_offsets)
    assert cmd[HEAD_YAW] == pytest.approx(1.5)  # head_yaw training max


def test_head_offset_composes_with_nonzero_base():
    # H2: with a non-zero base_command a legitimate negative offset must survive
    # (the old absolute clamp on the delta truncated -0.5 -> -0.34), and a
    # passing offset must not be double-applied to yield an out-of-range absolute.
    from open_duck_anim import transform
    base = np.array([0.8, 0.0, 0.0, 0.0])  # neck_pitch base near top of range
    delta = np.array([-0.5, 0.0, 0.0, 0.0])
    cmd = transform.clamp_training_range(base + delta)
    # 0.8 + (-0.5) = 0.3, within neck_pitch [-0.34, 1.1] — must NOT be truncated.
    assert cmd[0] == pytest.approx(0.3, abs=1e-9)


def test_dock_head_targets_clamped_to_joint_limits_not_command_range():
    # H2: DOCK head_targets are absolute JOINT targets and must clamp against the
    # configurable joint-limit table, not the command training range.
    wide = (np.array([-3.0, -3.0, -3.0, -3.0]), np.array([3.0, 3.0, 3.0, 3.0]))
    c = const_head_clip(2.0, loop_mode="wrap", blend_in_s=0.0)
    eng = Engine(head_joint_limits=wide)
    eng.evaluate(0.0, "dock", Triggers(clips=[c]))
    out = eng.evaluate(0.1, "dock")
    # 2.0 exceeds the command training range (1.5) but is within the wide joint
    # limits, so it must pass through unclamped — proving joint-limit clamping.
    assert out.head_targets[HEAD_YAW] == pytest.approx(2.0, abs=1e-6)

