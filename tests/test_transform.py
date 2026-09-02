"""Tests for transform.py (plan §6.3)."""

import numpy as np
import pytest

from open_duck_anim import transform as tf
from open_duck_anim.envelope import HeadEnvelope, COMBINED_L2_BUDGET


def test_animation_delta():
    pose = np.array([0.1, -0.2, 0.5, 0.0])
    nominal = np.array([0.0, 0.0, 0.0, 0.0])
    d = tf.animation_delta(pose, nominal)
    assert np.allclose(d, pose)


def test_delta_with_nonzero_nominal():
    pose = np.array([0.3, 0.1, 0.0, 0.2])
    nominal = np.array([0.1, 0.1, 0.0, 0.0])
    d = tf.animation_delta(pose, nominal)
    assert np.allclose(d, [0.2, 0.0, 0.0, 0.2])


def test_pose_to_command_additive():
    # With the envelope DELIBERATELY disabled, the §6.3 transform is pure
    # additive composition clamped to the training range.
    pose = np.array([0.1, 0.0, 0.2, 0.0])
    base = np.array([0.0, 0.0, 0.1, 0.0])
    joy = np.array([0.0, 0.0, 0.05, 0.0])
    cmd = tf.pose_to_command(pose, base_command=base, joystick_offset=joy,
                             head_envelope=HeadEnvelope.unbounded())
    assert cmd[2] == pytest.approx(0.1 + 0.2 + 0.05)


def test_pose_to_command_enforces_envelope_by_default():
    # SAFETY DEFAULT (reviewer E3): pose_to_command routes through the D13
    # envelope unless unbounded() is passed. This multi-axis command is within
    # the (wide, iteration-3) training ranges but over the combined L2 budget,
    # so the default enforces it down.
    pose = np.array([0.3, 0.0, 1.4, 0.0])
    cmd = tf.pose_to_command(pose)
    # command=[0.3,0,1.4,0]; L=min(|low|,high)=[0.34,0.78,1.5,0.5]
    # ||c/L||2 = sqrt((0.3/0.34)^2+(1.4/1.5)^2)=1.284 > 1.0 (=COMBINED_L2_BUDGET)
    # → uniform scale COMBINED_L2_BUDGET/1.284 → head_yaw 1.4*scale.
    norm = np.sqrt((0.3 / 0.34) ** 2 + (1.4 / 1.5) ** 2)
    assert cmd[2] == pytest.approx(1.4 * COMBINED_L2_BUDGET / norm, abs=1e-3)
    assert abs(cmd[2]) < 1.4  # strictly tighter than the raw training clamp


def test_clamp_at_training_boundaries():
    # each channel forced beyond its range → clamps to the boundary.
    big = np.array([10.0, 10.0, 10.0, 10.0])
    cmd = tf.clamp_training_range(big)
    assert cmd[0] == pytest.approx(1.1)    # neck_pitch high
    assert cmd[1] == pytest.approx(0.78)   # head_pitch high
    assert cmd[2] == pytest.approx(1.5)    # head_yaw high
    assert cmd[3] == pytest.approx(0.5)    # head_roll high
    small = np.array([-10.0, -10.0, -10.0, -10.0])
    cmd = tf.clamp_training_range(small)
    assert cmd[0] == pytest.approx(-0.34)
    assert cmd[1] == pytest.approx(-0.78)
    assert cmd[2] == pytest.approx(-1.5)
    assert cmd[3] == pytest.approx(-0.5)


def test_within_range_unchanged():
    v = np.array([0.5, 0.3, -0.4, 0.1])
    assert np.allclose(tf.clamp_training_range(v), v)


def test_pose_to_command_clamps():
    # neck_pitch huge → training clamp (+1.1) THEN envelope (+0.355 high side).
    pose = np.array([2.0, 0.0, 0.0, 0.0])  # neck_pitch huge
    cmd = tf.pose_to_command(pose)
    # training clamp → 1.1, then envelope: neck clamps to +0.355; the lone-axis
    # norm 0.355/0.34 = 1.044 > 1.0 → scaled to COMBINED_L2_BUDGET*L_neck = 1.0*0.34.
    assert cmd[0] == pytest.approx(COMBINED_L2_BUDGET * 0.34, abs=1e-6)
    assert cmd[0] < 0.355  # far below the unsafe 1.1 training max
    # with the envelope disabled it falls back to the raw training clamp.
    raw = tf.pose_to_command(pose, head_envelope=HeadEnvelope.unbounded())
    assert raw[0] == pytest.approx(1.1)


def test_bad_shape_raises():
    with pytest.raises(ValueError):
        tf.animation_delta(np.zeros(3))
    with pytest.raises(ValueError):
        tf.clamp_training_range(np.zeros(5))
