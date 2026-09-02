"""Tests for transform.py (plan §6.3)."""

import numpy as np
import pytest

from open_duck_anim import transform as tf


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
    pose = np.array([0.1, 0.0, 0.2, 0.0])
    base = np.array([0.0, 0.0, 0.1, 0.0])
    joy = np.array([0.0, 0.0, 0.05, 0.0])
    cmd = tf.pose_to_command(pose, base_command=base, joystick_offset=joy)
    assert cmd[2] == pytest.approx(0.1 + 0.2 + 0.05)


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
    pose = np.array([2.0, 0.0, 0.0, 0.0])  # neck_pitch huge
    cmd = tf.pose_to_command(pose)
    assert cmd[0] == pytest.approx(1.1)


def test_bad_shape_raises():
    with pytest.raises(ValueError):
        tf.animation_delta(np.zeros(3))
    with pytest.raises(ValueError):
        tf.clamp_training_range(np.zeros(5))
