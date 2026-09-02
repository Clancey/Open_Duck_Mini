"""Tests for limits.py (plan §6.4, §6.5)."""

import numpy as np
import pytest

from open_duck_anim import limits
from open_duck_anim.clip import DiscreteEvent


def test_joint_clamp():
    lim = limits.JointLimiter(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]))
    out = lim.clamp(np.array([2.0, -5.0]))
    assert np.allclose(out, [1.0, -1.0])


def test_joint_clamp_bad_limits():
    with pytest.raises(ValueError):
        limits.JointLimiter(low=np.array([1.0]), high=np.array([-1.0]))


def test_rate_limit_caps_step_at_dt():
    rl = limits.JointRateLimiter(max_velocity=5.24)
    prev = np.zeros(3)
    target = np.array([1.0, -1.0, 0.05])
    dt = 0.02
    out = rl.limit(prev, target, dt)
    max_step = 5.24 * dt  # 0.1048
    assert out[0] == pytest.approx(max_step)      # capped
    assert out[1] == pytest.approx(-max_step)     # capped
    assert out[2] == pytest.approx(0.05)          # within limit → unchanged


def test_rate_limit_converges_over_ticks():
    rl = limits.JointRateLimiter(max_velocity=5.24)
    prev = np.zeros(1)
    target = np.array([1.0])
    dt = 0.02
    for _ in range(200):
        prev = rl.limit(prev, target, dt)
    assert prev[0] == pytest.approx(1.0, abs=1e-9)


def test_rate_limit_rejects_events_structurally():
    rl = limits.JointRateLimiter()
    ev = np.array([DiscreteEvent(1, "sound", "a.wav")], dtype=object)
    with pytest.raises(TypeError, match="discrete events"):
        rl.limit(np.zeros(1), ev, 0.02)


def test_antenna_slew_and_clamp():
    sl = limits.AntennaSlewLimiter(max_slew=8.0)
    prev = np.array([0.0])
    target = np.array([1.0])
    dt = 0.02
    out = sl.limit(prev, target, dt)
    assert out[0] == pytest.approx(min(8.0 * dt, 1.0))
    # clamps into [-1,1] even with a huge target
    out2 = sl.limit(np.array([0.99]), np.array([5.0]), dt)
    assert out2[0] <= 1.0


def test_antenna_slew_clamps_normalised_joint_rate_does_not():
    # Structural difference, not a numeric-inequality tautology: the antenna slew
    # limiter clamps its output to the normalised [-1,1] range, while the joint
    # rate limiter works in radians and imposes NO such clamp. They are not
    # interchangeable.
    dt = 1.0  # large dt so neither is limited by its per-step velocity cap
    sl = limits.AntennaSlewLimiter(max_slew=limits.DEFAULT_ANTENNA_SLEW)
    ant = sl.limit(np.array([0.0]), np.array([5.0]), dt)
    assert ant[0] == pytest.approx(1.0)  # clamped to normalised max

    rl = limits.JointRateLimiter(max_velocity=100.0)
    jt = rl.limit(np.array([0.0]), np.array([2.5]), dt)
    assert jt[0] == pytest.approx(2.5)  # radians, NOT clamped to 1.0


def test_rate_limit_dt_must_be_positive():
    rl = limits.JointRateLimiter()
    with pytest.raises(ValueError):
        rl.limit(np.zeros(1), np.zeros(1), 0.0)
