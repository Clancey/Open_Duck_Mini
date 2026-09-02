"""Tests for joint_order.py (plan §4.1-C, Appendix A)."""

import numpy as np
import pytest

from open_duck_anim import joint_order as jo


def test_orders_differ_only_by_antennas():
    hw_from_16 = [n for n in jo.JOINT_ORDER_16 if n not in ("left_antenna", "right_antenna")]
    assert hw_from_16 == jo.HW_ORDER_14
    assert jo.JOINT_ORDER_16[9] == "left_antenna"
    assert jo.JOINT_ORDER_16[10] == "right_antenna"


def test_head_index_coincidence():
    # The head block occupies indices 5..8 in BOTH orders — assert, don't assume.
    assert (jo.NECK_PITCH_16, jo.HEAD_PITCH_16, jo.HEAD_YAW_16, jo.HEAD_ROLL_16) == (5, 6, 7, 8)
    assert (jo.NECK_PITCH_14, jo.HEAD_PITCH_14, jo.HEAD_YAW_14, jo.HEAD_ROLL_14) == (5, 6, 7, 8)
    assert jo.NECK_PITCH_16 == jo.NECK_PITCH_14
    assert jo.HEAD_ROLL_16 == jo.HEAD_ROLL_14
    for name in ("neck_pitch", "head_pitch", "head_yaw", "head_roll"):
        assert jo.JOINT_ORDER_16.index(name) == jo.HW_ORDER_14.index(name)


def test_single_frame_roundtrip_preserves_non_antenna():
    a16 = np.arange(16, dtype=float) + 0.123
    hw = jo.to_hw14(a16)
    assert hw.shape == (14,)
    back = jo.to_ref16(hw, antennas=(a16[9], a16[10]))
    assert np.allclose(back, a16)
    # Non-antenna channels survive even with default (zero) antennas.
    back0 = jo.to_ref16(hw)
    keep = [i for i in range(16) if i not in (9, 10)]
    assert np.allclose(back0[keep], a16[keep])
    assert back0[9] == 0.0 and back0[10] == 0.0


def test_batch_roundtrip_preserves_non_antenna():
    rng = np.random.default_rng(0)
    b16 = rng.standard_normal((7, 16))
    hw = jo.to_hw14(b16)
    assert hw.shape == (7, 14)
    back = jo.to_ref16(hw, antennas=b16[:, [9, 10]])
    assert np.allclose(back, b16)


def test_to_hw14_drops_correct_indices():
    a16 = np.arange(16, dtype=float)
    hw = jo.to_hw14(a16)
    assert list(hw) == [i for i in range(16) if i not in (9, 10)]


def test_bad_shapes_raise():
    with pytest.raises(ValueError):
        jo.to_hw14(np.zeros(14))
    with pytest.raises(ValueError):
        jo.to_ref16(np.zeros(16))
    with pytest.raises(ValueError):
        jo.to_ref16(np.zeros(14), antennas=(1.0, 2.0, 3.0))


def test_init_pos_matches_appendix_a():
    assert jo.INIT_POS_14.shape == (14,)
    # spot-check a few from Appendix A
    assert jo.INIT_POS_14[jo.HW_ORDER_14.index("left_knee")] == pytest.approx(1.368)
    assert jo.INIT_POS_14[jo.HW_ORDER_14.index("right_ankle")] == pytest.approx(-0.796)
    assert jo.INIT_POS_14[jo.NECK_PITCH_14] == 0.0


def test_init_pos_is_readonly():
    with pytest.raises(ValueError):
        jo.INIT_POS_14[0] = 1.0
