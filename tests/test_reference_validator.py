"""Tests for the 59-float reference-motion kinematic-consistency validator.

The headline acceptance test is :func:`test_known_bad_standing_wiggle_fails`: the
validator must **reject** the exact reference that silently cost four GPU training
runs, and the failure message must name the transposed angular-velocity axes and
the zeroed linear-velocity channel. A validator that passes the known-bad file is
worthless.
"""

import json
import os

import numpy as np
import pytest

from open_duck_anim import reference_validator as rv
from open_duck_anim.reference_validator import (
    validate_reference,
    validate_reference_file,
    angular_velocity_from_quats,
    quat_mul,
    quat_conj,
    quat_to_rotvec,
)
from open_duck_anim.joint_order import LEG_INDICES_16

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
KNOWN_BAD = os.path.join(FIXTURES, "standing_wiggle_known_bad.json")


# --------------------------------------------------------------------------- #
# Builders for self-consistent (known-good) 59-float references.
# --------------------------------------------------------------------------- #
def _quat_xyzw_from_rpy(roll, pitch, yaw):
    """XYZW quaternion (scipy ``as_quat()`` order) from roll/pitch/yaw."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([x, y, z, w])


def _backward_diff(x, dt):
    d = np.zeros_like(x)
    if x.shape[0] >= 2:
        d[1:] = (x[1:] - x[:-1]) / dt
        d[0] = d[1]
    return d


def build_good_reference(n=60, fps=50, move_legs=True, move_root=True):
    """Return ``(frames (N,59), fps)`` for a fully self-consistent reference.

    Every derived field is recomputed from the pose trajectory with the exact
    convention the validator uses, so a correct clip validates with no errors.
    """
    dt = 1.0 / fps
    t = np.arange(n) * dt

    root_pos = np.zeros((n, 3))
    if move_root:
        root_pos[:, 0] = 0.05 * np.sin(2 * np.pi * 0.5 * t)  # gentle forward sway
    root_pos[:, 2] = 0.15

    roll = 0.05 * np.sin(2 * np.pi * 1.0 * t)
    pitch = np.zeros(n)
    yaw = 0.08 * np.sin(2 * np.pi * 0.7 * t)
    root_quat = np.stack([_quat_xyzw_from_rpy(roll[i], pitch[i], yaw[i]) for i in range(n)])

    joints = np.zeros((n, 16))
    # neutral leg home pose
    home = {0: 0.002, 1: 0.053, 2: -0.63, 3: 1.368, 4: -0.784,
            11: -0.003, 12: -0.065, 13: 0.635, 14: 1.379, 15: -0.796}
    for k, v in home.items():
        joints[:, k] = v
    joints[:, 7] = 0.3 * np.sin(2 * np.pi * 1.0 * t)  # head_yaw
    if move_legs:
        joints[:, 3] += 0.15 * np.sin(2 * np.pi * 1.0 * t)   # left knee
        joints[:, 14] += 0.15 * np.sin(2 * np.pi * 1.0 * t)  # right knee

    lin_vel = _backward_diff(root_pos, dt)
    ang_vel = angular_velocity_from_quats(root_quat, dt)
    joints_vel = _backward_diff(joints, dt)
    contacts = np.ones((n, 2))

    frames = np.zeros((n, 59))
    frames[:, 0:3] = root_pos
    frames[:, 3:7] = root_quat
    frames[:, 7:23] = joints
    frames[:, 29:32] = lin_vel
    frames[:, 32:35] = ang_vel
    frames[:, 35:51] = joints_vel
    frames[:, 57:59] = contacts
    return frames, fps


# --------------------------------------------------------------------------- #
# Quaternion helpers match scipy (when available) and known identities.
# --------------------------------------------------------------------------- #
def test_quat_to_rotvec_identity():
    assert np.allclose(quat_to_rotvec(np.array([0.0, 0.0, 0.0, 1.0])), 0.0)


def test_quat_helpers_match_scipy():
    scipy_spatial = pytest.importorskip("scipy.spatial.transform")
    R = scipy_spatial.Rotation
    rng = np.random.default_rng(0)
    dt = 0.02
    quats = R.random(30, random_state=rng).as_quat()  # XYZW
    mine = angular_velocity_from_quats(quats, dt)
    sci = np.zeros_like(mine)
    for i in range(1, len(quats)):
        rel = R.from_quat(quats[i]) * R.from_quat(quats[i - 1]).inv()
        sci[i] = rel.as_rotvec() / dt
    sci[0] = sci[1]
    assert np.allclose(mine, sci, atol=1e-9)


def test_quat_mul_conj_roundtrip():
    q = np.array([0.1, -0.2, 0.3, 0.9])
    q = q / np.linalg.norm(q)
    ident = quat_mul(q, quat_conj(q))
    assert np.allclose(ident, [0, 0, 0, 1], atol=1e-12)


# --------------------------------------------------------------------------- #
# Known-good references validate cleanly.
# --------------------------------------------------------------------------- #
def test_good_reference_passes():
    frames, fps = build_good_reference()
    res = validate_reference(frames, fps, motion_type="full_body_test")
    assert res.ok, res.summary()
    assert res.errors == []


def test_good_reference_no_leg_warning_when_legs_move():
    frames, fps = build_good_reference(move_legs=True)
    res = validate_reference(frames, fps, motion_type="full_body_test")
    assert not any(i.field == "joints_pos" for i in res.warnings), res.summary()


# --------------------------------------------------------------------------- #
# THE acceptance test: the known-bad standing_wiggle must FAIL.
# --------------------------------------------------------------------------- #
def test_known_bad_standing_wiggle_fails():
    assert os.path.exists(KNOWN_BAD), "known-bad fixture missing"
    res = validate_reference_file(KNOWN_BAD)
    assert not res.ok, "validator must REJECT the known-bad reference"

    # It must be the angular-velocity axis transposition that trips it.
    ang_errs = [e for e in res.errors if e.field == "world_ang_vel"]
    assert ang_errs, "expected a world_ang_vel error naming the transposed axes"
    msg = ang_errs[0].message.lower()
    assert "transpos" in msg
    assert "x<->z" in msg or "z<->x" in msg

    # And the whole result must surface the zeroed linear-velocity channel.
    full = res.summary().lower()
    assert "world_lin_vel" in full
    assert "zero" in full


def test_known_bad_raise_on_error():
    with pytest.raises(rv.ReferenceValidationError) as exc:
        validate_reference_file(KNOWN_BAD, raise_on_error=True)
    assert "transpos" in str(exc.value).lower()


def test_known_bad_degenerate_legs_warning():
    """standing_wiggle's knees move ~0.0044 rad — a degenerate full-body ref."""
    res = validate_reference_file(KNOWN_BAD)
    leg_warns = [w for w in res.warnings if w.field == "joints_pos"]
    assert leg_warns, "expected a degenerate leg-motion warning"
    assert "knee" in leg_warns[0].message.lower()
    assert "0.0044" in leg_warns[0].message


# --------------------------------------------------------------------------- #
# Targeted per-defect detection on synthetic clips.
# --------------------------------------------------------------------------- #
def test_transposed_ang_vel_detected():
    frames, fps = build_good_reference()
    frames[:, [32, 34]] = frames[:, [34, 32]]  # swap ang_vel x and z
    res = validate_reference(frames, fps, motion_type="full")
    assert not res.ok
    assert any("transpos" in e.message.lower() for e in res.errors)


def test_zeroed_lin_vel_with_moving_root_is_error():
    frames, fps = build_good_reference(move_root=True)
    frames[:, 29:32] = 0.0
    res = validate_reference(frames, fps, motion_type="full")
    assert not res.ok
    errs = [e for e in res.errors if e.field == "world_lin_vel"]
    assert any("zero" in e.message.lower() for e in errs), res.summary()


def test_wrong_joint_vel_detected():
    frames, fps = build_good_reference()
    frames[:, 35:51] *= 0.5  # halve joint velocities → inconsistent with poses
    res = validate_reference(frames, fps)
    assert any(e.field == "joints_vel" for e in res.errors), res.summary()


def test_degenerate_legs_flagged_for_fullbody():
    frames, fps = build_good_reference(move_legs=False)
    res = validate_reference(frames, fps, motion_type="full_body")
    assert any(w.field == "joints_pos" for w in res.warnings), res.summary()


def test_non_fullbody_no_leg_warning():
    frames, fps = build_good_reference(move_legs=False)
    res = validate_reference(frames, fps, motion_type="head_nod")
    assert not any(w.field == "joints_pos" for w in res.warnings)


def test_bad_contacts_value_rejected():
    frames, fps = build_good_reference()
    frames[3, 57] = 0.5
    res = validate_reference(frames, fps)
    assert any(e.field == "foot_contacts" for e in res.errors)


def test_airborne_grounded_motion_rejected():
    frames, fps = build_good_reference()
    frames[:, 57:59] = 0.0  # both feet up for a "stand" motion
    res = validate_reference(frames, fps, motion_type="standing_wiggle")
    assert any(e.field == "foot_contacts" for e in res.errors)


def test_non_unit_quat_rejected():
    frames, fps = build_good_reference()
    frames[:, 3:7] *= 2.0
    res = validate_reference(frames, fps)
    assert any(e.field == "root_quat" for e in res.errors)


def test_wrong_frame_width_raises():
    with pytest.raises(ValueError):
        validate_reference(np.zeros((10, 58)), 50)
