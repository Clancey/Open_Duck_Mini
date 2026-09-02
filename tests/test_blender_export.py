"""Tests for the bpy-free Blender-export logic (plan §7 Phase 2).

Covers the four defect fixes (D2, D3, D4-adjacent determinism of assembly, D11),
the jnt_range→Euler limit derivation, clip metadata, the 59-float frame layout,
the head-envelope authoring warning, and the end-to-end export→compile path.

All logic here is exercised WITHOUT Blender (``bpy``); the Blender-facing shims
are imported only to prove they import cleanly with ``bpy is None``.
"""

import json

import numpy as np
import pytest

from open_duck_anim import clip as clipmod
from open_duck_anim.joint_order import JOINT_ORDER_16
from open_duck_anim.envelope import DEFLECTION_LIMITS

from open_duck_anim_blender import (
    contacts as contacts_mod,
    export as export_mod,
    jnt_range as jr,
    metadata as meta_mod,
    transform_table as tt,
)

DEG10 = tt.DEG10


# --------------------------------------------------------------------------- #
# D11 — transform table: zero pose + known-angle round-trip within 1e-6 rad.
# --------------------------------------------------------------------------- #
def _zero_eulers():
    return {b: (0.0, 0.0, 0.0) for b in tt.REQUIRED_BONES}


def test_table_order_matches_canonical():
    assert [t.joint_name for t in tt.JOINT_TRANSFORMS] == list(JOINT_ORDER_16)
    assert len(tt.JOINT_TRANSFORMS) == 16


def test_zero_pose_exports_expected_vector():
    """Rig rest/zero pose (all Euler 0) exports the declared offsets (D11)."""
    joints = tt.joints_from_bone_eulers(_zero_eulers())
    expected = [0.0] * 16
    expected[3] = -DEG10   # left_knee
    expected[4] = DEG10    # left_ankle
    expected[14] = -DEG10  # right_knee
    expected[15] = DEG10   # right_ankle
    assert np.allclose(joints, expected, atol=1e-6)
    # and the helper agrees
    assert np.allclose(tt.zero_pose_joints(), expected, atol=1e-6)


def test_known_joint_angles_export_within_1e6():
    """A set of known bone Euler angles exports to the expected joint vector."""
    eul = _zero_eulers()
    eul["knee_fk.l"] = (0.5, 0.0, 0.0)     # axis X
    eul["ankle_fk.l"] = (-0.2, 0.0, 0.0)   # axis X
    eul["head_yaw"] = (0.0, 0.0, 0.4)      # axis Z
    eul["hip_yaw_fk.l"] = (0.0, 0.3, 0.0)  # axis Y
    joints = tt.joints_from_bone_eulers(eul)
    assert joints[0] == pytest.approx(0.3, abs=1e-6)            # left_hip_yaw (Y)
    assert joints[3] == pytest.approx(0.5 - DEG10, abs=1e-6)    # left_knee (X - 10deg)
    assert joints[4] == pytest.approx(-0.2 + DEG10, abs=1e-6)   # left_ankle (X + 10deg)
    assert joints[7] == pytest.approx(0.4, abs=1e-6)            # head_yaw (Z)


def test_round_trip_bone_eulers():
    """forward then inverse recovers the per-bone axis value within 1e-6."""
    rng = np.random.default_rng(1234)
    eul = {}
    axis_vals = {}
    for t in tt.JOINT_TRANSFORMS:
        v = float(rng.uniform(-1.0, 1.0))
        e = [0.0, 0.0, 0.0]
        e[t.axis] = v
        eul[t.bone] = tuple(e)
        axis_vals[t.bone] = v
    joints = tt.joints_from_bone_eulers(eul)
    recovered = tt.bone_eulers_from_joints(joints)
    for bone, v in axis_vals.items():
        assert recovered[bone] == pytest.approx(v, abs=1e-6)


def test_missing_bone_raises():
    eul = _zero_eulers()
    del eul["antenna.l"]
    with pytest.raises(KeyError):
        tt.joints_from_bone_eulers(eul)


# --------------------------------------------------------------------------- #
# D2 — antenna L/R ordering fixed (idx 9 = left = antenna.l, idx 10 = right).
# --------------------------------------------------------------------------- #
def test_antenna_index_order_is_canonical():
    row9 = tt.JOINT_TRANSFORMS[9]
    row10 = tt.JOINT_TRANSFORMS[10]
    assert row9.joint_name == "left_antenna" and row9.bone == "antenna.l"
    assert row10.joint_name == "right_antenna" and row10.bone == "antenna.r"


def test_antenna_values_not_swapped_and_not_inverted():
    """antenna.l flows to idx 9, antenna.r to idx 10, with no sign inversion."""
    eul = _zero_eulers()
    eul["antenna.l"] = (0.0, 0.0, 0.3)
    eul["antenna.r"] = (0.0, 0.0, 0.7)
    joints = tt.joints_from_bone_eulers(eul)
    assert joints[9] == pytest.approx(0.3, abs=1e-9)   # left (not 0.7)
    assert joints[10] == pytest.approx(0.7, abs=1e-9)  # right (not 0.3)
    # sign convention (+1) preserved: no negation at record time.
    assert tt.JOINT_TRANSFORMS[9].sign == 1.0
    assert tt.JOINT_TRANSFORMS[10].sign == 1.0


# --------------------------------------------------------------------------- #
# jnt_range → Euler Limit Rotation bounds (mirrors MJCF; folds in D11 offset).
# --------------------------------------------------------------------------- #
def test_jnt_range_covers_14_dof_no_antennas():
    assert set(jr.constrained_joints()) == set(JOINT_ORDER_16) - {"left_antenna", "right_antenna"}
    assert len(jr.constrained_joints()) == 14


def test_head_yaw_euler_limit_matches_mjcf():
    axis, lo, hi = jr.euler_limit_for_joint("head_yaw")
    assert axis == 2  # Z
    assert lo == pytest.approx(-2.792526803190927, abs=1e-9)
    assert hi == pytest.approx(2.792526803190927, abs=1e-9)


def test_knee_euler_limit_folds_in_offset():
    """left_knee zero_offset=-DEG10, so the Euler limit shifts by +DEG10 (D11)."""
    axis, lo, hi = jr.euler_limit_for_joint("left_knee")
    assert axis == 0
    assert lo == pytest.approx(-1.5707963267948966 + DEG10, abs=1e-9)
    assert hi == pytest.approx(1.5707963267948966 + DEG10, abs=1e-9)


def test_antenna_has_no_jnt_range():
    with pytest.raises(KeyError):
        jr.euler_limit_for_joint("left_antenna")


# --------------------------------------------------------------------------- #
# D3 — foot contacts computed, and explicit invalid marker.
# --------------------------------------------------------------------------- #
def test_compute_foot_contacts_threshold():
    # left on ground, right lifted
    assert contacts_mod.compute_foot_contacts(0.005, 0.05, ground_z=0.0, threshold=0.01) == [1, 0]
    assert contacts_mod.compute_foot_contacts(0.02, 0.0, ground_z=0.0, threshold=0.01) == [0, 1]


def test_invalid_contacts_sentinel():
    assert contacts_mod.invalid_contacts() == [0, 0]


def test_episode_carries_footcontactvalid_marker():
    ep_valid = export_mod.new_episode(fps=50, contacts_valid=True)
    ep_invalid = export_mod.new_episode(fps=50, contacts_valid=False)
    assert ep_valid["FootContactValid"] is True
    assert ep_invalid["FootContactValid"] is False


# --------------------------------------------------------------------------- #
# 59-float frame assembly (Appendix B layout).
# --------------------------------------------------------------------------- #
def _dummy_segments(joints16, contacts):
    return dict(
        root_position=[0.0, 0.0, 0.1],
        root_quaternion=[0.0, 0.0, 0.0, 1.0],
        joint_positions=joints16,
        left_toe_pos=[0.0, 0.0, 0.0],
        right_toe_pos=[0.0, 0.0, 0.0],
        world_linear_vel=[0.0, 0.0, 0.0],
        world_angular_vel=[0.0, 0.0, 0.0],
        joint_velocities=[0.0] * 16,
        left_toe_vel=[0.0, 0.0, 0.0],
        right_toe_vel=[0.0, 0.0, 0.0],
        foot_contacts=contacts,
    )


def test_assemble_frame_layout():
    joints = list(range(16))
    frame = export_mod.assemble_frame(**_dummy_segments(joints, [1, 0]))
    assert len(frame) == 59
    assert frame[7:23] == [float(x) for x in joints]  # joints at 7:23
    assert frame[57:59] == [1.0, 0.0]                  # contacts at 57:59


def test_assemble_frame_bad_segment_length():
    seg = _dummy_segments(list(range(16)), [1, 1])
    seg["joint_positions"] = list(range(15))  # wrong
    with pytest.raises(ValueError):
        export_mod.assemble_frame(**seg)


# --------------------------------------------------------------------------- #
# Clip metadata → compiler meta.
# --------------------------------------------------------------------------- #
def test_metadata_to_compiler_meta_has_required_keys():
    md = meta_mod.ClipMetadata(name="curious", layer_mask="head")
    meta = md.to_compiler_meta()
    for k in (
        "name", "loop_mode", "blend_in_s", "blend_out_s", "show_blend_in_s",
        "show_blend_out_s", "layer_mask", "priority", "requires_mode",
        "antenna_calibration", "source_blend",
    ):
        assert k in meta
    assert meta["antenna_calibration"]["left"]["sign"] == 1
    assert meta["antenna_calibration"]["right"]["sign"] == -1


def test_metadata_validation_rejects_bad_enum():
    with pytest.raises(meta_mod.MetadataError):
        meta_mod.ClipMetadata(layer_mask="nonsense").to_compiler_meta()
    with pytest.raises(meta_mod.MetadataError):
        meta_mod.ClipMetadata(loop_mode="nope").to_compiler_meta()
    with pytest.raises(meta_mod.MetadataError):
        meta_mod.ClipMetadata(name="").to_compiler_meta()


# --------------------------------------------------------------------------- #
# Head safety-envelope authoring warning (advisory, D13).
# --------------------------------------------------------------------------- #
def test_envelope_warnings_empty_when_within_range():
    joints = [[0.0] * 16 for _ in range(5)]
    assert meta_mod.head_envelope_warnings(joints) == []


def test_envelope_warnings_flag_out_of_range_head():
    frame = [0.0] * 16
    # neck_pitch is index 5; safe high is DEFLECTION_LIMITS['neck_pitch'][1]
    over = DEFLECTION_LIMITS["neck_pitch"][1] + 0.5
    frame[5] = over
    warns = meta_mod.head_envelope_warnings([frame])
    assert len(warns) == 1
    assert warns[0].channel == "neck_pitch"
    assert warns[0].frame_index == 0
    assert "exceeds measured-safe range" in warns[0].message()


# --------------------------------------------------------------------------- #
# End-to-end: assemble episode → write JSON → compile .duckanim (reuses compiler)
# --------------------------------------------------------------------------- #
def _head_clip_episode(n=40):
    """A head-mask clip: legs held constant, head_yaw ramps, antennas move."""
    ep = export_mod.new_episode(fps=50, contacts_valid=False)
    leg_vals = {2: -0.63, 3: 1.368, 4: -0.784, 13: 0.635, 14: 1.379, 15: -0.796}
    for i in range(n):
        joints = [0.0] * 16
        for k, v in leg_vals.items():
            joints[k] = v
        frac = i / (n - 1)
        joints[7] = 0.5 * frac      # head_yaw ramp (within safe range)
        joints[9] = 0.3 * frac      # left_antenna rad
        joints[10] = 0.3 * frac     # right_antenna rad
        ep["Frames"].append(
            export_mod.assemble_frame(**_dummy_segments(joints, [0, 0]))
        )
    return ep


def test_export_and_compile_produces_valid_clip(tmp_path):
    ep = _head_clip_episode()
    meta = meta_mod.ClipMetadata(name="curious_head_tilt", layer_mask="head").to_compiler_meta()
    src = tmp_path / "curious.source.json"
    out = tmp_path / "curious.duckanim"
    result = export_mod.export_and_compile(ep, meta, str(src), str(out))

    assert src.exists() and out.exists()
    # 59-float source JSON round-trips and keeps our D3 marker.
    src_data = json.loads(src.read_text())
    assert src_data["FootContactValid"] is False
    assert len(src_data["Frames"][0]) == 59

    # The compiled clip validates and antennas came from indices 9/10.
    dclip = clipmod.load_clip(str(out))
    assert dclip.name == "curious_head_tilt"
    assert dclip.n_frames == 40
    l, r = dclip.runtime_antennas(39)  # last frame, ramped antennas
    assert l == pytest.approx(0.3 / 0.6, abs=1e-6)   # 0.3 rad over [-0.6,0.6] -> 0.5
    assert r == pytest.approx(-0.3 / 0.6, abs=1e-6)  # right sign = -1
    # provenance sha matches the written source text
    assert len(result["source_sha256"]) == 64


def test_export_is_deterministic(tmp_path):
    """Same episode → byte-identical compiled .duckanim (Phase 2 acceptance)."""
    ep = _head_clip_episode()
    meta = meta_mod.ClipMetadata(name="det", layer_mask="head").to_compiler_meta()
    from open_duck_anim import compiler

    src = tmp_path / "det.source.json"
    export_mod.write_source_json(ep, str(src))
    text = src.read_text()
    b1 = compiler.compile_to_json_bytes(text, meta)
    b2 = compiler.compile_to_json_bytes(text, meta)
    assert b1 == b2


def test_blender_shims_import_without_bpy():
    """recorder/constraints/panels import with bpy guarded to None on CI."""
    from open_duck_anim_blender import recorder, constraints, panels
    assert recorder.bpy is None
    assert constraints.bpy is None
    assert panels.bpy is None
    # Using a bpy-requiring entry point without Blender fails loudly, not silently.
    with pytest.raises(RuntimeError):
        recorder.DataRecorder()
