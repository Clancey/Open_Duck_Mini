"""Tests for clip.py loading and validation (plan §5, §6.2)."""

import json

import numpy as np
import pytest

from open_duck_anim import clip as clipmod
from open_duck_anim import compiler
from open_duck_anim.clip import ClipValidationError

from _helpers import make_source_text, make_meta, make_clip


def _valid_dict():
    return compiler.compile_to_dict(make_source_text(), make_meta())


def test_valid_clip_loads():
    c = make_clip()
    assert c.name == "clip"
    assert c.frame_count == 20
    assert c.duration_s == pytest.approx(0.4)
    assert c.joints.shape == (20, 16)
    assert c.show.antenna_l.shape == (20,)


def test_load_json_and_npz_roundtrip(tmp_path):
    # Full-structure round-trip: events, eyes, and antennas must all survive the
    # npz container, not just joints + antenna_l.
    src = make_source_text()
    meta = make_meta(
        events=[{"frame": 3, "type": "sound", "value": "a.wav"}],
        eyes=[1] * 20,
    )
    d = compiler.compile_to_dict(src, meta)
    jpath = tmp_path / "c.duckanim"
    jpath.write_bytes(compiler.compile_to_json_bytes(src, meta))
    c1 = clipmod.load_clip(str(jpath))
    # Write via the canonical package writer (the documented inverse of the reader).
    npz_path = tmp_path / "c.npz"
    clipmod.save_clip_npz(str(npz_path), d)
    c2 = clipmod.load_clip(str(npz_path))
    assert np.allclose(c1.joints, c2.joints)
    assert np.allclose(c1.show.antenna_l, c2.show.antenna_l)
    assert np.allclose(c1.show.antenna_r, c2.show.antenna_r)
    assert np.array_equal(c1.show.eyes, c2.show.eyes)
    assert [(e.frame, e.type, e.value) for e in c1.show.events] == \
           [(e.frame, e.type, e.value) for e in c2.show.events]
    assert len(c2.show.events) == 1 and c2.show.events[0].frame == 3


def test_antenna_precedence_no_joint_accessor():
    # Structural guarantee: the runtime CANNOT read antenna values from the joint
    # array — columns 9/10 are zeroed at load. The only source is show_functions.
    c = make_clip(antenna_const_rad=0.4)
    left, right = c.runtime_antennas(c.frame_count - 1)
    # antennas are still available via the show-function accessor (normalised).
    assert left == pytest.approx(c.show.antenna_l[-1])
    assert right == pytest.approx(c.show.antenna_r[-1])
    # ...but the joint array columns 9,10 are all zero — no radians leak through.
    assert np.all(c.joints[:, 9:11] == 0.0)
    # and the normalised antenna values are non-zero, proving they didn't come
    # from the (zeroed) joint columns.
    assert abs(c.show.antenna_l[-1]) > 1e-6


def test_nonfinite_joint_rejected_json(tmp_path):
    # C2: a NaN joint value must be rejected, not clamped/propagated.
    d = _valid_dict()
    frames = np.asarray(d["joints"]["frames"])
    frames[5, 7] = np.nan
    d["joints"]["frames"] = frames.tolist()
    with pytest.raises(ClipValidationError, match="non-finite"):
        clipmod.validate_clip_dict(d)


def test_nonfinite_antenna_rejected(tmp_path):
    # C2: NaN passes abs()>1 range checks (NaN comparisons are false); must be
    # caught by the explicit finite check.
    d = _valid_dict()
    a = list(d["show_functions"]["antenna_left"])
    a[3] = float("nan")
    d["show_functions"]["antenna_left"] = a
    with pytest.raises(ClipValidationError, match="non-finite"):
        clipmod.validate_clip_dict(d)


def test_nonfinite_rejected_via_npz(tmp_path):
    # C2: the npz path never touches the JSON encoder — finite checks must run
    # on the loaded arrays too.
    src = make_source_text()
    meta = make_meta()
    d = compiler.compile_to_dict(src, meta)
    frames = np.asarray(d["joints"]["frames"], dtype=np.float64)
    frames[5, 7] = np.nan
    d["joints"]["frames"] = frames
    npz_path = tmp_path / "bad.npz"
    clipmod.save_clip_npz(str(npz_path), d)
    with pytest.raises(ClipValidationError, match="non-finite"):
        clipmod.load_clip(str(npz_path))


def test_nan_json_literal_rejected(tmp_path):
    # C2: json.load accepts bare NaN/Infinity by default; the loader must reject
    # them via parse_constant.
    jpath = tmp_path / "c.duckanim"
    jpath.write_bytes(compiler.compile_to_json_bytes(make_source_text(), make_meta()))
    text = jpath.read_text()
    # inject a raw NaN literal into the joints frames
    bad = text.replace("0.0", "NaN", 1)
    jpath.write_text(bad)
    with pytest.raises(ClipValidationError, match="non-finite literal"):
        clipmod.load_clip(str(jpath))


def test_swapped_antenna_signs_rejected():
    # M3: signs are fixed hardware constants (LEFT=+1, RIGHT=-1, Appendix A).
    # A swapped-sign calibration (the upstream Blender L/R swap defect D2) must
    # be rejected, not silently inverted on hardware.
    d = _valid_dict()
    d["antenna_calibration"]["left"]["sign"] = -1
    d["antenna_calibration"]["right"]["sign"] = 1
    with pytest.raises(ClipValidationError, match="Appendix A"):
        clipmod.validate_clip_dict(d)


def test_missing_required_field_rejected():
    d = _valid_dict()
    del d["fps"]
    with pytest.raises(ClipValidationError, match="fps"):
        clipmod.validate_clip_dict(d)


def test_frame_count_mismatch_rejected():
    d = _valid_dict()
    d["frame_count"] = 19  # actual arrays are 20
    with pytest.raises(ClipValidationError, match="frame_count"):
        clipmod.validate_clip_dict(d)


def test_duration_mismatch_rejected():
    d = _valid_dict()
    d["duration_s"] = 1.23
    with pytest.raises(ClipValidationError, match="duration_s"):
        clipmod.validate_clip_dict(d)


def test_blend_overlap_rejected():
    d = _valid_dict()
    d["blend_in_s"] = 0.3
    d["blend_out_s"] = 0.3  # sum 0.6 > duration 0.4
    with pytest.raises(ClipValidationError, match="blend_in_s"):
        clipmod.validate_clip_dict(d)


def test_bad_loop_mode_rejected():
    d = _valid_dict()
    d["loop_mode"] = "bounce"
    with pytest.raises(ClipValidationError, match="loop_mode"):
        clipmod.validate_clip_dict(d)


def test_bad_layer_mask_rejected():
    d = _valid_dict()
    d["layer_mask"] = "arms"
    with pytest.raises(ClipValidationError, match="layer_mask"):
        clipmod.validate_clip_dict(d)


def test_bad_requires_mode_rejected():
    d = _valid_dict()
    d["requires_mode"] = "fly"
    with pytest.raises(ClipValidationError, match="requires_mode"):
        clipmod.validate_clip_dict(d)


def test_wrong_joint_order_rejected():
    d = _valid_dict()
    d["joints"]["order"] = list(reversed(d["joints"]["order"]))
    with pytest.raises(ClipValidationError, match="JOINT_ORDER_16"):
        clipmod.validate_clip_dict(d)


def test_show_track_length_rejected():
    d = _valid_dict()
    d["show_functions"]["antenna_left"] = d["show_functions"]["antenna_left"][:-1]
    with pytest.raises(ClipValidationError, match="antenna_left"):
        clipmod.validate_clip_dict(d)


def test_event_frame_out_of_range_rejected():
    d = _valid_dict()
    d["show_functions"]["events"] = [{"frame": 999, "type": "sound", "value": "a.wav"}]
    with pytest.raises(ClipValidationError, match="out of range"):
        clipmod.validate_clip_dict(d)


def test_antenna_track_out_of_range_rejected():
    d = _valid_dict()
    d["show_functions"]["antenna_left"] = [2.0] * d["frame_count"]
    with pytest.raises(ClipValidationError, match="normalised"):
        clipmod.validate_clip_dict(d)


def test_head_mask_with_moving_legs_rejected():
    # A head-masked clip whose leg channels move is illegal (plan §5.2/§6.2).
    src = make_source_text(move_legs=True)
    with pytest.raises(ClipValidationError, match="leg block"):
        compiler.compile_to_dict(src, make_meta(layer_mask="head"))


def test_legs_mask_rejected_in_supported_modes():
    # legs are never animation-movable in Phase 1 modes (plan §6.2).
    src = make_source_text(move_legs=True)
    with pytest.raises(ClipValidationError, match="§6.2|legs"):
        compiler.compile_to_dict(src, make_meta(layer_mask="legs", requires_mode="stand"))


def test_channel_violation_warn_mode():
    # Build a dict that would violate leg-neutrality, then validate in warn mode.
    good = _valid_dict()
    frames = np.asarray(good["joints"]["frames"])
    frames[:, 3] = np.linspace(1.368, 1.768, good["frame_count"])  # left_knee moves
    good["joints"]["frames"] = frames.tolist()
    with pytest.warns(UserWarning, match="leg block"):
        clipmod.validate_clip_dict(good, on_channel_violation="warn")



