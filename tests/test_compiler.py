"""Tests for compiler.py (plan §4.1-C, §5)."""

import hashlib
import json

import numpy as np
import pytest

from open_duck_anim import compiler
from open_duck_anim.compiler import CompileError, COMPILER_VERSION

from _helpers import make_source_text, make_meta


def test_deterministic_byte_identical():
    src = make_source_text()
    meta = make_meta()
    b1 = compiler.compile_to_json_bytes(src, meta)
    b2 = compiler.compile_to_json_bytes(src, meta)
    assert b1 == b2


def test_provenance_hash_matches_source():
    src = make_source_text()
    d = compiler.compile_to_dict(src, make_meta())
    expected = hashlib.sha256(src.encode("utf-8")).hexdigest()
    assert d["provenance"]["source_sha256"] == expected
    assert d["provenance"]["compiler_version"] == COMPILER_VERSION
    assert d["provenance"]["source_blend"] == "test.blend"
    assert d["provenance"]["source_frame_range"] == [1, 20]


def test_provenance_hash_changes_with_source():
    d1 = compiler.compile_to_dict(make_source_text(head_yaw_end=0.5), make_meta())
    d2 = compiler.compile_to_dict(make_source_text(head_yaw_end=0.6), make_meta())
    assert d1["provenance"]["source_sha256"] != d2["provenance"]["source_sha256"]
    # Different source content ⇒ different compiled bytes too.
    b1 = compiler.compile_to_json_bytes(make_source_text(head_yaw_end=0.5), make_meta())
    b2 = compiler.compile_to_json_bytes(make_source_text(head_yaw_end=0.6), make_meta())
    assert b1 != b2


def test_antenna_radians_to_normalized_sign_and_clamp():
    # left sign +1, right sign -1; rad 0.6 maps to +1.0 (left) / -1.0 (right).
    src = make_source_text(antenna_end_rad=0.6)
    d = compiler.compile_to_dict(src, make_meta())
    assert d["show_functions"]["antenna_left"][-1] == pytest.approx(1.0)
    assert d["show_functions"]["antenna_right"][-1] == pytest.approx(-1.0)
    # first frame (rad 0) maps to 0 for both.
    assert d["show_functions"]["antenna_left"][0] == pytest.approx(0.0)
    assert d["show_functions"]["antenna_right"][0] == pytest.approx(0.0)


def test_antenna_range_clamping():
    # rad beyond rad_max clamps to +/-1.
    src = make_source_text(antenna_end_rad=5.0)
    d = compiler.compile_to_dict(src, make_meta())
    assert d["show_functions"]["antenna_left"][-1] == pytest.approx(1.0)
    assert d["show_functions"]["antenna_right"][-1] == pytest.approx(-1.0)
    assert max(d["show_functions"]["antenna_left"]) <= 1.0
    assert min(d["show_functions"]["antenna_right"]) >= -1.0


def test_frame_range_selection():
    src = make_source_text(n_frames=50)
    d = compiler.compile_to_dict(src, make_meta(), frame_range=[10, 20])
    assert d["frame_count"] == 11
    assert d["provenance"]["source_frame_range"] == [10, 20]


def test_missing_meta_key_raises():
    with pytest.raises(CompileError, match="missing required key"):
        compiler.compile_to_dict(make_source_text(), {"name": "x"})


def test_bad_source_json_raises():
    with pytest.raises(CompileError, match="not valid JSON"):
        compiler.compile_to_dict("{not json", make_meta())


def test_wrong_frame_size_raises():
    bad = json.dumps({"FPS": 50, "Frames": [[0.0] * 40]})
    with pytest.raises(CompileError, match="expected 59"):
        compiler.compile_to_dict(bad, make_meta())


def test_out_of_bounds_frame_range_raises():
    with pytest.raises(CompileError, match="out of bounds"):
        compiler.compile_to_dict(make_source_text(n_frames=10), make_meta(), frame_range=[5, 99])


def test_compile_file_roundtrip(tmp_path):
    src_path = tmp_path / "src.json"
    src_path.write_text(make_source_text())
    out_path = tmp_path / "out.duckanim"
    sha = compiler.compile_file(str(src_path), make_meta(), str(out_path))
    assert out_path.exists()
    loaded = json.loads(out_path.read_bytes())
    assert loaded["provenance"]["source_sha256"] == sha
