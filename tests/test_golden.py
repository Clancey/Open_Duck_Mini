"""Golden-vector regression tests (plan §7 acceptance).

Recomputes a fixed compile + engine-evaluation scenario and asserts it matches
the committed golden vectors within an explicit tolerance. This guards against
cross-platform numerical drift (e.g. x86 dev host vs. ARM Raspberry Pi) and
against accidental behavioural changes.
"""

import hashlib
import json
import os

import numpy as np
import pytest

from open_duck_anim import Engine, Triggers, compiler
from open_duck_anim import clip as clipmod
from open_duck_anim.envelope import HeadEnvelope

from _helpers import make_meta, make_source_text

_GOLDEN = os.path.join(os.path.dirname(__file__), "golden", "golden_vectors.json")


def _load_golden():
    with open(_GOLDEN, "r") as fh:
        return json.load(fh)


def _rebuild():
    g = _load_golden()
    s = g["scenario"]
    src = make_source_text(
        n_frames=s["n_frames"],
        head_yaw_end=s["head_yaw_end"],
        antenna_end_rad=s["antenna_end_rad"],
        fps=s["fps"],
    )
    meta = make_meta(
        name="golden",
        loop_mode=s["loop_mode"],
        blend_in_s=s["blend_in_s"],
        blend_out_s=s["blend_out_s"],
        show_blend_in_s=s["show_blend_in_s"],
        show_blend_out_s=s["show_blend_out_s"],
        layer_mask="head",
        priority=5,
    )
    return g, src, meta


def test_compiled_bytes_hash_matches_golden():
    g, src, meta = _rebuild()
    cbytes = compiler.compile_to_json_bytes(src, meta)
    assert hashlib.sha256(cbytes).hexdigest() == g["compiled_sha256"]


def test_source_hash_matches_golden():
    g, src, meta = _rebuild()
    d = compiler.compile_to_dict(src, meta)
    assert d["provenance"]["source_sha256"] == g["source_sha256"]


def test_engine_outputs_match_golden():
    g, src, meta = _rebuild()
    tol = g["tolerance"]
    d = compiler.compile_to_dict(src, meta)
    c = clipmod.clip_from_dict(d)
    # The golden guards COMPILER + COMPOSITOR numeric determinism (cross-platform
    # drift), so it runs with the D13 safety envelope DELIBERATELY disabled
    # (reviewer E3) — the committed vectors are raw composited head deltas, and
    # decoupling them from the (separately tested) safety constants keeps this
    # regression guard stable across envelope re-derivations.
    eng = Engine(head_envelope=HeadEnvelope.unbounded())
    eng.evaluate(0.0, "stand", Triggers(clips=[c]))
    for row in g["samples"]:
        out = eng.evaluate(row["t"], "stand")
        np.testing.assert_allclose(
            out.head_command_offsets, row["head"], atol=tol,
            err_msg="head mismatch at t=%s" % row["t"],
        )
        assert out.show.antenna_l == pytest.approx(row["antenna_l"], abs=tol)
        assert out.show.antenna_r == pytest.approx(row["antenna_r"], abs=tol)
