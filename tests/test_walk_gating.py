"""Walk-gating guard: pin which clips are allowed to play WHILE WALKING.

Every emotion clip is ``layer_mask="head"`` — the RL policy owns the legs and a
head-masked overlay rides additively on top (plan §6.2; Disney §V-A: "we ignore
the leg joint positions"). So head-masked emotion during locomotion is
architecturally safe. What actually decides whether a clip may play while
walking is a *gating* choice encoded in ``requires_mode``:

  * ``any``  — plays in every mode, INCLUDING while walking.
  * ``walk`` — a locomotion-specific variant, only while walking.
  * ``stand``— standing/docked only; never plays while walking.
  * ``dock`` — the one full-body dock demo (``dock_wiggle``).

Each ``any``/``walk`` clip below was validated in the walking condition through
``experiments/animation/phase4_integrated_sim.py --walk`` against
``HEAD_PASSTHROUGH_300M.onnx`` at ×0.5 and full (×1.0) envelope: no falls, peak
tilt well under the 8.6° bound, zero joint position/velocity violations, and the
head tracks the authored motion (or its motion is intentionally sub-floor
carriage). The ``stand``-only clips are kept out of walking deliberately — some
because their sharp standing-tuned motion does not track over a gait, some
because they are simply wrong for a walking character (a duck does not yawn or
cower mid-stride), regardless of whether they are *safe*.

This golden map exists so a future edit to ``author_clips.py`` cannot silently
re-gate a clip (promote a stand-only pose into walking, or demote a validated
one) without a deliberate, reviewed change here.
"""

import glob
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from open_duck_anim import load_clip
from open_duck_anim.clip import HEAD_SLICE_16
from open_duck_anim.envelope import DEFAULT_ENVELOPE, HARDWARE_DERATING

_HERE = os.path.dirname(__file__)
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
_CLIPS_DIR = os.path.join(_REPO, "experiments", "animation", "clips")

# ---------------------------------------------------------------------------
# Golden gating map. Grouped by intent; the union is compared to what actually
# ships. Update this ONLY with a matching, reasoned change to author_clips.py.
# ---------------------------------------------------------------------------

# Head-masked clips that play in EVERY mode, including walking. The 12 idle /
# neutral clips that were always ``any`` plus the 6 emotion clips promoted after
# walk validation (see PROMOTED_TO_ANY).
MODE_ANY = {
    "affectionate", "confused_puzzled", "content_sigh", "curious_tilt",
    "idle_alive", "idle_breathe", "idle_lookaround", "idle_scan",
    "look_toward", "nod_yes", "nod_yes_soft", "scan_curious",
    # promoted to walking after validation:
    "double_take", "shake_no", "shake_no_reluctant", "disappointed",
    "suspicious_wary", "nervous_lookaround",
}

# The six emotion clips promoted from stand-only into ``any`` because they both
# track cleanly and read correctly while walking. ``shake_no`` /
# ``shake_no_reluctant`` are the headline fix: ``nod_yes`` was already ``any``,
# so the duck could say yes but not no while walking — that asymmetry is closed.
PROMOTED_TO_ANY = {
    "double_take", "shake_no", "shake_no_reluctant", "disappointed",
    "suspicious_wary", "nervous_lookaround",
}

# Locomotion-only variants (reduced amplitude, crisp enough to clear the gait
# disturbance floor). The two pre-existing ones plus four authored here.
MODE_WALK = {
    "walk_alert", "walk_look_around",
    "walk_excited", "walk_greeting", "walk_mood_sad", "walk_mood_alert",
}

# Kept standing/docked only. Reasons (see clips/README.md): fear/withdrawn poses
# that are incoherent mid-stride (cower, flinch, calm_down, startle, mood_scared,
# timid_shy); fast standing-tuned oscillation that does not track over a gait
# (excited, grumpy_annoyed, mood_alert, mood_content); subtle motion that sits at
# or below the walk disturbance floor and is not legible in motion (greeting,
# happy_bounce, perk_up, proud_pleased, sad_droop, flustered, mood_grumpy,
# mood_sad, mood_sleepy); and character-wrong-while-walking (sleepy_yawn — a duck
# does not yawn while marching).
MODE_STAND = {
    "calm_down", "cower", "excited", "flinch", "flustered", "greeting",
    "grumpy_annoyed", "happy_bounce", "mood_alert", "mood_content",
    "mood_grumpy", "mood_sad", "mood_scared", "mood_sleepy", "perk_up",
    "proud_pleased", "sad_droop", "sleepy_yawn", "startle", "timid_shy",
}

# The single full-body dock demo — stays dock-only, never touched.
MODE_DOCK = {"dock_wiggle"}

EXPECTED_MODE = {}
for _name in MODE_ANY:
    EXPECTED_MODE[_name] = "any"
for _name in MODE_WALK:
    EXPECTED_MODE[_name] = "walk"
for _name in MODE_STAND:
    EXPECTED_MODE[_name] = "stand"
for _name in MODE_DOCK:
    EXPECTED_MODE[_name] = "dock"

# Clips that are playable while walking (asserted safe + legible in motion).
WALKING_USABLE = MODE_ANY | MODE_WALK


def _shipped_modes():
    modes = {}
    for path in sorted(glob.glob(os.path.join(_CLIPS_DIR, "*.duckanim"))):
        name = os.path.basename(path)[: -len(".duckanim")]
        modes[name] = json.load(open(path))["requires_mode"]
    assert modes, "no .duckanim clips found in %s" % _CLIPS_DIR
    return modes


def test_requires_mode_matches_golden():
    """The shipped gating is exactly the reviewed golden map — no silent drift."""
    shipped = _shipped_modes()
    assert set(shipped) == set(EXPECTED_MODE), (
        "clip set changed; update the golden gating map in this test.\n"
        "  only shipped: %s\n  only expected: %s"
        % (sorted(set(shipped) - set(EXPECTED_MODE)),
           sorted(set(EXPECTED_MODE) - set(shipped)))
    )
    mismatched = {
        n: (shipped[n], EXPECTED_MODE[n])
        for n in shipped
        if shipped[n] != EXPECTED_MODE[n]
    }
    assert not mismatched, (
        "requires_mode changed without updating this guard (name: shipped->expected): %s"
        % mismatched
    )


def test_yes_and_no_are_symmetric_while_walking():
    """The duck must be able to say NO while walking, not only YES."""
    shipped = _shipped_modes()
    assert shipped["nod_yes"] == "any"
    assert shipped["shake_no"] == "any", (
        "shake_no must play while walking to match nod_yes; the yes/no asymmetry "
        "was the headline gap."
    )


@pytest.mark.parametrize("name", sorted(WALKING_USABLE))
def test_walking_clips_are_head_masked_and_within_derated_envelope(name):
    """Anything playable while walking must be a head overlay inside the ×0.5 envelope.

    This re-asserts the two invariants that make walking safe (head mask +
    derated envelope) specifically for the walking-usable set, so the guarantee
    is pinned to the walk gating and not only to the generic library test.
    """
    clip = load_clip(os.path.join(_CLIPS_DIR, name + ".duckanim"))
    assert clip.layer_mask == "head", (
        "%s plays while walking but is not head-masked" % name
    )
    env = DEFAULT_ENVELOPE.derated(HARDWARE_DERATING)
    head = clip.joints[:, HEAD_SLICE_16]
    prev = None
    max_clamp_delta = 0.0
    max_l2 = 0.0
    dt = 1.0 / clip.fps
    for row in head:
        clamped = env.clamp(np.asarray(row, dtype=np.float64),
                            prev_command_head=prev, dt=dt)
        max_clamp_delta = max(max_clamp_delta,
                              float(np.max(np.abs(clamped - row))))
        max_l2 = max(max_l2, float(np.sqrt(np.sum((row / env._L) ** 2))))
        prev = clamped
    assert max_l2 <= env.l2_budget + 1e-9, (
        "%s exceeds the derated combined budget (max ||c/L||_2=%.3f > %.3f)"
        % (name, max_l2, env.l2_budget)
    )
    assert max_clamp_delta <= 1e-3, (
        "%s would be clamped by the derated envelope (clampΔ=%.3e); re-author "
        "tighter instead of shipping clamped" % (name, max_clamp_delta)
    )


# ---------------------------------------------------------------------------
# Optional sim-backed check: run the promoted + authored walking clips through
# the integrated MuJoCo walk sim and assert they genuinely survive locomotion.
# Skips cleanly wherever the sim stack (mujoco + onnxruntime + a runtime clone +
# the policy onnx) is not present, so CI stays green.
# ---------------------------------------------------------------------------

TILT_BOUND_DEG = 8.6
_SIM_CLIPS = sorted(PROMOTED_TO_ANY | (MODE_WALK - {"walk_alert", "walk_look_around"}))


def _sim_prereqs_or_skip():
    pytest.importorskip("mujoco")
    pytest.importorskip("onnxruntime")
    runtime_home = os.environ.get("RUNTIME_HOME")
    if not runtime_home or not os.path.isdir(runtime_home):
        pytest.skip("RUNTIME_HOME not set to a runtime clone; walk-sim check skipped")
    onnx = os.path.join(_REPO, "HEAD_PASSTHROUGH_300M.onnx")
    if not os.path.isfile(onnx):
        pytest.skip("HEAD_PASSTHROUGH_300M.onnx not found at repo root; check skipped")
    sim = os.path.join(_REPO, "experiments", "animation", "phase4_integrated_sim.py")
    if not os.path.isfile(sim):
        pytest.skip("phase4_integrated_sim.py not found; check skipped")
    return runtime_home, onnx, sim


def _first_json_object(text):
    """phase4 prints a JSON summary followed by human trailer lines."""
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return json.loads(text[start : i + 1])
    raise ValueError("no JSON object found in sim output")


@pytest.mark.parametrize("name", _SIM_CLIPS)
def test_promoted_and_walk_clips_pass_walk_sim(name):
    """Each promoted/authored walking clip survives the integrated walk sim.

    Asserts, in the walking condition at the ×0.5 derating: no fall, peak tilt
    under the 8.6° bound, and zero joint position/velocity violations.
    """
    runtime_home, onnx, sim = _sim_prereqs_or_skip()
    env = dict(os.environ)
    env["RUNTIME_HOME"] = runtime_home
    env.setdefault("OPEN_DUCK_ANIM_HOME", _REPO)
    clip_path = os.path.join(_CLIPS_DIR, name + ".duckanim")
    out = subprocess.run(
        [sys.executable, sim, "--onnx", onnx, "--clip", clip_path,
         "--walk", "--derating", "0.5"],
        cwd=_REPO, env=env, capture_output=True, text=True, timeout=300,
    )
    assert out.returncode == 0, "sim failed for %s:\n%s" % (name, out.stderr[-2000:])
    summary = _first_json_object(out.stdout)
    assert not summary["fell"], "%s fell while walking" % name
    assert summary["peak_tilt_deg"] < TILT_BOUND_DEG, (
        "%s peak tilt %.2f° exceeds %.1f° bound"
        % (name, summary["peak_tilt_deg"], TILT_BOUND_DEG)
    )
    assert summary["pos_limit_violations"] == 0, "%s hit joint position limits" % name
    assert summary["vel_limit_violations"] == 0, "%s hit joint velocity limits" % name
