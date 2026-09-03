"""Unit tests for the pure mood-driven idle planner (``runtime/pi/idle_moods``).

These exercise the behaviour the owner asked for — the duck drifting between
moods, firing *congruent* one-shots, never repeating a beat, keeping the
big/startling reactions and the leg-moving ``dock_wiggle`` out of unprompted
play, and every emotion being able to return to neutral — entirely off the
robot (no engine, no hardware).
"""
import os
import random
import sys

import pytest

# The planner lives with the Pi service (runtime/pi) so the robot imports one
# copy; add it to the path for the off-robot test run.
_HERE = os.path.dirname(__file__)
_PI = os.path.abspath(os.path.join(_HERE, "..", "runtime", "pi"))
if _PI not in sys.path:
    sys.path.insert(0, _PI)

import idle_moods as im  # noqa: E402


# Every clip the planner could ever name, so "availability" never masks a bug.
ALL_CLIPS = sorted(
    {v for v in im.MOOD_LOOPS.values() if v}
    | {c for pool in im.CONGRUENT.values() for c in pool}
    | im.EXCLUDED_UNPROMPTED
)


def _planner(seed=0, available=None):
    return im.MoodPlanner(
        available if available is not None else ALL_CLIPS,
        rng=random.Random(seed),
    )


def test_tables_assert_safe():
    im.assert_unprompted_safe()          # raises on any violation


def test_dock_wiggle_and_big_reactions_never_scheduled():
    """The full-body ``dock_wiggle`` and the startling reactions appear in no
    congruent pool and are no mood's background loop — asserted, not omitted."""
    scheduled = {c for pool in im.CONGRUENT.values() for c in pool}
    scheduled |= {v for v in im.MOOD_LOOPS.values() if v}
    for banned in ("dock_wiggle", "startle", "flinch", "cower", "excited",
                   "happy_bounce"):
        assert banned in im.EXCLUDED_UNPROMPTED
        assert banned not in scheduled


def test_planner_never_returns_excluded_clip():
    """Drive a long randomised run and assert no beat is ever an excluded clip
    and no mood ever exposes a leg-moving/full-body loop."""
    for seed in range(25):
        p = _planner(seed)
        p.start(0.0)
        now = 0.0
        for _ in range(20000):
            now += 5.0
            p.maybe_transition(now)
            shot = p.maybe_oneshot(now)
            if shot is not None:
                assert shot not in im.EXCLUDED_UNPROMPTED
                assert shot in ALL_CLIPS
            assert p.background_clip() not in im.EXCLUDED_UNPROMPTED


def test_no_immediate_repeat_of_a_oneshot():
    p = _planner(3)
    p.start(0.0)
    now = 0.0
    prev = None
    fired = 0
    for _ in range(40000):
        now += 3.0
        p.maybe_transition(now)
        shot = p.maybe_oneshot(now)
        if shot is not None:
            assert shot != prev, "one-shot %r fired twice in a row" % shot
            prev = shot
            fired += 1
    assert fired > 50                    # the run actually produced beats


def test_oneshots_are_congruent_with_current_mood():
    """Every fired one-shot must belong to the pool of the mood that is active
    at the moment it fires."""
    p = _planner(7)
    p.start(0.0)
    now = 0.0
    for _ in range(40000):
        now += 3.0
        p.maybe_transition(now)
        shot = p.maybe_oneshot(now)
        if shot is not None:
            assert shot in im.CONGRUENT[p.mood], (
                "%r not congruent with mood %r" % (shot, p.mood)
            )


def test_incongruent_pairing_is_impossible():
    """A concrete regression guard: a happy_bounce-style clip must never be
    reachable from mood_sad, and sad_droop must never be reachable from a
    content mood."""
    assert "happy_bounce" not in im.CONGRUENT["sad"]
    assert "excited" not in im.CONGRUENT["sad"]
    assert "sad_droop" not in im.CONGRUENT["content"]
    assert "disappointed" not in im.CONGRUENT["content"]


def test_every_mood_can_reach_neutral():
    for start in im.MOODS:
        seen, frontier, reached = set(), [start], False
        while frontier:
            m = frontier.pop()
            if m == im.NEUTRAL:
                reached = True
                break
            if m in seen:
                continue
            seen.add(m)
            frontier.extend(im.TRANSITIONS[m])
        assert reached, "mood %r cannot reach neutral" % start


def test_scared_is_only_entered_via_alert():
    """Calm-by-default: no mood except ``alert`` (or ``scared`` staying put) can
    transition *into* ``scared``, so the duck never jumps straight into fear
    from a calm state."""
    for mood, succ in im.TRANSITIONS.items():
        if "scared" in succ and mood != "scared":
            assert mood == "alert", (
                "mood %r can transition to scared; only alert should" % mood
            )


def test_sleepy_does_not_jump_to_high_arousal_negative_moods():
    succ = set(im.TRANSITIONS["sleepy"])
    assert "grumpy" not in succ and "scared" not in succ and "sad" not in succ


def test_slow_blink_only_in_slow_blink_moods():
    p = _planner(11)
    p.start(0.0)
    now = 0.0
    for _ in range(20000):
        now += 2.0
        p.maybe_transition(now)
        if p.maybe_slow_blink(now):
            assert p.mood in im.SLOW_BLINK_MOODS


def test_seeded_runs_are_deterministic():
    def trace(seed):
        p = _planner(seed)
        p.start(0.0)
        out, now = [], 0.0
        for _ in range(3000):
            now += 5.0
            t = p.maybe_transition(now)
            s = p.maybe_oneshot(now)
            out.append((round(now, 3), t, s, p.mood))
        return out
    assert trace(42) == trace(42)
    assert trace(42) != trace(43)


def test_missing_mood_loop_falls_back_to_neutral():
    """If a mood's loop clip isn't deployed, the planner must never select that
    mood (it would have no background), but must still run."""
    avail = [c for c in ALL_CLIPS if c != "mood_scared"]
    p = _planner(5, available=avail)
    p.start(0.0)
    now = 0.0
    for _ in range(30000):
        now += 4.0
        p.maybe_transition(now)
        assert p.mood != "scared"


def test_quiet_beats_occur():
    """With quiet stretches enabled, some beats resolve to no one-shot (None)
    even though a beat was due."""
    p = _planner(1)
    p.start(0.0)
    now = 0.0
    beats_due = 0
    quiet = 0
    last_beat_marker = p._next_beat
    for _ in range(60000):
        now += 2.0
        p.maybe_transition(now)
        due = now >= p._next_beat
        shot = p.maybe_oneshot(now)
        if due:
            beats_due += 1
            if shot is None:
                quiet += 1
    # Not asserting an exact rate (mood pools of size 1 can also yield None
    # after a no-repeat filter), just that quiet beats genuinely happen.
    assert beats_due > 100
    assert quiet > 0
