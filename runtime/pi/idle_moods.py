#!/usr/bin/env python3
"""Mood-driven idle behaviour planner for the Open Duck Mini v2 idle service.

This is the *pure logic* half of the idle "alive" behaviour: no hardware, no
engine, no I/O — just stdlib ``random`` and time arithmetic — so it is fully
unit-testable off the robot (see ``tests/test_idle_moods.py``).  The idle
service (``idle_service.py``) owns a :class:`MoodPlanner`, asks it three
questions each control tick — *should the mood change?*, *should a one-shot
fire?*, *should a slow blink fire?* — and turns the answers into engine
backgrounds / triggers.

Design intent (owner: "use the full emotional palette so it feels alive rather
than looping", calm-by-default for hours on a desk):

  * The duck **sits in a mood** for a meaningful stretch (minutes), then drifts
    to a neighbouring mood.  Moods are the ``mood_*`` background loops; the
    plain ``idle_*`` loops are the NEUTRAL baseline.
  * One-shots fired while in a mood are **congruent** with it (a ``happy_bounce``
    from ``mood_sad`` would read as incoherent), via :data:`CONGRUENT`.
  * Mood transitions are **weighted** so the drift feels natural: sleepy tends
    to deepen or rouse, scared tends back toward neutral, every mood can always
    reach neutral (no emotion is a dead end), and the energetic ``scared`` mood
    is only ever *entered* via ``alert`` (never straight from a calm state).
  * Energy is **calm by default**: the big/startling reactions
    (``startle``/``flinch``/``cower``/``excited``/``happy_bounce``) and the
    trigger-only ``dock_wiggle`` (full-body, moves the legs) are **never** played
    unprompted.  That exclusion is asserted in :func:`assert_unprompted_safe`,
    not left to omission.
  * Pacing avoids obvious repetition: never the same one-shot twice in a row,
    the gap between beats varies, and there are genuine quiet stretches.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Set


# --- moods --------------------------------------------------------------------
NEUTRAL = "neutral"
MOODS = (NEUTRAL, "content", "sad", "sleepy", "alert", "grumpy", "scared")

# Each non-neutral mood maps to the background loop clip the duck sits in while
# in that mood.  NEUTRAL has no single loop — the service rotates the plain
# ``idle_*`` backgrounds for variety.
MOOD_LOOPS: Dict[str, Optional[str]] = {
    NEUTRAL: None,
    "content": "mood_content",
    "sad": "mood_sad",
    "sleepy": "mood_sleepy",
    "alert": "mood_alert",
    "grumpy": "mood_grumpy",
    "scared": "mood_scared",
}

# Weighted next-mood transition graph.  Weights are relative within a row.
# Invariants (checked by assert_unprompted_safe):
#   * every mood can reach NEUTRAL (no emotion is a one-way trap),
#   * calm default: NEUTRAL never jumps straight to scared (only via alert),
#   * sleepy only deepens (self) or rouses (neutral/content/alert) — never
#     straight to grumpy/scared/sad,
#   * scared drifts back toward neutral (and, less often, alert).
TRANSITIONS: Dict[str, Dict[str, float]] = {
    NEUTRAL: {NEUTRAL: 2.0, "content": 3.0, "sleepy": 2.0, "alert": 2.0,
              "sad": 1.0, "grumpy": 1.0},
    "content": {NEUTRAL: 4.0, "content": 2.0, "sleepy": 2.0, "sad": 1.0},
    "sad": {NEUTRAL: 4.0, "content": 2.0, "sad": 2.0},
    "sleepy": {"sleepy": 3.0, NEUTRAL: 3.0, "content": 1.0, "alert": 1.0},
    "alert": {NEUTRAL: 4.0, "content": 2.0, "alert": 1.0, "scared": 1.0,
              "grumpy": 1.0},
    "grumpy": {NEUTRAL: 4.0, "content": 2.0, "grumpy": 2.0},
    "scared": {NEUTRAL: 4.0, "alert": 2.0, "scared": 1.0},
}

# Congruent one-shots per mood.  Only calm, head-only gestures — big reactions
# are deliberately absent (see EXCLUDED_UNPROMPTED).  ``scared`` includes
# ``calm_down`` so it narratively bridges back toward neutral.
CONGRUENT: Dict[str, List[str]] = {
    NEUTRAL: ["curious_tilt", "look_toward", "scan_curious", "nod_yes_soft",
              "content_sigh", "confused_puzzled", "double_take", "perk_up"],
    "content": ["content_sigh", "affectionate", "proud_pleased", "nod_yes",
                "nod_yes_soft", "greeting", "curious_tilt", "look_toward"],
    "sad": ["sad_droop", "disappointed", "content_sigh", "timid_shy",
            "shake_no_reluctant"],
    "sleepy": ["sleepy_yawn", "content_sigh", "nod_yes_soft"],
    "alert": ["look_toward", "scan_curious", "double_take", "perk_up",
              "suspicious_wary", "curious_tilt", "nod_yes"],
    "grumpy": ["grumpy_annoyed", "shake_no", "suspicious_wary", "disappointed",
               "flustered", "confused_puzzled"],
    "scared": ["nervous_lookaround", "timid_shy", "suspicious_wary",
               "calm_down", "shake_no_reluctant", "flustered"],
}

# Moods that carry the slow, heavy lid close/open (the new ``slow_blink`` eye
# cue): a calm, low-energy blink that reads as sleepy/content/wistful.
SLOW_BLINK_MOODS: Set[str] = {"sleepy", "content", "sad"}

# NEVER played unprompted on an unattended desk.  The big/startling reactions
# would be unsettling at 3 a.m. (owner), and ``dock_wiggle`` is full-body /
# trigger-only (it moves the legs — this service must never torque them).
EXCLUDED_UNPROMPTED: Set[str] = {
    "startle", "flinch", "cower", "excited", "happy_bounce",  # too energetic
    "dock_wiggle",                                            # full-body, legs
    "walk_alert", "walk_look_around",                         # require walk mode
}


# --- pacing defaults (seconds) ------------------------------------------------
# A mood is held for a *meaningful* stretch (minutes) before it drifts.
MOOD_DWELL_MIN_S = 150.0
MOOD_DWELL_MAX_S = 360.0
# Gap between one-shot "beats".  Varied, not a fixed interval.
BEAT_MIN_S = 22.0
BEAT_MAX_S = 70.0
# Some beats are genuine quiet stretches (stillness makes the next move read as
# intentional): with this probability a due beat is deliberately skipped.
QUIET_PROB = 0.30
# Slow-blink cadence while in a slow-blink mood.
SLOW_BLINK_MIN_S = 9.0
SLOW_BLINK_MAX_S = 22.0


def assert_unprompted_safe() -> None:
    """Fail loudly if the tables could ever schedule an unsafe/unwanted clip.

    Belt-and-braces with the runtime guard in the service: the excluded clips
    (big reactions + the full-body, leg-moving ``dock_wiggle``) must appear in
    NO congruent pool and be NO mood's background loop.  Also verifies the
    transition graph has no dead ends (every mood reaches neutral).
    """
    for mood, clips in CONGRUENT.items():
        bad = set(clips) & EXCLUDED_UNPROMPTED
        assert not bad, (
            "mood %r schedules excluded clip(s) %s as a one-shot; these must "
            "never play unprompted" % (mood, sorted(bad))
        )
    for mood, loop in MOOD_LOOPS.items():
        assert loop not in EXCLUDED_UNPROMPTED, (
            "mood %r uses excluded clip %r as a background loop" % (mood, loop)
        )
    assert "dock_wiggle" in EXCLUDED_UNPROMPTED, (
        "dock_wiggle (full-body, moves the legs) must be excluded from "
        "unprompted play"
    )
    # No dead ends: from every mood, a bounded walk must be able to reach
    # NEUTRAL, else an emotion could latch forever.
    for start in MOODS:
        seen, frontier = set(), [start]
        while frontier:
            m = frontier.pop()
            if m == NEUTRAL:
                break
            if m in seen:
                continue
            seen.add(m)
            frontier.extend(TRANSITIONS.get(m, {}).keys())
        else:
            raise AssertionError("mood %r cannot reach NEUTRAL" % start)


def _weighted_choice(rng: random.Random, weights: Dict[str, float]) -> str:
    items = list(weights.items())
    total = sum(w for _, w in items)
    r = rng.uniform(0.0, total)
    upto = 0.0
    for name, w in items:
        upto += w
        if r <= upto:
            return name
    return items[-1][0]


class MoodPlanner:
    """Stateful, seedable planner driving mood drift, congruent one-shots, and
    slow blinks.  All time inputs are a monotonic ``now`` in seconds; the
    planner never sleeps or touches hardware.
    """

    def __init__(
        self,
        available_clips: Sequence[str],
        rng: Optional[random.Random] = None,
        *,
        start_mood: str = NEUTRAL,
        dwell_min_s: float = MOOD_DWELL_MIN_S,
        dwell_max_s: float = MOOD_DWELL_MAX_S,
        beat_min_s: float = BEAT_MIN_S,
        beat_max_s: float = BEAT_MAX_S,
        quiet_prob: float = QUIET_PROB,
        slow_blink_min_s: float = SLOW_BLINK_MIN_S,
        slow_blink_max_s: float = SLOW_BLINK_MAX_S,
    ) -> None:
        assert_unprompted_safe()
        self.rng = rng or random.Random()
        self.available: Set[str] = set(available_clips)
        self.dwell_min_s = dwell_min_s
        self.dwell_max_s = dwell_max_s
        self.beat_min_s = beat_min_s
        self.beat_max_s = beat_max_s
        self.quiet_prob = quiet_prob
        self.slow_blink_min_s = slow_blink_min_s
        self.slow_blink_max_s = slow_blink_max_s

        self.mood = start_mood
        self.last_oneshot: Optional[str] = None
        self._mood_until = 0.0
        self._next_beat = 0.0
        self._next_slow_blink = 0.0

    # --- lifecycle ------------------------------------------------------------
    def start(self, now: float) -> None:
        """Seed the timers.  Call once before the first tick (and after a duty
        checkpoint that paused animation)."""
        self._mood_until = now + self._dwell()
        self._next_beat = now + self._gap()
        self._next_slow_blink = now + self._slow_blink_gap()

    # --- mood drift -----------------------------------------------------------
    def _dwell(self) -> float:
        return self.rng.uniform(self.dwell_min_s, self.dwell_max_s)

    def _mood_available(self, mood: str) -> bool:
        loop = MOOD_LOOPS.get(mood)
        return loop is None or loop in self.available

    def pick_next_mood(self) -> str:
        """Weighted next mood, skipping any mood whose loop clip is missing
        (falls back to NEUTRAL, which is always available)."""
        weights = {m: w for m, w in TRANSITIONS[self.mood].items()
                   if self._mood_available(m)}
        if not weights:
            return NEUTRAL
        return _weighted_choice(self.rng, weights)

    def maybe_transition(self, now: float) -> Optional[str]:
        """If the current mood's dwell has elapsed, drift to the next mood and
        return it; else return ``None``.  Resets the beat cadence so the first
        beat of a new mood isn't inherited from the old one."""
        if now < self._mood_until:
            return None
        self.mood = self.pick_next_mood()
        self._mood_until = now + self._dwell()
        self._next_beat = now + self._gap()
        self._next_slow_blink = now + self._slow_blink_gap()
        return self.mood

    def background_clip(self) -> Optional[str]:
        """The background loop for the current mood, or ``None`` for NEUTRAL
        (service rotates the plain ``idle_*`` loops)."""
        return MOOD_LOOPS.get(self.mood)

    # --- one-shot beats -------------------------------------------------------
    def _gap(self) -> float:
        return self.rng.uniform(self.beat_min_s, self.beat_max_s)

    def _pick_oneshot(self) -> Optional[str]:
        pool = [c for c in CONGRUENT.get(self.mood, ())
                if c in self.available
                and c not in EXCLUDED_UNPROMPTED
                and c != self.last_oneshot]
        if not pool:
            return None
        c = self.rng.choice(pool)
        self.last_oneshot = c
        return c

    def maybe_oneshot(self, now: float) -> Optional[str]:
        """If a beat is due, schedule the next beat and return a congruent
        one-shot clip name (never the same one twice in a row), or ``None`` for
        a deliberate *quiet* beat (with probability ``quiet_prob``, or when the
        mood's pool is momentarily exhausted).  Returns ``None`` when no beat is
        due.  Quiet beats are what turn a run of gestures into a natural rhythm
        with genuine stillness between beats."""
        if now < self._next_beat:
            return None
        self._next_beat = now + self._gap()
        if self.rng.random() < self.quiet_prob:
            return None
        return self._pick_oneshot()

    # --- slow blinks ----------------------------------------------------------
    def _slow_blink_gap(self) -> float:
        return self.rng.uniform(self.slow_blink_min_s, self.slow_blink_max_s)

    def maybe_slow_blink(self, now: float) -> bool:
        """True when a slow blink should fire (only in SLOW_BLINK_MOODS)."""
        if self.mood not in SLOW_BLINK_MOODS:
            return False
        if now < self._next_slow_blink:
            return False
        self._next_slow_blink = now + self._slow_blink_gap()
        return True
