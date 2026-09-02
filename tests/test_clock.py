"""Tests for clock ownership and discrete events (plan §6.1)."""

import numpy as np
import pytest

from open_duck_anim import Engine, Triggers
from open_duck_anim.clip import DiscreteEvent

from _helpers import make_clip

HEAD_YAW = 2


def test_phase_from_elapsed_time_not_tick_count():
    # head_yaw ramps 0→0.5 over 20 frames (0.4s). The value must depend on
    # elapsed time, not on the number of evaluate() calls.
    c = make_clip(loop_mode="clamp", head_yaw_end=0.5, blend_in_s=0.0, blend_out_s=0.0)
    eng = Engine()
    eng.evaluate(0.0, "stand", Triggers(clips=[c]))
    # jump straight to t=0.2 (halfway) — value should be ~0.25 regardless of
    # having skipped the intermediate ticks.
    v = eng.evaluate(0.2, "stand").head_command_offsets[HEAD_YAW]
    # 20 frames span head_yaw 0..0.5 as 0.5*i/19; cum=10 → frame 10.
    assert v == pytest.approx(0.5 * 10 / 19, abs=1e-6)


def test_skipped_frames_land_on_correct_time():
    c = make_clip(loop_mode="clamp", head_yaw_end=0.5, blend_in_s=0.0, blend_out_s=0.0)
    eng_smooth = Engine()
    eng_smooth.evaluate(0.0, "stand", Triggers(clips=[c]))
    # advance smoothly
    v_smooth = None
    t = 0.0
    while t < 0.3:
        t += 0.02
        v_smooth = eng_smooth.evaluate(t, "stand").head_command_offsets[HEAD_YAW]

    eng_jump = Engine()
    eng_jump.evaluate(0.0, "stand", Triggers(clips=[c]))
    v_jump = eng_jump.evaluate(t, "stand").head_command_offsets[HEAD_YAW]
    assert v_jump == pytest.approx(v_smooth, abs=1e-6)


def _events_clip(loop_mode="once"):
    return make_clip(
        loop_mode=loop_mode,
        head_yaw_end=0.0,   # keep head neutral so mask stays legal
        move_head=False,
        events=[{"frame": 10, "type": "sound", "value": "ping.wav"}],
        blend_in_s=0.0,
        blend_out_s=0.0,
    )


def test_event_fires_exactly_once_normal_ticks():
    c = _events_clip("once")
    eng = Engine()
    eng.evaluate(0.0, "stand", Triggers(clips=[c]))
    fired = []
    t = 0.0
    while t < 0.4:
        t += 0.02
        out = eng.evaluate(t, "stand")
        fired.extend(out.show.events)
    assert len(fired) == 1
    assert fired[0].value == "ping.wav"


def test_event_fires_exactly_once_across_overrun():
    # A single huge overrun that crosses the event frame fires it exactly once.
    c = _events_clip("once")
    eng = Engine()
    eng.evaluate(0.0, "stand", Triggers(clips=[c]))
    out = eng.evaluate(0.4, "stand")  # jump past frame 10 (t=0.2) in one tick
    assert len(out.show.events) == 1
    # subsequent ticks do not re-fire it
    out2 = eng.evaluate(0.6, "stand")
    assert len(out2.show.events) == 0


def test_multiple_events_fire_in_order_during_overrun():
    c = make_clip(
        loop_mode="once",
        move_head=False,
        events=[
            {"frame": 12, "type": "projector", "value": "on"},
            {"frame": 3, "type": "sound", "value": "a.wav"},
        ],
        blend_in_s=0.0,
        blend_out_s=0.0,
    )
    eng = Engine()
    eng.evaluate(0.0, "stand", Triggers(clips=[c]))
    out = eng.evaluate(0.4, "stand")  # crosses frames 3 and 12
    vals = [e.value for e in out.show.events]
    assert vals == ["a.wav", "on"]  # ordered by frame


def test_wrap_event_fires_once_per_loop():
    c = make_clip(
        loop_mode="wrap",
        move_head=False,
        events=[{"frame": 10, "type": "sound", "value": "tick.wav"}],
        blend_in_s=0.0,
        blend_out_s=0.0,
    )
    eng = Engine()
    eng.evaluate(0.0, "stand", Triggers(clips=[c]))
    fired = []
    t = 0.0
    # run two full loops (0.8s); event at frame 10 (t=0.2, 0.6) → exactly 2.
    while t < 0.8:
        t += 0.02
        fired.extend(eng.evaluate(t, "stand").show.events)
    assert len(fired) == 2


def test_cancelled_clip_events_do_not_fire():
    # Cancelling a clip cancels its pending events (plan §6.5-a).
    c = _events_clip("once")
    eng = Engine()
    eng.evaluate(0.0, "stand", Triggers(clips=[c]))
    eng.evaluate(0.1, "stand", Triggers(cancel=True))  # cancel before frame 10 (t=0.2)
    fired = []
    t = 0.1
    while t < 0.4:
        t += 0.02
        fired.extend(eng.evaluate(t, "stand").show.events)
    assert fired == []


def test_events_are_discrete_objects():
    c = _events_clip("once")
    eng = Engine()
    eng.evaluate(0.0, "stand", Triggers(clips=[c]))
    out = eng.evaluate(0.4, "stand")
    assert all(isinstance(e, DiscreteEvent) for e in out.show.events)


def _frame0_clip(loop_mode):
    return make_clip(
        loop_mode=loop_mode,
        move_head=False,
        events=[{"frame": 0, "type": "sound", "value": "open.wav"}],
        blend_in_s=0.0,
        blend_out_s=0.0,
    )


def test_frame0_event_fires_once_in_once_and_clamp():
    # C1: a frame-0 event (opening cue) must fire exactly once, not zero times.
    for mode in ("once", "clamp"):
        c = _frame0_clip(mode)
        eng = Engine()
        fired = list(eng.evaluate(0.0, "stand", Triggers(clips=[c])).show.events)
        t = 0.0
        while t < 0.4:
            t += 0.02
            fired.extend(eng.evaluate(t, "stand").show.events)
        assert len(fired) == 1, "mode=%s fired %d" % (mode, len(fired))


def test_frame0_event_wrap_fires_per_loop():
    # C1: over two loops a frame-0 event fires 3 times (t=0, 0.4, 0.8), not 2.
    c = _frame0_clip("wrap")
    eng = Engine()
    fired = list(eng.evaluate(0.0, "stand", Triggers(clips=[c])).show.events)
    t = 0.0
    while t < 0.8:
        t += 0.02
        fired.extend(eng.evaluate(t, "stand").show.events)
    assert len(fired) == 3


def test_events_not_refired_on_backwards_time():
    # M1: t=0.30 → 0.10 → 0.30 must fire the occurrence exactly once.
    c = _events_clip("once")  # event at frame 10 (t=0.2)
    eng = Engine()
    eng.evaluate(0.0, "stand", Triggers(clips=[c]))
    f1 = eng.evaluate(0.30, "stand").show.events  # crosses frame 10
    f2 = eng.evaluate(0.10, "stand").show.events  # backwards → no fire, no rewind
    f3 = eng.evaluate(0.30, "stand").show.events  # forward again → already fired
    assert len(f1) == 1
    assert len(f2) == 0
    assert len(f3) == 0


def test_zero_dt_fires_nothing():
    # M1: a repeated timestamp (dt==0) must not re-fire an already-fired event.
    c = _events_clip("once")
    eng = Engine()
    eng.evaluate(0.0, "stand", Triggers(clips=[c]))
    eng.evaluate(0.30, "stand")  # fire frame 10
    f = eng.evaluate(0.30, "stand").show.events  # same t
    assert f == []


def test_background_clip_events_fire():
    # M2: the always-on background loop must fire its events every loop.
    bg = make_clip(
        loop_mode="wrap",
        move_head=False,
        events=[{"frame": 5, "type": "sound", "value": "bg.wav"}],
        blend_in_s=0.0,
        blend_out_s=0.0,
    )
    eng = Engine(background=bg)
    eng.evaluate(0.0, "stand")  # anchor background phase at t=0
    fired = []
    t = 0.0
    while t < 1.0:  # 2.5 loops; event at cum 5, 25, 45 → 3 fires
        t += 0.02
        fired.extend(eng.evaluate(t, "stand").show.events)
    assert len(fired) == 3
    assert all(e.value == "bg.wav" for e in fired)
