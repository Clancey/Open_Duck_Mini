#!/usr/bin/env python3
"""verify_demo.py -- Open Duck Mini v2 visual verification demo.

Purpose-built to let a human confirm by eye the two things software cannot
self-check after the hardware bring-up:

  1. The eye LEDs actually ILLUMINATE and BLINK naturally.
  2. The physical antenna LEFT/RIGHT wiring matches the (confirmed-correct)
     software mapping -- i.e. the antennas are not physically swapped.

Hardware touched (electrically trivial -- no torque, no fall/thermal risk):
  * Eyes    : GPIO D24 (LEFT) / D23 (RIGHT), digital on/off, lit == "open".
  * Antennas: PWM   D13 (LEFT,  sign +1) / D12 (RIGHT, sign -1).
NO servo bus, NO legs, NO head are touched by this script.

============================  LEFT / RIGHT  ============================
Every "LEFT"/"RIGHT" in this demo means the ROBOT'S OWN left/right -- as if
YOU were the duck. When you stand FACING the robot, the robot's LEFT antenna
is on YOUR right-hand side, and the robot's RIGHT antenna is on YOUR left.
Pick a side by the robot's body, not by your view.
=======================================================================

Phases:
  1  EYES ON, STEADY            -- do they light at all?
  2  EYES BLINK (counted) + natural idle blink for a while
  3  LEFT antenna ONLY moves    -- right held dead still
  4  RIGHT antenna ONLY moves   -- left held dead still
  5  BOTH, alternating L,R,L,R  -- final confirmation

Re-runnable. Ctrl-C safe at any point. `--phase N` repeats one phase.
On any exit (normal, Ctrl-C, SIGTERM, SIGHUP, atexit) the eyes are turned OFF
and the antennas are returned to neutral and de-initialised (pins released).
"""

import argparse
import atexit
import math
import signal
import sys
import time

RUNTIME_PKG = "/home/clancey/duck/Open_Duck_Mini_Runtime/mini_bdx_runtime"
if RUNTIME_PKG not in sys.path:
    sys.path.insert(0, RUNTIME_PKG)

from mini_bdx_runtime.antennas import (  # noqa: E402
    Antennas,
    value_to_duty_cycle,
    LEFT_SIGN,
    RIGHT_SIGN,
    LEFT_ANTENNA_PIN,
    RIGHT_ANTENNA_PIN,
)
from mini_bdx_runtime.eyes import Eyes, LEFT_EYE_PIN, RIGHT_EYE_PIN  # noqa: E402

UPDATE_HZ = 50
DT = 1.0 / UPDATE_HZ

# Live hardware handles (module-level so signal/atexit handlers can reach them).
_eyes = None
_antennas = None
_cleaning = False


# --------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------
def _p(*a):
    print(*a, flush=True)


def banner(title, *sub):
    bar = "=" * 72
    _p("")
    _p(bar)
    _p("  " + title)
    for s in sub:
        _p("  " + s)
    _p(bar)


def note(msg):
    _p("    - " + msg)


def countdown_pause(seconds, next_desc):
    """A visible, quiet gap between phases -- nothing moves. No stdin needed
    (this may run detached), so the pause is timed, not a keypress."""
    _p("")
    _p("    ...... PAUSE %ds (everything still) -> next: %s" % (seconds, next_desc))
    for r in range(seconds, 0, -1):
        _p("        %d" % r)
        time.sleep(1.0)


# --------------------------------------------------------------------------
# Cleanup / safety
# --------------------------------------------------------------------------
def cleanup(reason="normal"):
    global _eyes, _antennas, _cleaning
    if _cleaning:
        return
    _cleaning = True
    _p("")
    _p("[cleanup] reason=%s -- eyes OFF, antennas neutral + deinit" % reason)
    if _eyes is not None:
        try:
            _eyes.stop()  # sets LEDs dark + deinit()s the pins
            note("eyes: stopped (LEDs off, pins released)")
        except Exception as e:  # noqa: BLE001
            note("eyes: stop error: %r" % e)
        _eyes = None
    if _antennas is not None:
        try:
            _antennas.stop()  # neutral, then deinit() both PWM pins
            note("antennas: neutral + deinit (pins released, passive)")
        except Exception as e:  # noqa: BLE001
            note("antennas: stop error: %r" % e)
        _antennas = None


def _signal_handler(signum, _frame):
    name = signal.Signals(signum).name
    _p("")
    _p("[signal] caught %s -- cleaning up and exiting" % name)
    cleanup("signal:%s" % name)
    # Re-raise default behaviour so the exit code reflects the signal.
    sys.exit(128 + signum)


def install_handlers():
    atexit.register(lambda: cleanup("atexit"))
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _signal_handler)
        except (ValueError, OSError):
            pass  # e.g. not in main thread; atexit still covers us


# --------------------------------------------------------------------------
# Hardware setup (lazy -- only build what the requested phase needs)
# --------------------------------------------------------------------------
def ensure_eyes():
    global _eyes
    if _eyes is None:
        note("constructing Eyes()  (LEFT=%s  RIGHT=%s) -> LEDs light + idle-blink thread starts"
             % (LEFT_EYE_PIN, RIGHT_EYE_PIN))
        _eyes = Eyes()
        # Suppress the background idle blink until we explicitly want it, so the
        # "steady" and "counted" sub-phases are unambiguous.
        _idle_off(_eyes)
    return _eyes


def _idle_off(eyes):
    """Push the automatic idle blink far into the future (deterministic phases)."""
    with eyes._lock:
        eyes._next_auto = time.monotonic() + 1e9


def _idle_on(eyes, min_interval=2.0, max_interval=6.0, double_prob=0.18):
    """(Re-)enable the natural randomised idle blink."""
    eyes.min_interval = min_interval
    eyes.max_interval = max_interval
    eyes.double_blink_prob = double_prob
    with eyes._lock:
        eyes._blink_requests = 0
        eyes._hold_until = 0.0
        eyes._next_auto = time.monotonic() + eyes._rand_interval()


def stop_eyes():
    global _eyes
    if _eyes is not None:
        try:
            _eyes.stop()
            note("eyes stopped (LEDs OFF, pins released) -- so eye motion cannot be confused with antenna motion")
        except Exception as e:  # noqa: BLE001
            note("eyes stop error: %r" % e)
        _eyes = None


def ensure_antennas():
    global _antennas
    if _antennas is None:
        note("constructing Antennas() -> both PWM pins driven to NEUTRAL (a one-time settle of both)")
        note("  LEFT  antenna = %s  sign %+d   |   RIGHT antenna = %s  sign %+d"
             % (LEFT_ANTENNA_PIN, LEFT_SIGN, RIGHT_ANTENNA_PIN, RIGHT_SIGN))
        _antennas = Antennas()
        _report_duty("after neutral init")
        time.sleep(0.5)
    return _antennas


def stop_antennas():
    global _antennas
    if _antennas is not None:
        try:
            _antennas.stop()
            note("antennas neutral + deinit (pins released, passive)")
        except Exception as e:  # noqa: BLE001
            note("antennas stop error: %r" % e)
        _antennas = None


def _report_duty(tag):
    if _antennas is None:
        return
    try:
        dl = _antennas.pwm_left.duty_cycle
        dr = _antennas.pwm_right.duty_cycle
        note("duty[%s]: LEFT(D13)=%d  RIGHT(D12)=%d  (neutral=%d)"
             % (tag, dl, dr, value_to_duty_cycle(0)))
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------
# Antenna motion primitives
# --------------------------------------------------------------------------
def _hold_seconds(sec):
    end = time.monotonic() + sec
    while time.monotonic() < end:
        time.sleep(DT)


def sweep_one(side, cycles=4, amp=0.8, period=1.4):
    """Smoothly oscillate ONE antenna through a full sine for `cycles`, leaving
    the OTHER completely untouched (it keeps whatever it was last set to --
    neutral). Reports the commanded value range and confirms the still side's
    PWM duty never changes."""
    ant = _antennas
    if side == "left":
        drive = ant.set_position_left
        still_pwm = ant.pwm_right
        moving_lbl, still_lbl = "LEFT (D13)", "RIGHT (D12)"
    else:
        drive = ant.set_position_right
        still_pwm = ant.pwm_left
        moving_lbl, still_lbl = "RIGHT (D12)", "LEFT (D13)"

    still_before = still_pwm.duty_cycle
    vmin, vmax = 1.0, -1.0
    n = int(round(cycles * period / DT))
    for i in range(n + 1):
        t = i * DT
        v = amp * math.sin(2 * math.pi * t / period)
        drive(v)
        vmin = min(vmin, v)
        vmax = max(vmax, v)
        time.sleep(DT)
    drive(0.0)  # return the moving one to neutral
    time.sleep(DT)
    still_after = still_pwm.duty_cycle
    note("commanded %s over [%+.2f, %+.2f] for %d cycles (~%.1fs)"
         % (moving_lbl, vmin, vmax, cycles, cycles * period))
    note("%s held STILL: duty %d -> %d  (delta %d -- should be 0)"
         % (still_lbl, still_before, still_after, still_after - still_before))


# --------------------------------------------------------------------------
# Phases
# --------------------------------------------------------------------------
def phase1(steady_secs=4.0):
    banner("PHASE 1 / 5  --  EYES SHOULD NOW BE ON, STEADY.",
           "Look at the robot's eyes: BOTH should be lit and holding steady (no blink).",
           "This alone settles whether the LEDs illuminate at all.")
    eyes = ensure_eyes()
    eyes.hold_open(steady_secs + 0.5)  # suppress any blink, hold wide
    note("commanded: eyes HELD OPEN (lit) for %.1fs" % steady_secs)
    note("eye state now: LEFT=%s RIGHT=%s (lit=%s)"
         % (eyes.left_eye.value, eyes.right_eye.value, eyes._lit))
    _hold_seconds(steady_secs)
    note("PASS looks like: both eyes were clearly ON and unchanging for the whole %ds." % int(steady_secs))


def phase2(blinks=3, idle_secs=25.0):
    banner("PHASE 2 / 5  --  EYES BLINK (counted), then NATURAL idle blink.",
           "First: %d deliberate, counted blinks -- each announced as it fires." % blinks,
           "Then: the natural idle blink for ~%ds (random 2-6s, ~18%% double-blinks)." % int(idle_secs))
    eyes = ensure_eyes()
    _idle_off(eyes)                 # deterministic counted blinks, no idle interference
    eyes.hold_open(0.4)
    _hold_seconds(0.6)
    for i in range(1, blinks + 1):
        _p("        >>> BLINK %d of %d ..." % (i, blinks))
        eyes.blink()
        _hold_seconds(1.0)          # each blink stands alone, clearly counted
    note("counted-blink phase done: commanded %d discrete blinks (0.12s dark flick each)." % blinks)

    _p("")
    _p("    Now watch the NATURAL idle blink for ~%ds -- it should read ALIVE, not metronomic." % int(idle_secs))
    _idle_on(eyes)                  # randomised 2-6s interval, ~18% double
    end = time.monotonic() + idle_secs
    last = None
    while time.monotonic() < end:
        # Report each dark flick we observe, so the log shows blinks really fired.
        lit = eyes._lit
        if last is True and lit is False:
            _p("        (idle blink at t=%+.1fs)" % (idle_secs - (end - time.monotonic())))
        last = lit
        time.sleep(0.01)
    _idle_off(eyes)
    eyes.hold_open(0.3)
    note("PASS looks like: 3 clear counted blinks, then irregular lifelike blinks (some doubles), never a steady metronome.")


def phase3(cycles=4):
    banner("PHASE 3 / 5  --  MOVING **LEFT** ANTENNA ONLY.",
           "LEFT = the ROBOT'S OWN left (facing the robot, that is on YOUR right).",
           "ONLY the left antenna should move; the RIGHT one must stay dead still.")
    ensure_antennas()
    for rep in range(1, 3):
        _p("        >>> LEFT-only sweep, pass %d of 2 ..." % rep)
        sweep_one("left", cycles=cycles)
        _hold_seconds(0.6)
    note("PASS looks like: the robot's LEFT antenna (your right) waved several times; the right never twitched.")
    note("If the RIGHT one moved instead -> antennas are PHYSICALLY SWAPPED (see VERIFY.md fix).")


def phase4(cycles=4):
    banner("PHASE 4 / 5  --  MOVING **RIGHT** ANTENNA ONLY.",
           "RIGHT = the ROBOT'S OWN right (facing the robot, that is on YOUR left).",
           "ONLY the right antenna should move; the LEFT one must stay dead still.")
    ensure_antennas()
    for rep in range(1, 3):
        _p("        >>> RIGHT-only sweep, pass %d of 2 ..." % rep)
        sweep_one("right", cycles=cycles)
        _hold_seconds(0.6)
    note("PASS looks like: the robot's RIGHT antenna (your left) waved several times; the left never twitched.")
    note("If the LEFT one moved instead -> antennas are PHYSICALLY SWAPPED (see VERIFY.md fix).")


def phase5(rounds=2):
    banner("PHASE 5 / 5  --  BOTH antennas, ALTERNATING  L, R, L, R.",
           "Final confirmation. Announced each time. LEFT = robot's own left (your right).")
    ensure_antennas()
    for r in range(1, rounds + 1):
        _p("        >>> round %d: LEFT (robot's left / your right) ..." % r)
        sweep_one("left", cycles=2)
        _hold_seconds(0.4)
        _p("        >>> round %d: RIGHT (robot's right / your left) ..." % r)
        sweep_one("right", cycles=2)
        _hold_seconds(0.4)
    note("PASS looks like: left, right, left, right -- alternation matched every announcement.")


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
EYE_PHASES = {1, 2}
ANT_PHASES = {3, 4, 5}


def run_all(args):
    banner("OPEN DUCK MINI v2 -- VISUAL VERIFICATION DEMO",
           "Convention: LEFT/RIGHT = the ROBOT'S OWN sides (facing it, mirror them).",
           "Checks: (1) eyes illuminate + blink,  (2) antenna L/R wiring not swapped.",
           "Eyes and antennas are shown SEPARATELY so each phase tests exactly one thing.")
    _p("    (Ctrl-C is safe at any time: eyes go off, antennas return to neutral + release.)")
    time.sleep(1.0)

    # ---- Eye phases -----------------------------------------------------
    phase1(steady_secs=args.steady_secs)
    countdown_pause(3, "PHASE 2 (eyes blink)")
    phase2(blinks=args.blinks, idle_secs=args.idle_secs)

    # Transition: eyes fully OFF before any antenna moves, so the two can never
    # be visually confused.
    banner("TRANSITION -- eyes OFF; switching to ANTENNAS.",
           "The eyes are now dark on purpose. From here ONLY antennas move.")
    stop_eyes()
    countdown_pause(3, "PHASE 3 (LEFT antenna only)")

    # ---- Antenna phases -------------------------------------------------
    phase3(cycles=args.cycles)
    countdown_pause(3, "PHASE 4 (RIGHT antenna only)")
    phase4(cycles=args.cycles)
    countdown_pause(3, "PHASE 5 (both, alternating)")
    phase5(rounds=args.rounds)

    banner("DEMO COMPLETE.",
           "Leaving robot passive: eyes OFF, antennas neutral + released, nothing torqued.")


def run_single(args):
    n = args.phase
    if n in EYE_PHASES:
        if n == 1:
            phase1(steady_secs=args.steady_secs)
        else:
            phase2(blinks=args.blinks, idle_secs=args.idle_secs)
        stop_eyes()
    elif n in ANT_PHASES:
        if n == 3:
            phase3(cycles=args.cycles)
        elif n == 4:
            phase4(cycles=args.cycles)
        else:
            phase5(rounds=args.rounds)
        stop_antennas()
    else:
        _p("Unknown phase %r (valid: 1-5)" % n)
        return 2
    banner("PHASE %d COMPLETE." % n, "Robot passive: eyes OFF, antennas neutral + released.")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Open Duck Mini v2 eyes + antennas visual verification demo.")
    ap.add_argument("--phase", type=int, default=None,
                    help="Run ONLY this phase (1-5) and exit. Default: run all in order.")
    ap.add_argument("--steady-secs", type=float, default=4.0,
                    help="Phase 1 steady-on duration (default 4).")
    ap.add_argument("--blinks", type=int, default=3,
                    help="Phase 2 counted-blink count (default 3).")
    ap.add_argument("--idle-secs", type=float, default=25.0,
                    help="Phase 2 natural-idle-blink duration (default 25).")
    ap.add_argument("--cycles", type=int, default=4,
                    help="Antenna sweep cycles per pass in phases 3/4 (default 4).")
    ap.add_argument("--rounds", type=int, default=2,
                    help="Phase 5 L/R alternation rounds (default 2).")
    args = ap.parse_args(argv)

    install_handlers()
    rc = 0
    try:
        if args.phase is None:
            run_all(args)
        else:
            rc = run_single(args)
    except KeyboardInterrupt:
        _p("")
        _p("[interrupt] KeyboardInterrupt -- cleaning up")
        rc = 130
    finally:
        cleanup("finally")
    return rc


if __name__ == "__main__":
    sys.exit(main())
