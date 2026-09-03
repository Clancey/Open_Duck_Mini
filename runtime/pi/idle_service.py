#!/usr/bin/env python3
"""Open Duck Mini v2 - boot-time IDLE "alive" service.

Plays the idle loops continuously through the REAL three-layer open_duck_anim
engine so the robot comes alive on its own at power-on: head moving gently,
eyes blinking. Designed to run UNATTENDED on every boot, with nobody watching.

SAFETY (the whole point - read scripts/../duck_bringup_log.md for context):
  * HEAD + SHOW ONLY.  The 10 leg servos (10-14, 20-24) are NEVER torqued by
    this service.  No dock hold, no stand, no policy inference.  If the robot is
    picked up / knocked / sitting oddly at boot, nothing fights it - legs limp.
  * Torque off cleanly on stop.  SIGTERM/SIGHUP/SIGINT/atexit all return the head
    gently to its measured rest pose, disable head torque, neutralise+release the
    antennas and turn the eyes off.  A raw-Feetech belt-and-suspenders torque-off
    runs if anything is left energised.
  * Stays inside the x0.5 derated safety envelope (open_duck_anim enforced).
  * Thermal duty-cycle guard: every ACTIVE_S of animation the head is eased to
    rest and FULLY DE-ENERGISED for RELAX_S while a raw-Feetech thermal/voltage/
    error scan is taken and logged, then it re-arms and resumes.  This both gives
    real thermal telemetry (rustypot 0.1.0 cannot read temp inline) and relieves
    the head servos over an hours-long run.  Abort thresholds -> clean passive exit.
  * Fail safe, not loud.  Missing serial bus, a clip that will not load, or any
    exception at startup -> exit cleanly leaving hardware passive.  It never spins
    retrying servo writes.  systemd Restart=on-failure + StartLimitBurst caps a
    persistent fault instead of thrashing.
  * Waits for the USB serial device (by-id path) at boot before doing anything.

Head servos: neck_pitch=30 head_pitch=31 head_yaw=32 head_roll=33, kp=8 soft.
"""
import os
import sys
import gc
import time
import glob
import signal
import atexit
import random
import logging

import numpy as np

# --- runtime import path (open_duck_anim is pip-installed; runtime is on disk) --
RUNTIME = "/home/clancey/duck/Open_Duck_Mini_Runtime/mini_bdx_runtime"
if RUNTIME not in sys.path:
    sys.path.insert(0, RUNTIME)
import open_duck_anim as oda  # noqa: E402

# The mood planner (pure logic) lives beside this file so the robot imports one
# copy and it stays unit-testable off the robot (tests/test_idle_moods.py).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import idle_moods  # noqa: E402


# ----------------------------- configuration ---------------------------------
def _envf(name, default):
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return float(default)


def _envb(name, default):
    return (os.environ.get(name, str(default)).strip().lower()
            in ("1", "true", "yes", "on"))


PORT = os.environ.get(
    "DUCK_IDLE_PORT",
    "/dev/serial/by-id/usb-1a86_USB_Single_Serial_58FA095764-if00")
CLIPS = os.path.expanduser(os.environ.get("DUCK_IDLE_CLIPS", "~/duck/clips"))

HEAD_IDS = [30, 31, 32, 33]
HEAD_NAMES = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]
LEG_IDS = [10, 11, 12, 13, 14, 20, 21, 22, 23, 24]
ALL_IDS = LEG_IDS[:5] + HEAD_IDS + LEG_IDS[5:]

KP = _envf("DUCK_IDLE_KP", 8.0)
DT = 0.02                       # 50 Hz control period
DERATING = _envf("DUCK_IDLE_DERATING", 0.5)

BAUD = 1_000_000

# Neutral-baseline background idle clip rotation (so a plain-neutral stretch does
# not look identical for hours). While in a MOOD the background is that mood's
# loop, held for the whole mood dwell; only NEUTRAL rotates the idle_* loops.
BG_MIN_S = _envf("DUCK_IDLE_BG_MIN_S", 45.0)
BG_MAX_S = _envf("DUCK_IDLE_BG_MAX_S", 95.0)
XFADE_S = _envf("DUCK_IDLE_XFADE_S", 0.6)   # smooth background handoff

# Mood-drift + one-shot pacing (env-overridable; see idle_moods for the model).
MOOD_DWELL_MIN_S = _envf("DUCK_IDLE_MOOD_MIN_S", idle_moods.MOOD_DWELL_MIN_S)
MOOD_DWELL_MAX_S = _envf("DUCK_IDLE_MOOD_MAX_S", idle_moods.MOOD_DWELL_MAX_S)
BEAT_MIN_S = _envf("DUCK_IDLE_BEAT_MIN_S", idle_moods.BEAT_MIN_S)
BEAT_MAX_S = _envf("DUCK_IDLE_BEAT_MAX_S", idle_moods.BEAT_MAX_S)
QUIET_PROB = _envf("DUCK_IDLE_QUIET_PROB", idle_moods.QUIET_PROB)

# Thermal / duty-cycle guard
ACTIVE_S = _envf("DUCK_IDLE_ACTIVE_S", 480.0)   # animate before a relax checkpoint
RELAX_S = _envf("DUCK_IDLE_RELAX_S", 20.0)      # head de-energised + thermal scan
WARN_TEMP = _envf("DUCK_IDLE_WARN_TEMP", 48.0)
ABORT_TEMP = _envf("DUCK_IDLE_ABORT_TEMP", 55.0)
WARN_VOLT = _envf("DUCK_IDLE_WARN_VOLT", 7.2)
ABORT_VOLT = _envf("DUCK_IDLE_ABORT_VOLT", 7.0)

SERIAL_WAIT_S = _envf("DUCK_IDLE_SERIAL_WAIT_S", 30.0)
# Antennas are "very noisy" (owner) -> OFF by default for unattended desk use.
# The head+eyes show layer is always played faithfully; this only gates whether
# the antenna show channel is energised.  No antenna motion is ever hardcoded.
ANTENNAS = _envb("DUCK_IDLE_ANTENNAS", False)
MAX_RUN_S = _envf("DUCK_IDLE_MAX_RUN_S", 0.0)   # 0 = run forever (service default)

# Candidate clips (kept only if present + loadable). Idle backgrounds are the
# NEUTRAL baseline (loop=wrap, runnable in DOCK). Mood loops are the emotional
# backgrounds the duck sits in. One-shots are the congruent gestures the planner
# fires; only calm, HEAD-ONLY clips are ever scheduled unprompted (the big
# reactions and the full-body dock_wiggle are excluded in idle_moods and, belt &
# braces, re-checked at trigger time by _trigger_is_safe below).
IDLE_CANDIDATES = ["idle_alive", "idle_breathe", "idle_lookaround", "idle_scan"]
MOOD_LOOP_CANDIDATES = [c for c in idle_moods.MOOD_LOOPS.values() if c]
ONESHOT_CANDIDATES = sorted(
    {c for pool in idle_moods.CONGRUENT.values() for c in pool}
)

log = logging.getLogger("duck-idle")


def _trigger_is_safe(clip):
    """Hard runtime gate: this HEAD-ONLY service must NEVER trigger a clip that
    moves the legs or is otherwise reserved. Asserts the exclusion in code (not
    by omission) so a mis-scheduled clip cannot energise the legs of a docked
    robot. Returns True iff ``clip`` is safe to fire unprompted."""
    if clip.name in idle_moods.EXCLUDED_UNPROMPTED:
        return False
    if getattr(clip, "layer_mask", None) != "head":
        return False
    if getattr(clip, "requires_mode", None) == "dock":
        # dock == full-body/leg animation here (dock_wiggle). Never unprompted.
        return False
    return True


# ----------------------------- raw Feetech I/O --------------------------------
# Version-independent (rustypot 0.1.0 exposes no temp/voltage).  Protocol 1.0.
def _cksum(vals):
    return (~sum(vals)) & 0xFF


def _read_reg(ser, sid, addr, nbytes):
    pkt = [0xFF, 0xFF, sid, 0x04, 0x02, addr, nbytes]
    pkt.append(_cksum(pkt[2:]))
    ser.reset_input_buffer()
    ser.write(bytes(pkt))
    want = 6 + nbytes
    resp = ser.read(want)
    if len(resp) < want or resp[0] != 0xFF or resp[1] != 0xFF or resp[2] != sid:
        return None, None
    return resp[4], list(resp[5:5 + nbytes])


def _write_reg1(ser, sid, addr, value):
    pkt = [0xFF, 0xFF, sid, 0x04, 0x03, addr, value & 0xFF]
    pkt.append(_cksum(pkt[2:]))
    ser.write(bytes(pkt))
    time.sleep(0.003)
    ser.reset_input_buffer()


def _open_serial(port, timeout=0.05, tries=8, delay=0.25):
    """Open a pyserial handle, retrying briefly on EBUSY (rustypot releases the
    port asynchronously when its Rust handle is dropped)."""
    import serial
    last = None
    for _ in range(max(1, tries)):
        try:
            return serial.Serial(port, BAUD, timeout=timeout)
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(delay)
    raise last


def _open_feetech(tries=10, delay=0.3):
    """Open the rustypot Feetech bus, retrying briefly on EBUSY."""
    import rustypot
    last = None
    for _ in range(max(1, tries)):
        try:
            return rustypot.feetech(PORT, BAUD)
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(delay)
    raise last


def raw_torque_off(ids=ALL_IDS, port=PORT):
    """Best-effort: disable torque (reg 40 = 0) on the given servos via pyserial.
    Safe to call any time; used as the ultimate passivity backstop."""
    try:
        ser = _open_serial(port, timeout=0.1)
    except Exception as e:  # noqa: BLE001
        log.warning("raw_torque_off: cannot open %s: %s", port, e)
        return False
    try:
        time.sleep(0.05)
        for sid in ids:
            try:
                _write_reg1(ser, sid, 40, 0)
            except Exception:  # noqa: BLE001
                pass
    finally:
        ser.close()
    return True


def thermal_scan(tag, port=PORT):
    """Read voltage(62)/temp(63)/torque(40)/error on all 14 servos. Returns a
    dict + logs one compact line. Requires the rustypot handle to be released."""
    by = {}
    try:
        ser = _open_serial(port, timeout=0.05)
    except Exception as e:  # noqa: BLE001
        log.warning("thermal_scan[%s]: cannot open port: %s", tag, e)
        return None
    try:
        time.sleep(0.05)
        for sid in ALL_IDS:
            ev, v = _read_reg(ser, sid, 62, 1)
            et, t = _read_reg(ser, sid, 63, 1)
            eq, q = _read_reg(ser, sid, 40, 1)
            err = 0
            for e in (ev, et, eq):
                if e:
                    err |= e
            by[sid] = dict(volt=(v[0] / 10.0 if v else None),
                           temp=(t[0] if t else None),
                           torque=(q[0] if q else None), err=err)
    finally:
        ser.close()
    temps = [by[i]["temp"] for i in ALL_IDS if by.get(i, {}).get("temp") is not None]
    volts = [by[i]["volt"] for i in ALL_IDS if by.get(i, {}).get("volt") is not None]
    errs = [(i, by[i]["err"]) for i in ALL_IDS if by.get(i, {}).get("err")]
    leg_on = [i for i in LEG_IDS if by.get(i, {}).get("torque")]
    head_on = [i for i in HEAD_IDS if by.get(i, {}).get("torque")]
    tmin = min(temps) if temps else None
    tmax = max(temps) if temps else None
    vmin = min(volts) if volts else None
    vmax = max(volts) if volts else None
    abort = False
    reasons = []
    if tmax is not None and tmax >= ABORT_TEMP:
        abort = True
        reasons.append("temp %s>=%s" % (tmax, ABORT_TEMP))
    if vmin is not None and vmin <= ABORT_VOLT:
        abort = True
        reasons.append("volt %.1f<=%.1f" % (vmin, ABORT_VOLT))
    if leg_on:
        abort = True
        reasons.append("LEG torque ON %s" % leg_on)
    if errs:
        abort = True
        reasons.append("servo err %s" % errs)
    lvl = logging.ERROR if abort else (
        logging.WARNING if (tmax and tmax >= WARN_TEMP) or (vmin and vmin <= WARN_VOLT)
        else logging.INFO)
    log.log(lvl, "thermal[%s] temp=%s-%s C volt=%s-%s V head_torque=%s legs_torque=%s errs=%s%s",
            tag, tmin, tmax, vmin, vmax, head_on or "off", leg_on or "off",
            errs or "none", ("  ABORT: " + "; ".join(reasons)) if abort else "")
    return dict(tag=tag, tmin=tmin, tmax=tmax, vmin=vmin, vmax=vmax,
                errs=errs, leg_on=leg_on, head_on=head_on, abort=abort)


# ------------------------------ show drivers ----------------------------------
class EyeDriver:
    """Background irregular blink + routes the engine eye channel/events.

    Understands the full cue vocabulary of the updated runtime eyes (branch
    feature/animation-engine): the sustained wide/fear hold and its release, and
    the slow heavy blink. Every call is guarded so an older Eyes build (without
    these methods) degrades to a plain blink instead of crashing the service."""

    def __init__(self):
        from mini_bdx_runtime.eyes import Eyes
        self.eyes = Eyes()          # lit + background blink thread
        self._last = None

    def _call(self, name, *args):
        fn = getattr(self.eyes, name, None)
        if callable(fn):
            try:
                fn(*args)
                return True
            except Exception:  # noqa: BLE001
                pass
        return False

    def slow_blink(self):
        """One long, heavy lid close/open (sleepy/content/sad). Falls back to a
        normal blink on an older Eyes build."""
        if not self._call("slow_blink"):
            self._call("blink")

    def tick(self, show):
        try:
            self.eyes.note_authored(int(getattr(show, "eyes", 1) or 0))
        except Exception:  # noqa: BLE001
            pass
        evs = [(e.type, e.value) for e in getattr(show, "events", [])
               if getattr(e, "type", None) == "eye"]
        if evs and evs != self._last:
            for _, val in evs:
                v = str(val).lower()
                if v in ("fear", "wide_hold", "cower", "afraid"):
                    if not self._call("enter_wide_hold"):
                        self._call("hold_open", 1.0)
                elif v in ("release", "wide_release", "fear_release", "relief",
                           "calm"):
                    self._call("release_wide_hold")
                elif v in ("slow_blink", "sleepy", "slow"):
                    self.slow_blink()
                elif v == "squint":
                    pass                 # binary LEDs cannot squint: explicit no-op
                elif v in ("wide", "startle", "alert", "open"):
                    self._call("hold_open", 1.0)
                elif v in ("happy", "double", "double_blink"):
                    self._call("double_blink")
                else:
                    self._call("blink")
        self._last = evs

    def stop(self):
        try:
            self.eyes.stop()
        except Exception:  # noqa: BLE001
            pass


class AntennaDriver:
    """Faithful antenna show playback, lazily energised on first non-neutral
    command (consent design).  Disabled entirely when ANTENNAS is False, so the
    pins stay dark and silent."""

    def __init__(self, enabled):
        self.enabled = enabled
        self.ant = None

    def tick(self, show):
        if not self.enabled:
            return
        al = float(getattr(show, "antenna_l", 0.0) or 0.0)
        ar = float(getattr(show, "antenna_r", 0.0) or 0.0)
        if self.ant is None:
            if abs(al) < 1e-3 and abs(ar) < 1e-3:
                return
            from mini_bdx_runtime.antennas import Antennas
            self.ant = Antennas()
            log.info("antennas energised (first non-neutral show command)")
        self.ant.set_position_left(max(-1.0, min(1.0, al)))
        self.ant.set_position_right(max(-1.0, min(1.0, ar)))

    def stop(self):
        try:
            if self.ant is not None:
                self.ant.stop()
        except Exception:  # noqa: BLE001
            pass


# ------------------------- global cleanup state -------------------------------
_IO = None          # live rustypot handle (None when port released)
_EYED = None
_ANTD = None
_REST = None        # measured limp rest pose captured at startup
_TORQUED = False    # is head torque currently enabled?
_CLEANED = False
_STOP = False


def _ramp(io, start, goal, dur=1.2, step_dt=0.04):
    start = np.asarray(start, float)
    goal = np.asarray(goal, float)
    n = max(1, int(dur / step_dt))
    for i in range(1, n + 1):
        a = i / n
        cur = (1 - a) * start + a * goal
        io.write_goal_position(HEAD_IDS, [float(x) for x in cur])
        time.sleep(step_dt)


def graceful_shutdown(reason="exit"):
    """Idempotent: gently return the head to rest, torque off, stop the show,
    then a raw-Feetech backstop.  Safe from a signal handler (Python delivers
    signals on the main thread, so no concurrent rustypot access)."""
    global _CLEANED, _IO, _TORQUED
    if _CLEANED:
        return
    _CLEANED = True
    log.info("shutdown (%s): returning head to rest + torque off", reason)
    io = _IO
    disabled_ok = False
    if io is not None:
        try:
            if _TORQUED and _REST is not None:
                cur = io.read_present_position(HEAD_IDS)
                _ramp(io, cur, _REST, dur=1.2)   # gentle, no drop
        except Exception as e:  # noqa: BLE001
            log.warning("shutdown ramp err: %s", e)
        try:
            io.disable_torque(HEAD_IDS)
            _TORQUED = False
            disabled_ok = True
            log.info("head torque disabled")
        except Exception as e:  # noqa: BLE001
            log.warning("shutdown disable_torque err: %s", e)
        _IO = None
        try:
            del io
        except Exception:  # noqa: BLE001
            pass
        gc.collect()
        time.sleep(0.2)
    # Backstop: only force a raw torque-off if the rustypot disable did not run
    # (avoids a spurious EBUSY when a live handle already disabled torque cleanly).
    if not disabled_ok:
        raw_torque_off()
    if _ANTD is not None:
        _ANTD.stop()
    if _EYED is not None:
        _EYED.stop()
    log.info("shutdown complete - hardware passive")


def _on_signal(signum, _frame):
    global _STOP
    _STOP = True
    log.info("signal %s received -> stopping", signal.Signals(signum).name
             if hasattr(signal, "Signals") else signum)
    # Cleanup runs on the main thread (safe for rustypot). If we were blocked in
    # a long sleep this returns after the handler; the main loop also checks _STOP.
    graceful_shutdown("signal %s" % signum)
    sys.exit(0)


# ------------------------------- helpers --------------------------------------
def wait_for_serial(port, timeout):
    """Return True once the by-id device exists and opens, else False."""
    deadline = time.monotonic() + timeout
    announced = False
    while True:
        if os.path.exists(port):
            try:
                s = _open_serial(port, timeout=0.05, tries=1)
                s.close()
                return True
            except Exception as e:  # noqa: BLE001
                if not announced:
                    log.info("serial present but busy/not-ready (%s); waiting...", e)
                    announced = True
        elif not announced:
            log.info("serial device %s not present yet; waiting up to %.0fs", port, timeout)
            announced = True
        if time.monotonic() >= deadline:
            return False
        time.sleep(1.0)


def load_present(names):
    out = []
    for n in names:
        p = os.path.join(CLIPS, n + ".duckanim")
        if not os.path.exists(p):
            continue
        try:
            out.append(oda.load_clip(p))
        except Exception as e:  # noqa: BLE001
            log.warning("clip %s failed to load: %s", n, e)
    return out


def new_engine(bg_clip):
    return oda.Engine(background=bg_clip,
                      head_envelope=oda.DEFAULT_ENVELOPE.derated(DERATING))


def arm(io):
    """Set soft kp, capture rest, hold present (no jump), enable torque, ease to
    a neutral upright pose.  Returns True on success."""
    global _TORQUED
    io.set_kps(HEAD_IDS, [KP] * 4)
    present = io.read_present_position(HEAD_IDS)
    io.write_goal_position(HEAD_IDS, [float(x) for x in present])
    io.enable_torque(HEAD_IDS)
    _TORQUED = True
    time.sleep(0.2)
    chk = io.read_present_position(HEAD_IDS)
    jump = float(np.max(np.abs(np.asarray(chk) - np.asarray(present))))
    log.info("arm: torque-on jump max=%.4f rad; easing to neutral", jump)
    _ramp(io, present, [0.0, 0.0, 0.0, 0.0], dur=1.5)
    return True


# --------------------------------- main ---------------------------------------
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout)
    log.info("=== duck-idle service starting ===")
    log.info("port=%s clips=%s kp=%.0f derating=%.2f antennas=%s",
             PORT, CLIPS, KP, DERATING, ANTENNAS)
    log.info("mood dwell=%.0f-%.0fs beats=%.0f-%.0fs quiet=%.2f | duty active=%.0fs relax=%.0fs",
             MOOD_DWELL_MIN_S, MOOD_DWELL_MAX_S, BEAT_MIN_S, BEAT_MAX_S,
             QUIET_PROB, ACTIVE_S, RELAX_S)

    # 1) Wait for the serial bus (USB may not be ready at boot).
    if not wait_for_serial(PORT, SERIAL_WAIT_S):
        log.error("serial device %s not available after %.0fs -> exiting (passive)",
                  PORT, SERIAL_WAIT_S)
        return 1

    # 2) Load clips (fail safe if none). Neutral idle backgrounds are mandatory;
    #    mood loops + congruent one-shots are the emotional palette (best-effort:
    #    whatever is deployed is used, the planner adapts to what's present).
    idle_bgs = load_present(IDLE_CANDIDATES)
    if not idle_bgs:
        log.error("no idle background clips loadable in %s -> exiting (passive)", CLIPS)
        return 1
    mood_loops = {c.name: c for c in load_present(MOOD_LOOP_CANDIDATES)}
    oneshots = {c.name: c for c in load_present(ONESHOT_CANDIDATES)}
    # Belt & braces: never let a leg-moving / reserved clip into the schedulable
    # pools even if it somehow parsed as a one-shot candidate.
    oneshots = {n: c for n, c in oneshots.items() if _trigger_is_safe(c)}
    idle_bg_by_name = {c.name: c for c in idle_bgs}
    log.info("idle backgrounds: %s", [c.name for c in idle_bgs])
    log.info("mood loops: %s", sorted(mood_loops) or "none")
    log.info("congruent one-shots: %s", sorted(oneshots) or "none")

    # 3) Baseline thermal scan BEFORE energising anything (catch a hot/faulted bot).
    pre = thermal_scan("pre")
    if pre is None:
        log.error("could not read servo bus at startup -> exiting (passive)")
        return 1
    if pre["abort"]:
        log.error("unsafe baseline (see thermal[pre]) -> exiting (passive)")
        return 1

    # 4) Install signal + atexit safety BEFORE touching the head.
    global _EYED, _ANTD, _IO, _REST, _TORQUED, _STOP
    atexit.register(lambda: graceful_shutdown("atexit"))
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        try:
            signal.signal(sig, _on_signal)
        except Exception:  # noqa: BLE001
            pass

    # 5) Bring up the show (eyes lit + blink) and the servo bus.
    try:
        _EYED = EyeDriver()
        _ANTD = AntennaDriver(ANTENNAS)
        log.info("eyes constructed -> lit + blinking")
        io = _open_feetech()
        _IO = io
        _REST = list(io.read_present_position(HEAD_IDS))
        log.info("rest head pose: %s", ["%.3f" % x for x in _REST])
        arm(io)
    except Exception as e:  # noqa: BLE001
        log.exception("startup failure: %s -> shutting down passive", e)
        graceful_shutdown("startup-error")
        return 1

    # 6) Main 50 Hz loop.
    rng = random.Random()
    available = set(idle_bg_by_name) | set(mood_loops) | set(oneshots)
    planner = idle_moods.MoodPlanner(
        available, rng=rng,
        dwell_min_s=MOOD_DWELL_MIN_S, dwell_max_s=MOOD_DWELL_MAX_S,
        beat_min_s=BEAT_MIN_S, beat_max_s=BEAT_MAX_S, quiet_prob=QUIET_PROB)

    sel_bg = rng.choice(idle_bgs)
    engine = new_engine(sel_bg)
    cur_bg_name = sel_bg.name

    def _bg_for_mood():
        """Resolve the current mood to a concrete background clip. NEUTRAL
        rotates the plain idle_* loops; a mood sits in its own loop (falling
        back to an idle_* loop if that mood clip isn't deployed)."""
        name = planner.background_clip()
        if name and name in mood_loops:
            return mood_loops[name]
        choices = [c for c in idle_bgs if c.name != cur_bg_name] or idle_bgs
        return rng.choice(choices)

    log.info("mood -> %s | background -> %s", planner.mood, cur_bg_name)

    t0 = time.monotonic()
    next_t = t0
    last = t0
    active_start = t0
    planner.start(t0)
    next_bg = t0 + rng.uniform(BG_MIN_S, BG_MAX_S)   # NEUTRAL idle_* rotation only
    last_head = np.zeros(4)
    xfade_from = None      # np.array of the pose to blend out of, or None
    xfade_end = 0.0

    win_ticks = 0
    win_misses = 0
    win_pmax = 0.0
    rc = 0

    try:
        while not _STOP:
            now = time.monotonic()
            if MAX_RUN_S and (now - t0) >= MAX_RUN_S:
                log.info("MAX_RUN_S reached -> stopping")
                break

            # --- duty-cycle thermal checkpoint -------------------------------
            if (now - active_start) >= ACTIVE_S:
                span = now - active_start
                log.info("checkpoint: %.1f min animated, %d ticks, %.2f Hz, "
                         "%d deadline misses, period_max=%.1f ms",
                         span / 60.0, win_ticks,
                         (win_ticks / span) if span else 0.0, win_misses,
                         win_pmax * 1e3)
                io = None   # drop this loop's handle ref so the port can release
                if not do_relax_checkpoint():
                    rc = 1
                    break
                io = _open_feetech()
                _IO = io
                arm(io)
                active_start = time.monotonic()
                next_t = active_start
                last = active_start
                win_ticks = win_misses = 0
                win_pmax = 0.0
                planner.start(active_start)
                next_bg = active_start + rng.uniform(BG_MIN_S, BG_MAX_S)
                xfade_from = None
                continue

            io = _IO
            t = now - t0

            # --- mood drift: settle into / leave a mood ----------------------
            new_mood = planner.maybe_transition(now)
            if new_mood is not None and xfade_from is None:
                sel_bg = _bg_for_mood()
                engine = new_engine(sel_bg)
                cur_bg_name = sel_bg.name
                xfade_from = last_head.copy()
                xfade_end = now + XFADE_S
                next_bg = now + rng.uniform(BG_MIN_S, BG_MAX_S)
                log.info("mood -> %s | background -> %s", planner.mood, cur_bg_name)

            # --- NEUTRAL baseline: rotate idle_* backgrounds for variety -----
            if (planner.background_clip() is None and now >= next_bg
                    and xfade_from is None):
                choices = [c for c in idle_bgs if c.name != cur_bg_name] or idle_bgs
                sel_bg = rng.choice(choices)
                engine = new_engine(sel_bg)
                cur_bg_name = sel_bg.name
                xfade_from = last_head.copy()
                xfade_end = now + XFADE_S
                next_bg = now + rng.uniform(BG_MIN_S, BG_MAX_S)
                log.info("background -> %s", cur_bg_name)

            # --- congruent one-shot beat (calm, head-only, no repeats) -------
            trg = None
            if xfade_from is None:
                name = planner.maybe_oneshot(now)
                if name and name in oneshots:
                    cc = oneshots[name]
                    if _trigger_is_safe(cc):   # belt & braces vs leg/dock clips
                        trg = oda.Triggers(clips=[cc])
                        log.info("mood=%s one-shot -> %s", planner.mood, cc.name)
                    else:
                        log.warning("blocked unsafe one-shot -> %s", name)

            # --- slow blink (sleepy / content / sad moods) -------------------
            if planner.maybe_slow_blink(now):
                _EYED.slow_blink()

            # --- evaluate + drive --------------------------------------------
            w0 = time.perf_counter()
            meas = io.read_present_position(HEAD_IDS)
            out = engine.evaluate(t, oda.MODE_DOCK, trg or oda.Triggers())
            head = np.asarray(out.head_targets, float)
            if xfade_from is not None:
                if now >= xfade_end:
                    xfade_from = None
                else:
                    a = 1.0 - (xfade_end - now) / XFADE_S
                    head = (1 - a) * xfade_from + a * head
            io.write_goal_position(HEAD_IDS, [float(x) for x in head])
            _EYED.tick(out.show)
            _ANTD.tick(out.show)
            last_head = head.copy()
            w1 = time.perf_counter()

            mv = np.asarray(meas, float)
            if float(np.max(np.abs(mv))) > 3.0:
                raise RuntimeError("head measured out of range: %s" % mv)

            if (w1 - w0) > DT:
                win_misses += 1
            win_ticks += 1
            p = now - last
            last = now
            if p > win_pmax:
                win_pmax = p

            next_t += DT
            sl = next_t - time.monotonic()
            if sl > 0:
                time.sleep(sl)
            elif sl < -0.1:
                next_t = time.monotonic()   # fell behind; resync, do not spiral
    except Exception as e:  # noqa: BLE001
        log.exception("loop error: %s -> shutting down passive", e)
        rc = 1
    finally:
        io = None            # drop any lingering handle so the post scan can open
        graceful_shutdown("main-exit")
        gc.collect()
        thermal_scan("post")
    return rc


def do_relax_checkpoint():
    """Ease the head to rest, DE-ENERGISE it, release the port, take a thermal
    scan, hold relaxed for RELAX_S, then return.  Returns False on abort."""
    global _IO, _TORQUED
    io = _IO
    try:
        cur = io.read_present_position(HEAD_IDS)
        _ramp(io, cur, _REST, dur=1.2)          # gentle to rest (no drop)
        io.disable_torque(HEAD_IDS)
        _TORQUED = False
    except Exception as e:  # noqa: BLE001
        log.warning("relax ramp/disable err: %s", e)
    _IO = None
    try:
        del io
    except Exception:  # noqa: BLE001
        pass
    gc.collect()
    time.sleep(0.3)                              # fully release the port
    sc = thermal_scan("relax")
    t_end = time.monotonic() + max(0.0, RELAX_S)
    while time.monotonic() < t_end and not _STOP:
        time.sleep(0.2)                          # head limp, eyes keep blinking
    if sc is not None and sc["abort"]:
        log.error("abort condition during relax checkpoint -> stopping (passive)")
        return False
    return True


if __name__ == "__main__":
    sys.exit(main())
