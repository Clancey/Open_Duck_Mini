# VERIFY.md — Open Duck Mini v2 eyes + antennas visual verification

A short, purpose-built demo to confirm **by eye** two things software cannot
self-check. Everything it touches is electrically trivial: **eye LEDs and
antenna servos only**. It never opens the Feetech servo bus, never torques a
leg or the head — **zero fall/thermal risk**.

Script: `~/duck/verify_demo.py`

---

## LEFT / RIGHT convention — read this first

All "LEFT"/"RIGHT" mean the **ROBOT'S OWN** left/right (as if *you were the
duck*). When you stand **facing** the robot, they are mirrored:

| Banner says        | The antenna physically on … |
|--------------------|-----------------------------|
| robot's **LEFT**   | **your RIGHT** hand side    |
| robot's **RIGHT**  | **your LEFT** hand side     |

Software mapping being confirmed (known-correct in code):
`left → pwm_left → board.D13, sign +1` · `right → pwm_right → board.D12, sign −1`.
This demo checks the **physical wiring** matches that.

---

## Run it

One command on the Pi (foreground — you watch the robot and read the banners
live; **Ctrl-C is safe at any moment**):

```bash
~/duck/venv/bin/python -u ~/duck/verify_demo.py
```

If your SSH link is flaky, run it detached and follow the log instead (the demo
still cleans up on a dropped link):

```bash
setsid bash -c '~/duck/venv/bin/python -u ~/duck/verify_demo.py > ~/duck/verify_demo.log 2>&1 < /dev/null' & tail -f ~/duck/verify_demo.log
```

Repeat a single phase (1–5), e.g. the left antenna:

```bash
~/duck/venv/bin/python -u ~/duck/verify_demo.py --phase 3
```

Full run is ~90 s. Eyes and antennas are shown **separately** — eyes are turned
fully OFF before any antenna moves — so each phase tests exactly one thing.
On any exit (normal, Ctrl-C, SIGTERM/SIGHUP, crash) the eyes go OFF and the
antennas return to neutral and release their pins.

---

## What PASS looks like, phase by phase

**Phase 1 — EYES ON, STEADY.** Both eyes light and hold steady (no blink) for
~4 s. → PASS: both clearly illuminate. FAIL: one or both stay dark.

**Phase 2 — EYES BLINK.** Three deliberate blinks, each announced
("BLINK 1 of 3 …") — count them. Then ~25 s of the natural idle blink.
→ PASS: 3 clean counted blinks, then **irregular**, lifelike blinks (random
2–6 s apart, an occasional quick double) that read *alive*, not metronomic.
FAIL: no blink during the counted part, or perfectly evenly-spaced ticking.

**Phase 3 — LEFT antenna ONLY.** Only the robot's **left** antenna
(**your right**) sweeps, twice. The right one must not twitch.
→ PASS: left moves, right dead still. **FAIL (swapped): the right one moves.**

**Phase 4 — RIGHT antenna ONLY.** Only the robot's **right** antenna
(**your left**) sweeps, twice. The left one must not twitch.
→ PASS: right moves, left dead still. **FAIL (swapped): the left one moves.**

**Phase 5 — BOTH, alternating L, R, L, R.** Each move announced.
→ PASS: the antenna that moves matches every announcement, in order.

---

## If a phase FAILS

### Eyes don't light (Phase 1) or don't blink (Phase 2)
- Re-seat the eye LED connectors (channels are **board.D24 = left**,
  **board.D23 = right**).
- Confirm the runtime is the fixed one: the LED bring-up bugs
  (connect() de-initialising the pins; `_drive_show` dropping `eye` events)
  are fixed at commit **7190500** or later. Check the eyes module exists and is
  the rewritten version:
  `grep -n "def double_blink\|def note_authored" ~/duck/Open_Duck_Mini_Runtime/mini_bdx_runtime/mini_bdx_runtime/eyes.py`
- Standalone LED check (drives D24/D23 directly, background blink):
  `~/duck/venv/bin/python ~/duck/Open_Duck_Mini_Runtime/mini_bdx_runtime/mini_bdx_runtime/eyes.py` (Ctrl-C to stop).

### Antennas are SWAPPED (Phase 3 moves the right one, or Phase 4 the left)
The software mapping is correct, so a swap is a **physical wiring** issue. The
one-line fix is in:

```
~/duck/Open_Duck_Mini_Runtime/mini_bdx_runtime/mini_bdx_runtime/antennas.py
```

Lines 6–9 are the entire mapping:

```python
LEFT_ANTENNA_PIN  = board.D13   # left  antenna signal pin
RIGHT_ANTENNA_PIN = board.D12   # right antenna signal pin
LEFT_SIGN  = 1                  # deflection direction for +left command
RIGHT_SIGN = -1                 # deflection direction for +right command
```

- **L/R swapped** (most common — a "left" command drives the physical right
  antenna): **swap the two pin values on lines 6–7** so they read
  `LEFT_ANTENNA_PIN = board.D12` and `RIGHT_ANTENNA_PIN = board.D13`.
  (Equivalently, physically swap the two servo signal leads on D12/D13.)
- **One antenna deflects the *wrong direction*** (side is right, but it bends
  backward): flip that side's sign — `LEFT_SIGN = -1` or `RIGHT_SIGN = 1` on
  lines 8–9. The demo's sweep is symmetric, so this only matters for
  deflection *direction*, not for the L/R identity check.

After editing, re-run just the antenna phases to confirm:

```bash
~/duck/venv/bin/python -u ~/duck/verify_demo.py --phase 3
~/duck/venv/bin/python -u ~/duck/verify_demo.py --phase 4
```

Nothing else in the runtime needs to change — this file is the single source of
truth for the antenna L/R + sign mapping (`real_robot.set_antennas` calls
`set_position_left(arg0)` / `set_position_right(arg1)`, i.e. joint index 9 =
left, index 10 = right).

---

## Safety notes
- No servo bus is opened; **legs and head are never energised**. Verified
  passive after every run (all 14 servos torque-OFF).
- Antennas de-init (pins released, servos passive) and eyes off at the end and
  on any interruption.
