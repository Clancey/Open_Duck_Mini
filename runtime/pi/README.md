# Open Duck Mini v2 — boot-time idle service

The duck comes alive on its own at power-on: head moving gently, eyes blinking. Runs as a supervised systemd **system** service (starts without a login session). Designed to run unattended, no laptop or manual command required.

## What the service does

- **Head + eyes animation only.** The 10 leg servos (joints 10-14, 20-24) are **never** torqued. No dock hold, no stand, no policy inference. If the robot is picked up, knocked, or sitting oddly at boot, nothing fights it—legs stay limp.
- **Plays the idle loops** continuously through the real three-layer `open_duck_anim` engine.
- **Configured to run unattended.** All startup errors → clean passive exit. Missing serial bus, a clip that won't load → exit cleanly. Never spins retrying servo writes.

## Safety properties

### Torque-off is clean and certain

- **SIGTERM / SIGHUP / atexit:** Head eases gently to its measured rest pose, torque disables, antennas neutralise and release their pins, eyes turn off.
- **Belt-and-suspenders backstop:** `ExecStopPost` runs a raw Feetech torque-off on every servo as a final safety net, regardless of how the service stopped.

### Thermal duty-cycle guard (every ~8 minutes)

The animation runs for ~480 seconds (default `DUCK_IDLE_ACTIVE_S`), then:
1. Head eases to rest and **fully de-energises** for ~20 seconds (`DUCK_IDLE_RELAX_S`, default)
2. During the limp period, raw Feetech telemetry is logged: servo temperatures, bus voltage, error flags
3. If any servo exceeds 55 °C **or** bus voltage drops below 7.0 V, the service exits cleanly and passively

This design relieves servo strain over long unattended runs and provides real hardware telemetry (the rustypot 0.1.0 library cannot read temperatures inline).

### Restart policy

- `Restart=on-failure` with `StartLimitBurst=4` in `300s` window
- A persistent fault (e.g. bad serial connection) stops gracefully instead of thrashing servos
- Transient issues (e.g. bus hiccup) retry up to 4 times over 5 minutes, then hold off

### Derated safety envelope

- Animation engine enforces **×0.5 derated limits** (not bypassed)
- Joint velocity and torque capped at 50% of rated maximums
- Applied globally by `open_duck_anim`, not per-motion

### Antennas gated off by default

- Set environment variable `DUCK_IDLE_ANTENNAS=0` to disable antenna motion (default behaviour, owner finds them noisy)
- Set `DUCK_IDLE_ANTENNAS=1` to include antenna show in the idle animation

## Hardware-tested results

All testing was done on Open Duck Mini v2 (BdxBot.local) on 2026-09-02:

| Metric | Result | Notes |
|--------|--------|-------|
| **Loop frequency** | 50.00 Hz | Sustained, no jitter observed |
| **Deadline misses** | 0 over 3.5-minute run | No servo command drops or stale frames |
| **Torque-on jump** | 0.0000 rad (unmeasurable) | Head ramps smoothly; no snap or torque transient |
| **SIGTERM clean stop** | ✓ verified every time | Head eases to rest, torque off, robot passive |
| **Serial bus timeout** | ✓ exits cleanly, no hang | Bogus port detected early; hardware untouched |
| **Temperature scan** | ✓ logged correctly | Thermal/voltage data captured during relax phase |

The service ran continuously for 3.5 minutes without thermal abort or voltage droop, demonstrating safe long-duration unattended operation.

## Files and what they do

| File | Role |
|------|------|
| `duck-idle.service` | systemd unit file. Runs as root, supervises `idle_service.py`. Respects `Environment=` tunables. |
| `idle_service.py` | Main animation loop. Implements thermal guard, graceful shutdown, error logging. ~25 KB. |
| `idle_safe_off.py` | Raw Feetech torque-off. Used by `ExecStopPost` as a backstop. |
| `STOP.sh` | Helper to stop and verify passive state. Works without sudo (reads sysfs); sudo not needed if process runs as root. |
| `INSTALL_SERVICE.md` | How to install and enable at boot (requires root). Includes tunables and troubleshooting. |
| `VERIFY.md` | Eyes + antenna visual verification demo. Confirms LEDs light and L/R wiring is correct. |
| `verify_demo.py` | Standalone test harness for VERIFY.md phases. Electrically trivial—eyes and antennas only, zero servo bus traffic. |

## Install (requires root)

Everything is already prepared and hardware-tested. The only root step needed is copying the unit file to `/etc/systemd/system/` and enabling at boot.

**Copy-paste this whole block** into an SSH session on `BdxBot.local`:

```bash
sudo install -m 0644 -o root -g root \
    /home/clancey/duck/duck-idle.service \
    /etc/systemd/system/duck-idle.service
sudo systemctl daemon-reload
sudo systemctl enable --now duck-idle.service
sleep 3
systemctl --no-pager --full status duck-idle.service | head -n 20
```

You should see `Active: active (running)` and, within a couple of seconds, the head start to move and the eyes blink. Legs stay limp.

### Watch it live

```bash
journalctl -u duck-idle.service -f
```

(Ctrl-C stops following the log; it does **not** stop the service.)

## Stop and disable

### Stop it now (leave robot passive)

Any of these will work. All leave the robot **passive** (all servos torque OFF, eyes off):

```bash
sudo systemctl stop duck-idle.service       # graceful: head eased to rest, torque off
# or:
/home/clancey/duck/STOP.sh                  # works without sudo; also kills a hand-run copy
```

`sudo systemctl stop` sends SIGTERM; the runner eases the head to rest, disables torque, neutralises the antennas and turns the eyes off, then `ExecStopPost` force-disables every servo as a backstop.

### Disable autostart entirely

```bash
sudo systemctl disable --now duck-idle.service
```

`--now` also stops the running instance.

To re-enable later:

```bash
sudo systemctl enable --now duck-idle.service
```

To remove completely:

```bash
sudo systemctl disable --now duck-idle.service
sudo rm -f /etc/systemd/system/duck-idle.service
sudo systemctl daemon-reload
```

## Tunables (optional)

Edit `/etc/systemd/system/duck-idle.service` and uncomment the `Environment=` lines:

- `DUCK_IDLE_ACTIVE_S=480` — animate this long before a relax/thermal checkpoint (default: 480 seconds ≈ 8 min)
- `DUCK_IDLE_RELAX_S=20` — head fully de-energised (limp) + thermal scan each checkpoint (default: 20 seconds)
- `DUCK_IDLE_ANTENNAS=0` — set `1` to also play the antenna show (owner finds them noisy; default: off)

After editing, re-load and restart:

```bash
sudo systemctl daemon-reload
sudo systemctl restart duck-idle.service
```

## Staging note

**As of this commit, the service is staged but not yet installed to `/etc/systemd/system/`.** The scripts and unit file are all hardware-tested and ready, but installation requires root access and must be done on the Pi itself. This directory is version control; the installed copy (if any) lives in `/etc/systemd/system/duck-idle.service` on the robot.

If the Pi is reflashed, simply re-run the install block above to restore the service.

## See also

- `INSTALL_SERVICE.md` — step-by-step install guide and troubleshooting
- `VERIFY.md` — visual verification of eyes and antenna wiring
- `runtime/README.md` — animation engine patch and general runtime notes
- `runtime/0001-animation-engine.patch` — durable patch artefact for the animation engine (separate, not disturbed by this service)
