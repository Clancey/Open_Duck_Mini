# Runtime Animation Engine Integration (Phase 4)

## Overview

This captures the Phase 4 integration of the `open_duck_anim` animation engine into `apirrone/Open_Duck_Mini_Runtime`. The work is version-controlled here (in this repo) so it survives development and can be reviewed, but it is **already committed** on the owner's runtime fork as the `feature/animation-engine` branch of `Clancey/Open_Duck_Mini_Runtime`. That branch is the primary way to get the code onto a robot; the `0001-animation-engine.patch` in this directory is a durable fallback for applying the same change onto a different base (e.g. upstream `v2`).

## What This Patch Contains

The Phase 4 integration includes:

- **Mode FSM**: A finite-state machine with states BOOT/DISARMED, ARMING, DOCK_DEMO, STAND, WALK, and a latched FAULT state. The machine transitions from DISARMED to ARMING with a measured-pose ramp (smooth transition from the current joint angles), then to mode-specific states. Faults latch and must be cleared with an explicit reset.

- **Minimum Safety Set**:
  - Controlled abort mechanism to stop the robot mid-animation
  - E-stop button integration
  - Watchdog timers to detect control loop timeouts
  - Duty-cycle-based thermal management (since the rustypot HWI exposes no temperature readings)

- **AnimationController**: The core runtime component that loads and plays `open_duck_anim` animation clips, providing frame-by-frame joint command synthesis.

- **RobotInterface Abstraction**: A hardware abstraction layer with a MockRobot test seam for validation without hardware. This layer exposes measured state (joint angles, IMU) and accepts joint commands. The real adapter (`anim/real_robot.py`) imports the on-robot device modules lazily, so the FSM, controller and safety code stay importable on a laptop / CI with no `board` / `rustypot` present.

- **Consent-gated antenna motion**: `RealRobot.connect()` opens the servo bus and IMU **passively** — it never torques the leg/head servos on and never constructs `Antennas()` (whose constructor immediately drives the PWM pins). The antenna PWM is energised lazily, only on the first genuine (non-neutral) animation command, i.e. once the FSM has left DISARMED. Merely connecting, or holding in BOOT/DISARMED/FAULT, leaves the antennas dark. `RealRobot(enable_antennas=False)`, `connect(actuate=False)` (a read-only inspection attach), and `dock_demo.py --no-antennas` disable them unconditionally.

- **Entry Points**:
  - `scripts/dock_demo.py`: Standalone demo that bypasses the policy entirely and directly plays animation clips on the docked robot
  - `scripts/v2_rl_walk_mujoco.py`: Extended with `--head_animation` overlay to compose head-tracking animations with walking policy outputs

- **Safety Enforcement**: Re-enabled 5.24 rad/s velocity clip on final bus targets to prevent unrealistic joint velocities.

- **Example Clip**: `clips/idle_alive.duckanim` — a short looping idle animation for testing.

- **Eyes — end-to-end wiring (hardware-recovered)**: two bugs that only surfaced on the robot are fixed. `RealRobot.connect()` used to build `Eyes()` then immediately `stop()` it, which deinitialised the GPIO pins so every later `set_eyes()` wrote to dead pins (the swallowed exception meant the eyes never lit); `connect()` no longer stops the eyes. `_drive_show()` handled `sound`/`projector` events but silently dropped `eye` events, so clip cues (`wide`/`blink`/`happy`) never reached hardware; it now routes them via `set_eye_event` (mock-safe `getattr`). `eyes.py` was rewritten so a single background thread owns a natural idle blink (lit baseline; 0.12 s dark flicks; uniform 2–6 s interval; ~18% spontaneous double-blink) with thread-safe cues; `set_eyes` maps to `note_authored` (a 1→0 authored edge = a blink, no longer forcing the eyes dark) and clip cues take precedence over the idle loop, then idle resumes.

- **IMU optional for non-balancing paths (hardware-recovered, safety-hardened)**: `connect()` tolerates an unavailable BNO055 (I2C disabled) by degrading to `self.imu = None` with a loud warning, so the dock / head-only demo runs on a bench without an IMU. `read()` reports `SensorSnapshot.tilt_valid`; a zero placeholder tilt is flagged invalid so it can never be mistaken for "upright". The FSM **requires** valid tilt sensing for the balancing modes: STAND/WALK refuse to be entered — and latch FAULT if entered — when `tilt_valid` is False. Only DOCK_DEMO / head paths may run IMU-less.

- **Test Suite**: 71 unit and integration tests covering FSM transitions, animation playback, mock hardware, safety constraints, the antenna consent gate (`tests/test_real_robot.py`), the recovered eye wiring and idle-blink composition (`tests/test_eyes.py`), and the IMU-optional tilt-validity gating (`tests/test_fsm.py`, `tests/test_real_robot.py`).

## Getting the Code

### Primary path — clone the fork branch (already committed)

The integration is committed on the owner's fork, so no patch application is needed:

```bash
git clone -b feature/animation-engine https://github.com/Clancey/Open_Duck_Mini_Runtime
cd Open_Duck_Mini_Runtime
```

### Fallback path — apply the patch onto a different base

To re-apply the same change onto upstream `apirrone/Open_Duck_Mini_Runtime@v2` (or any other base), use the patch in this directory. It is a standard `git format-patch` artifact — now a single file containing the **three** commits of `v2..feature/animation-engine` (the Phase 4 integration, the non-actuating-connect/antenna-defer fix, and the hardware-recovered eye + IMU fixes):

```bash
git clone -b v2 https://github.com/apirrone/Open_Duck_Mini_Runtime
cd Open_Duck_Mini_Runtime
git am /path/to/runtime/0001-animation-engine.patch
# or, if you do not want commits: git apply /path/to/runtime/0001-animation-engine.patch
```

The patch is verified to apply cleanly onto `v2` with `git apply --check` and `git am` (fresh clone), after which all 71 runtime tests pass.

## Setting Up a Fresh Raspberry Pi (Pi Zero 2 W, Debian 13 trixie, Python 3.13)

This section reflects a **real** setup on a freshly-reset Pi Zero 2 W running Debian 13 (trixie), which ships **only** Python 3.13. The upstream base README assumes older Raspberry Pi OS, `mkvirtualenv`, and pins that predate cp313 — do not follow it verbatim on trixie. A plain `pip install -e .` of the runtime **fails as pinned** on Python 3.13 (see below). Use a plain venv and the versions here instead.

### Why `pip install -e .` fails as-pinned

The runtime `setup.cfg` pins `numpy==1.26.4` and `onnxruntime==1.18.1` (neither has a cp313 aarch64 wheel) and pulls `pypot` from git (which needs `git` installed). Installing the whole pinned set on Python 3.13 will fail. The animation runtime does not need the full set for a dock demo: install the pieces that actually have cp313 aarch64 wheels.

### Working install

```bash
# From an empty working directory next to the two repos:
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip wheel

# Animation core (numpy-only) + the deps the runtime actually needs:
pip install ./Open_Duck_Mini/packaging/open_duck_anim   # pulls numpy
pip install onnxruntime pytest
pip install 'rustypot==0.1.0'                           # see rustypot note below
```

Versions that work on this Pi (all prebuilt manylinux aarch64 cp313 wheels — no source build, no compiler, no swap needed):

- **numpy 2.5.2** (upstream pin was `1.26.4`, which has no cp313 wheel). `open_duck_anim` needs only `numpy>=1.21` and its full test suite passes on 2.5.2.
- **onnxruntime 1.29.0** (upstream pin was `1.18.1`, which has no cp313 wheel). PyPI has prebuilt cp313 aarch64 wheels from `onnxruntime>=1.20`, so `pip install onnxruntime` pulls a binary wheel; the feared "build from source OOMs on a 415 MiB Pi" never happens.

> Note the piwheels caveat: `/etc/pip.conf` lists piwheels as an extra index, but piwheels serves 32-bit armv6l/armv7l wheels that do **not** install on this 64-bit OS. The working wheels come from PyPI manylinux aarch64.

### rustypot API break (this one bites) — pin `rustypot==0.1.0`

The HWI (`mini_bdx_runtime/rustypot_position_hwi.py`) calls `rustypot.feetech(port, 1000000)`. That factory exists only in **rustypot 0.x**. The current PyPI default, **1.7.0**, removed `feetech` in favour of per-model classes (`Sts3215PyController(port, baud, timeout)`), so a plain `pip install rustypot` gives 1.7.0 and `rustypot.feetech(...)` raises `AttributeError`. **Pin `rustypot==0.1.0`** (it has a cp313 aarch64 wheel) so the HWI imports and its read path works.

Tradeoff: 0.1.0's `feetech` exposes only position/velocity (no voltage/temperature/current), which matches the gap already documented in `real_robot.py` and the duty-cycle thermal manager. rustypot **1.7.0 is fine for ad-hoc read-only** voltage/temperature/current queries (`Sts3215PyController` has `read_present_voltage` / `_temperature` / `_current`); install it temporarily for a scan, then reinstall `0.1.0` to restore HWI compatibility. A longer-term fix is to port the HWI to the 1.x `Sts3215PyController` API, which also unlocks temp/current for the thermal manager.

### IMU driver (required before the walking script runs)

The IMU driver is **not installed by default** and is needed by `raw_imu.Imu`:

```bash
pip install adafruit-circuitpython-bno055
```

(Enable I2C for the IMU if it is not already enabled; that needs `raspi-config` / root.)

### Antenna PWM stack (Adafruit-Blinka / GPIO on trixie)

`pip install Adafruit-Blinka` tries to build `RPi.GPIO`, `lgpio`, and `rpi_ws281x` from source, which fails on trixie because `python3-dev` (Python.h) and `swig` are missing. Two ways forward:

1. **With root (clean):**
   ```bash
   sudo apt install python3-dev swig      # or: sudo apt install python3-rpi-lgpio
   pip install Adafruit-Blinka
   ```

2. **Without root (venv `.pth` to the apt GPIO backend):** install only Blinka's pure-python parts and point the venv at the apt-provided GPIO backend (`lgpio`/`gpiod`/`rpi_lgpio` under `python3-lgpio`/`libgpiod`):
   ```bash
   pip install --no-deps Adafruit-Blinka Adafruit-PlatformDetect Adafruit-PureIO adafruit-circuitpython-typing
   SP="$VIRTUAL_ENV/lib/python3.13/site-packages"
   echo /usr/lib/python3/dist-packages > "$SP/zz-system-gpio.pth"
   ```
   The `.pth` is **appended** (the `zz` prefix sorts last), so venv packages still win; verify `numpy`/`onnxruntime` still resolve to the venv. With either path, `import board` / `import pwmio` succeed and `board.D13` / `board.D12` resolve.

### Serial port (CH343 → `/dev/ttyACM0`, not `/dev/ttyUSB0`)

The base README assumes an FTDI adapter at `/dev/ttyUSB0` bound by `ftdi_sio`, plus a latency-timer udev rule. The owner's adapter is a **WCH CH343** (`1a86:55d3`), which uses the `cdc_acm` driver and enumerates as **`/dev/ttyACM0`** — there is no `/dev/ttyUSB0` on this box, and the `ftdi_sio` rule will not match it.

Prefer the **stable by-id path** (a bare `ttyACM` number can renumber across reboots/replug). The owner's adapter is:

```
/dev/serial/by-id/usb-1a86_USB_Single_Serial_58FA095764-if00
```

Find your own with:

```bash
ls /dev/serial/by-id/
```

The HWI already defaults to `/dev/ttyACM0`, and `scripts/dock_demo.py` takes `--usb-port`, so pass the by-id path — no code edit required:

```bash
python scripts/dock_demo.py --usb-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_58FA095764-if00 ...
```

(Optional, needs root: a udev rule `SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{serial}=="58FA095764", SYMLINK+="ttyDUCK"` gives a permanent `/dev/ttyDUCK`.)

## Dependencies

The runtime must have `open_duck_anim` importable on the Pi. Provide it either by:
- Installing the package: `pip install /path/to/open_duck_anim` (the `packaging/open_duck_anim` project), or
- Setting the environment variable `OPEN_DUCK_ANIM_HOME=/path/to/Open_Duck_Mini` before running the daemon or tests.

## Running the Dock Demo on Physical Hardware

The dock demo plays animation clips directly on the robot without the walking policy.

### Prerequisites

1. Get the code (fork branch above) and set up the venv + dependencies as above.
2. Ensure `open_duck_anim` is importable (see Dependencies).
3. Copy the animation clips to the Pi (or clone this repo's `experiments/animation/clips`):
   ```bash
   scp clips/idle_alive.duckanim clancey@<robot-ip>:~/
   ```

### Steps to Run

1. **Power the robot in the dock/cradle** with a hand near the physical power switch. This ensures safe failure modes. On this build all 14 servos power up torque-OFF (limp); the bus is live (~7.8 V on a healthy 2S pack) but nothing holds position until you arm.

2. On the Pi, start the demo (using the stable by-id serial path):
   ```bash
   PORT=/dev/serial/by-id/usb-1a86_USB_Single_Serial_58FA095764-if00
   python scripts/dock_demo.py --usb-port "$PORT" \
     --clip ~/idle_alive.duckanim --derating 0.5 --max-demo-s 120
   ```

3. Control the robot interactively:
   - Press `a` to arm (the robot ramps from its measured current pose)
   - Press `d` to confirm dock and enter DOCK_DEMO mode (plays the animation clip)
   - Press `1`..`9` to fire a clip trigger
   - Press `x` to e-stop (latches; reset with `r`)
   - Press `r` to reset a latched fault state
   - Press `q` to quit cleanly

4. **For first trials, keep `--derating 0.5`** to limit motor current. Increase toward 1.0 only as hardware data accrues and you gain confidence in the robot's structural integrity.

5. Antennas move only once an animation actually articulates them (after you arm and enter DOCK_DEMO), not at connect time. Pass `--no-antennas` to keep the antenna PWM dark for a leg/head-only trial.

6. Monitor the robot for:
   - Jerky motion (sign of clipping or safety constraints being hit)
   - Heat in the motor housings
   - Any audible grinding or unusual sounds
   - Foot slippage if docking contact is not perfect

## Performance on the Pi Zero 2 W (measured)

The animation engine is comfortably affordable on this hardware. Measured on the actual Pi Zero 2 W (Debian 13, Python 3.13, `CPUExecutionProvider`):

- **ONNX inference** (`HEAD_PASSTHROUGH_300M.onnx`): **0.777 ms** per call (median 0.763, p95 0.851; ~1290 inferences/s).
- **`Engine.evaluate`** hot path (DOCK mode, background + triggers): **665 µs** per tick (median 659, p95 694, max 991).
- **Combined ≈ 1.45 ms/tick**, about **7% of the 20 ms (50 Hz) control budget** — roughly 30× headroom on `Engine.evaluate` alone — leaving ample room for bus IO and IMU reads.

## Hardware bring-up results (measured on the physical robot)

Recorded during the first physical bring-up of the robot (Open Duck Mini v2, Pi Zero 2 W). These are hard-won, decision-relevant numbers from running the **unmodified** `scripts/dock_demo.py` on hardware:

- **Dock demo end-to-end**: ran **1352 ticks with 0 overruns** at 50 Hz (zero >20 ms deadline misses over the full ~27 s demo; a dry run beforehand was 1333 ticks / 0 overruns), with the 10 leg servos torqued at `kp=30` and **held at `init_pos`** (load-relieving dock posture) while the head, neck, antennas and eyes animated. The FSM armed from the measured limp pose (no snap), entered DOCK_DEMO, played `curious_tilt`, `happy_bounce`, `nod_yes`, `perk_up` and `double_take` over the looping `idle_alive` background, then shut down cleanly (torque off + neutral show).
- **Thermals / power flat**: all servos **27–31 °C** and bus **7.8–7.9 V**, identical before and after the demo — the static `kp=30` leg hold produced no measurable heating and no thermal-cooldown events fired.
- **Engine cost on-Pi vs sim**: `Engine.evaluate` measured **~732 µs/tick** on the Pi (idle_alive, mean; p95 856 µs) versus **665 µs** in sim — about 10% higher on-hardware, still tiny (~3.7% of the 20 ms budget). Per-tick control work was ~2.2 ms mean (~11% of budget, ~17.5 ms headroom).
- **The head / dock path does NOT run the ONNX policy.** The head is driven by absolute head targets (policy bypassed), so the measured **0.777 ms** ONNX inference cost applies **only to STAND/WALK**, not to the dock/head-animation path. Budget headroom on the dock path is therefore even larger than the combined figure above.
- **IMU-less validation**: `connect()` succeeded with the IMU-unavailable warning (I2C was disabled on this Pi), proving the optional-IMU degrade; the demo ran on a zeroed (`tilt_valid=False`) tilt snapshot, which is correct for a docked robot and is explicitly blocked from ever entering STAND/WALK.
- **Eyes**: after the wiring fixes, the standalone eye test drove the LED pins with the background idle blink plus single/double/wide-hold cues, and the dock demo exercised the idle blink + clip eye cues together. (Physical left/right illumination is for the owner to eyeball.)

> Not attempted: no walking (Rung 6) and no calibration that requires the robot to stand were run. Legs were never torqued except to hold `init_pos` in the dock. Standing/walking remains a separate, IMU-required, owner-supervised session.

## Known Limitations

The following remain true on this build:

- **Thermal Management**: The pinned `rustypot==0.1.0` HWI does not expose motor temperature or current draw, so thermal management is duty-cycle-based only (sustained load time + cooldown, plus the load-relieving dock posture). No real-time over-temperature shutdown is possible. Porting the HWI to `rustypot` 1.x `Sts3215PyController` would surface per-servo temperature/current and lift this limitation.

- **Foot Contact Sensing**: This build has no foot-contact sensors, so contact state is hard-coded as `[1, 1]` (both feet always "in contact"). The dock demo holds the legs, so this only affects the STAND/dock-handoff guard; the controller will not detect unexpected slips.

- **Tilt Estimation**: The IMU does not expose a quaternion, so tilt is estimated from the accelerometer gravity vector and is valid only quasi-statically. During dynamic motion the tilt estimate is unreliable. The IMU itself is now **optional for the dock / head-only path** (if the BNO055 / I2C is unavailable, `connect()` degrades to `imu=None` with a warning); `read()` then reports `tilt_valid=False`, and the FSM refuses to enter — and faults out of — STAND/WALK on an invalid tilt, so balancing modes still **require** a working IMU.

- **Peripheral Hardware**: Antennas, eyes, sound, and projector are exercised in tests via mocks/spies, and antennas + eyes were additionally validated on hardware during bring-up (see Hardware bring-up results). On hardware, antennas are consent-gated (see above); the eyes run a background idle blink plus clip cues once `connect()` runs; eyes/sounds/projector are each gated by their own `enable_*` flags / `--no-*` dock-demo switches.

## Design and Policy Documents

For the full animation system design, see `docs/animation_system_plan.md` in this repository.

For policy checkpoints and walking algorithms, see `POLICIES.md`.

## Testing

To validate before hardware deployment (a venv outside the repo is fine; set `OPEN_DUCK_ANIM_HOME` so the runtime tests find the core):

```bash
cd Open_Duck_Mini_Runtime
OPEN_DUCK_ANIM_HOME=/path/to/Open_Duck_Mini python -m pytest tests/ -q
```

This runs all 71 runtime tests, including FSM transitions, mock hardware, animation playback, safety constraints, the antenna consent gate, the recovered eye wiring + idle-blink composition, and the IMU-optional tilt-validity gating. The `open_duck_anim` core library additionally has 247 tests in this repo (`python -m pytest tests/ -q` from the repo root).
