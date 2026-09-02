# Runtime Animation Engine Integration (Phase 4)

## Overview

This patch series captures the Phase 4 integration of the `open_duck_anim` animation engine into `apirrone/Open_Duck_Mini_Runtime`. The work is version-controlled here rather than directly in the runtime repository to preserve it during development and enable review, but the canonical target is the runtime repo's `v2` branch.

## What This Patch Contains

The Phase 4 integration includes:

- **Mode FSM**: A finite-state machine with states BOOT, DISARMED, ARMING, DOCK_DEMO, STAND, WALK, and a latched FAULT state. The machine transitions from DISARMED to ARMING with a measured-pose ramp (smooth transition from the current joint angles), then to mode-specific states. Faults latch and must be cleared with an explicit reset.

- **Minimum Safety Set**:
  - Controlled abort mechanism to stop the robot mid-animation
  - E-stop button integration
  - Watchdog timers to detect control loop timeouts
  - Duty-cycle-based thermal management (since the rustypot HWI exposes no temperature readings)

- **AnimationController**: The core runtime component that loads and plays `open_duck_anim` animation clips, providing frame-by-frame joint command synthesis.

- **RobotInterface Abstraction**: A hardware abstraction layer with a MockRobot test seam for validation without hardware. This layer exposes measured state (joint angles, IMU) and accepts joint commands.

- **Entry Points**:
  - `scripts/dock_demo.py`: Standalone demo that bypasses the policy entirely and directly plays animation clips on the docked robot
  - `scripts/v2_rl_walk_mujoco.py`: Extended with `--head_animation` overlay to compose head-tracking animations with walking policy outputs

- **Safety Enforcement**: Re-enabled 5.24 rad/s velocity clip on final bus targets to prevent unrealistic joint velocities.

- **Example Clip**: `clips/idle_alive.duckanim` — a short looping idle animation for testing.

- **Test Suite**: 49 unit and integration tests covering FSM transitions, animation playback, mock hardware, and safety constraints.

## Upstream Base

This patch applies to branch `v2` of `apirrone/Open_Duck_Mini_Runtime`:
```
https://github.com/apirrone/Open_Duck_Mini_Runtime
```

## How to Apply

Clone the v2 branch and apply the patch series:

```bash
git clone -b v2 https://github.com/apirrone/Open_Duck_Mini_Runtime
cd Open_Duck_Mini_Runtime
git am /path/to/runtime/patches/*.patch
```

After applying, the patch introduces:
- New mode FSM and state management in the runtime core
- The `AnimationController` class and hardware abstraction
- Test suite covering FSM, controller, and mock hardware behavior
- Demo and overlay scripts

## Dependencies

The runtime must have `open_duck_anim` importable on the Pi. Provide it either by:
- Installing the package: `pip install -e /path/to/open_duck_anim`
- Setting the environment variable: `OPEN_DUCK_ANIM_HOME=/path/to/open_duck_anim` before running the daemon

## Running the Dock Demo on Physical Hardware

The dock demo allows you to play animation clips directly on the robot without the walking policy.

### Prerequisites

1. On the Pi, after cloning and applying the patch:
   ```bash
   pip install -e .
   ```

2. Ensure `open_duck_anim` is importable (see Dependencies above).

3. Copy the animation clip to the Pi:
   ```bash
   scp clips/idle_alive.duckanim pi@<robot-ip>:~/
   ```

### Steps to Run

1. **Power the robot in the dock/cradle** with a hand near the physical power switch. This ensures safe failure modes.

2. On the Pi, start the demo:
   ```bash
   python scripts/dock_demo.py --clip ~/idle_alive.duckanim --derating 0.5 --max-demo-s 120
   ```

3. Control the robot interactively:
   - Press `a` to arm (the robot ramps from its measured current pose)
   - Press `d` to enter DOCK_DEMO mode (plays the animation clip)
   - Press `x` to e-stop (hard stop, resets to DISARMED on next arm attempt)
   - Press `r` to reset a latched fault state
   - Press `q` to quit cleanly

4. **For first trials, keep `--derating 0.5`** to limit motor current. Increase toward 1.0 only as hardware data accrues and you gain confidence in the robot's structural integrity.

5. Monitor the robot for:
   - Jerky motion (sign of clipping or safety constraints being hit)
   - Heat in the motor housings
   - Any audible grinding or unusual sounds
   - Foot slippage if docking contact is not perfect

## Known Limitations

The following could not be verified without hardware access:

- **Thermal Management**: The rustypot HWI does not expose motor temperature or current draw, so thermal management is duty-cycle-based only (duty cycle is reduced uniformly to limit sustained power). No real-time over-temperature shutdown is possible.

- **Foot Contact Sensing**: This build has no foot-contact sensors, so contact state is hard-coded as `[1, 1]` (both feet always "in contact"). The animation clips were designed with this in mind, but the controller will not detect unexpected slips.

- **Tilt Estimation**: The IMU does not expose a quaternion, so tilt is estimated from accelerometer gravity and is valid only quasi-statically (low-frequency). During dynamic motion, the tilt estimate is unreliable.

- **Peripheral Hardware**: The real PWM antenna, eyes, sound, and projector wiring are exercised only via mocks. Integration of these peripherals is not included in this patch.

## Design and Policy Documents

For the full animation system design, see `docs/animation_system_plan.md` in this repository.

For policy checkpoints and walking algorithms, see `POLICIES.md`.

## Testing

To validate the patch before hardware deployment:

```bash
cd Open_Duck_Mini_Runtime
python -m pytest tests/ -v
```

This runs all 49 tests, including FSM transitions, mock hardware, and animation playback.
