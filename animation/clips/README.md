# Starter clips

`generate_starter_clips.py` produces the committed, 50 Hz `.duckanim.json`
library using only Python and NumPy. Run it after changing motion generation,
then run `python verify_clips.py`. The verifier checks frame shapes, finite
values, URDF joint names and limits, and ensures override clips do not move
legs.

| Clip | Duration | Layer | Joints |
| --- | ---: | --- | --- |
| idle_breathe | 4.0s | override | neck_pitch, head_pitch |
| look_around | 4.0s | override | head_yaw, head_pitch |
| nod_yes | 1.2s | override | head_pitch |
| shake_no | 1.4s | override | head_yaw, head_roll |
| curious_tilt | 2.0s | override | head_roll, head_pitch, antennas |
| alert_perk | 0.8s | override | head_pitch, antennas |
| sad_droop | 2.5s | override | neck_pitch, head_pitch, antennas |
| antenna_wiggle | 1.0s | override | antennas |
| happy_bounce | 1.0s | additive | hip_pitch, knees, ankles, head_pitch |
| dance_sway | 2.0s | additive | hip_roll, ankles, head_roll |
