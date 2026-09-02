# Policy Checkpoints

This repository ships three trained neural network policies for bipedal locomotion, each with distinct head-tracking characteristics and safety envelopes.

## BEST_WALK_ONNX.onnx and BEST_WALK_ONNX_2.onnx

Pre-existing upstream walking policies. These were trained without active head passthrough (the head motor targets were set to zero during training), so the policy learned to ignore head commands entirely. Measured DC gain across all four head channels is approximately 0.

**Head envelope:** These policies do not track external head commands. Head motion in the runtime relies entirely on the additive path: the `open_duck_anim` runtime overlays head motion on top of the learned leg motion within a conservative safety envelope (envelope constants in `envelope.py`).

**Deployment:** Use when head tracking is delegated to runtime logic only.

## HEAD_PASSTHROUGH_300M.onnx

Trained 2026-09-02, 300M environment steps on `flat_terrain_backlash`, RTX 3090. Final reward: 254.1 (peak: 282.7 at 279M steps).

**Training difference:** Trained with the head passthrough **active** during training. The runtime equation `motor_targets[5:9] = command[3:7] + motor_targets[5:9]` was applied during rollouts, so the policy learned to balance the robot's legs while an external agent simultaneously drove the head via the command vector.

**Head envelope:** With this policy, the measured safe head envelope widens substantially:
- `head_yaw` dangerous side: −0.29 → −1.50 (unlocked)
- `neck_pitch`: −0.16 → −0.34 (doubled)
- Combined L2 budget: 0.55 → 0.70

**Locomotion improvement:** Stand tilt 2.70°, walk tilt 3.85°, no falls across all tested commands. Deflections that previously toppled the robot now survive the full trained envelope.

**Important:** The safety envelope constants in `open_duck_anim/envelope.py` are pinned to this specific checkpoint and are valid **only** when deploying `HEAD_PASSTHROUGH_300M.onnx`. If the deployed policy changes, the envelope constants must be re-derived via the measurement harnesses in `experiments/animation/`.

## References

- Full design and measurement methodology: `docs/animation_system_plan.md`
- Measurement harnesses and envelope tuning: `experiments/animation/`
- Source training: branch `feature/head-passthrough-training` @ `1c555e2` of a fork of `apirrone/Open_Duck_Playground`
