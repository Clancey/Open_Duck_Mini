"""Run the dock animation demo at the robot's 50 Hz control rate."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo.dock import DockMode
from duck_anim.joints import ANTENNA_JOINTS, HEAD_JOINTS, JOINT_INDEX


def _run_mujoco(dock: DockMode, duration: float) -> None:
    import mujoco
    import mujoco.viewer

    scene = Path(__file__).resolve().parents[2] / "mini_bdx/robots/open_duck_mini_v2/scene.xml"
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    joint_addresses = {
        name: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        for name in HEAD_JOINTS + ANTENNA_JOINTS
    }
    with mujoco.viewer.launch_passive(model, data) as viewer:
        deadline = time.monotonic() + duration
        while viewer.is_running() and time.monotonic() < deadline:
            target = dock.step(0.02)
            for name, address in joint_addresses.items():
                data.qpos[address] = target[JOINT_INDEX[name]]
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.02)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("print", "mujoco"), default="print")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mood", default="curious")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--clips", default=str(Path(__file__).resolve().parents[1] / "clips"))
    args = parser.parse_args()
    if args.backend == "mujoco":
        try:
            import mujoco  # noqa: F401
        except ImportError:
            print("MuJoCo is not installed; falling back to print backend.", file=sys.stderr)
            args.backend = "print"
    dock = DockMode(args.clips, seed=args.seed)
    dock.set_mood(args.mood)
    steps = max(1, round(args.duration * 50))
    if args.backend == "mujoco":
        _run_mujoco(dock, args.duration)
        return
    for step in range(steps):
        target = dock.step(0.02)
        if step % 10 == 0:
            values = ", ".join(f"{name}={target[JOINT_INDEX[name]]:+.3f}" for name in HEAD_JOINTS + ANTENNA_JOINTS)
            print(f"{step / 50:5.2f}s {values}")
        time.sleep(0.02)


if __name__ == "__main__":
    main()
