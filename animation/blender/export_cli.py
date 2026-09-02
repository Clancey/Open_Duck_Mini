"""Headless clip exporter: blender file.blend --background --python export_cli.py -- --out DIR."""

from __future__ import annotations

import argparse
import sys

import bpy

from duck_rig.exporter import export_action


def main() -> int:
    arguments = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Export Open Duck Mini actions")
    parser.add_argument("--out", required=True, help="Directory for .duckanim.json clips")
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--all-actions", action="store_true", help="Export every Blender Action")
    args = parser.parse_args(arguments)
    rigs = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    if not rigs:
        raise RuntimeError("No armature exists in this blend file")
    rig = rigs[0]
    actions = list(bpy.data.actions) if args.all_actions else [rig.animation_data.action if rig.animation_data else None]
    actions = [action for action in actions if action is not None]
    if not actions:
        raise RuntimeError("No Action to export; use --all-actions or assign an active Action")
    for action in actions:
        print(f"Exported {export_action(rig, action, args.out, args.fps)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Open Duck Mini export failed: {error}", file=sys.stderr)
        raise SystemExit(1)
