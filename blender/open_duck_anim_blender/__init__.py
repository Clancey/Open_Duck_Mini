"""open_duck_anim_blender — Blender addon for authoring Open Duck Mini v2 clips.

A fork of ``pollen-robotics/Open_Duck_Blender`` (Apache-2.0) that fixes the four
blocking exporter defects (D2 antenna swap, D3 hardcoded contacts, D4
non-deterministic recording, D11 baked knee/ankle offsets), adds MJCF
``jnt_range`` Limit Rotation constraints and a clip-metadata panel, and routes
export through the ``open_duck_anim`` library (59-float authoring JSON →
``.duckanim`` compile). See ``NOTICE`` for the full list of changes and
attribution.

The pure-Python logic (transform table, contacts, metadata, frame assembly) is
in bpy-free modules so it is unit-tested on CI without Blender >= 4.3.2:

* :mod:`open_duck_anim_blender.transform_table` — bone→joint calibration (D2/D11)
* :mod:`open_duck_anim_blender.jnt_range`       — MJCF ranges → Euler limits
* :mod:`open_duck_anim_blender.contacts`        — foot contacts (D3)
* :mod:`open_duck_anim_blender.metadata`        — clip metadata + envelope warn
* :mod:`open_duck_anim_blender.export`          — 59-float assembly + compile

Blender-facing shims (import ``bpy`` guarded): ``recorder`` (D4 deterministic
loop), ``constraints`` (Limit Rotation), ``panels`` (UI/operators).
"""

from __future__ import annotations

bl_info = {
    "name": "Open Duck Anim Export",
    "author": "Open Duck Mini v2 contributors; forked from pollen-robotics/Open_Duck_Blender",
    "version": (0, 2, 0),
    "blender": (4, 3, 2),
    "location": "View3D > Sidebar > Duck Anim",
    "description": "Deterministic .duckanim clip authoring/export for Open Duck Mini v2",
    "warning": "Requires Blender >= 4.3.2 and the open_duck_anim library on the Python path.",
    "category": "Animation",
}


def register():
    from . import panels

    panels.register()


def unregister():
    from . import panels

    panels.unregister()


if __name__ == "__main__":  # pragma: no cover - run inside Blender
    register()
