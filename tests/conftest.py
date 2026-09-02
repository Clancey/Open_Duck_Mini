"""pytest configuration: make the repo root and tests dir importable.

Adds the repo root (for ``open_duck_anim``), the tests directory (for the shared
``_helpers`` module) and ``blender/`` (for the bpy-free ``open_duck_anim_blender``
modules) to ``sys.path`` so tests run without installation.
"""

import os
import sys

_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_BLENDER = os.path.join(_ROOT, "blender")
for p in (_ROOT, _HERE, _BLENDER):
    if p not in sys.path:
        sys.path.insert(0, p)
