"""pytest configuration: make the repo root and tests dir importable.

Adds the repo root (for ``open_duck_anim``) and the tests directory (for the
shared ``_helpers`` module) to ``sys.path`` so tests run without installation.
"""

import os
import sys

_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
