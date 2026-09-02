# open_duck_anim_blender

Blender addon for authoring **Open Duck Mini v2** animation clips and exporting
them as `.duckanim` files. It is a fork of
[`pollen-robotics/Open_Duck_Blender`](https://github.com/pollen-robotics/Open_Duck_Blender)
(Apache-2.0) that fixes four blocking exporter defects and adds clip metadata +
rig safety constraints. See [`NOTICE`](./NOTICE) for attribution and the full
change list, and [`LICENSE`](./LICENSE) for the Apache-2.0 terms.

This implements **Phase 2** of `docs/animation_system_plan.md`.

## What it fixes / adds

| Defect | Fix |
|---|---|
| **D2** antenna L/R swap | Correct canonical order (idx 9 = `antenna.l`, idx 10 = `antenna.r`) in `transform_table.py`; no double sign inversion. |
| **D3** hardcoded `[1,1]` contacts | Computed from foot geometry (`contacts.py`) + explicit `FootContactValid` marker / `contacts_valid` toggle. |
| **D4** wall-clock timer recording | Deterministic `scene.frame_set()` loop (`recorder.py`) — byte-identical re-records. |
| **D11** baked ±10° knee/ankle | Single explicit calibrated transform table (`transform_table.py`) + regression test. |
| Limit Rotation constraints | `jnt_range.py` + `constraints.py`, mirroring the MJCF `jnt_range`, idempotent. |
| Clip metadata panel | `panels.py` — `layer_mask`, blend times, `loop_mode`, `requires_mode`, `name`. |
| Export path | 59-float JSON → `open_duck_anim.compiler` → `.duckanim` (`export.py`). |

## Architecture: bpy-free core + thin Blender shims

All maths lives in **bpy-free** modules (import-safe without Blender, unit-tested
on CI): `transform_table`, `jnt_range`, `contacts`, `metadata`, `export`.
The Blender-facing shims import `bpy` guarded: `recorder`, `constraints`,
`panels`, and the addon `__init__`.

## Requirements

- **Blender >= 4.3.2** (the upstream `open-duck-mini.blend` rig is saved as 4.03).
  This addon's `bl_info` requires 4.3.2.
- The `open_duck_anim` library on Blender's Python path (`pip install
  ./packaging/open_duck_anim`, or add the repo root to `PYTHONPATH`).

## Getting the rig `.blend`

The `open-duck-mini.blend` binary is **not** vendored here (it is ~36 MB and
tracked with git-lfs upstream). Obtain it from upstream:

```bash
git lfs install
git clone https://github.com/pollen-robotics/Open_Duck_Blender.git
cd Open_Duck_Blender
git lfs pull   # materialise the .blend (verify it is a real Blender file, not an LFS pointer)
file open-duck-mini.blend   # -> "Blender3D, saved as ... version 4.03"
```

## Installing / running

Inside Blender >= 4.3.2, with `open_duck_anim` importable:

```python
import sys; sys.path.insert(0, "/path/to/repo")          # so open_duck_anim + this addon import
import open_duck_anim_blender
open_duck_anim_blender.register()                         # adds the "Duck Anim" sidebar panel
```

Or apply just the rig limits as a script:

```python
from open_duck_anim_blender import constraints
constraints.apply_limit_rotation_constraints("Armature")  # idempotent
```

Then in the **View3D > Sidebar > Duck Anim** panel: set the clip metadata, click
**Apply jnt_range Limits**, then **Record + Export .duckanim**. Output goes to the
configured directory as `<name>.source.json` (59-float authoring JSON) and
`<name>.duckanim`.
