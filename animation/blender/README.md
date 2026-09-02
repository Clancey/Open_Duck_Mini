# Open Duck Mini Blender authoring

Install `duck_rig` as a Blender 4.x add-on. In the **Duck** 3D-view sidebar,
use **Build Open Duck Rig** and select
`mini_bdx/robots/open_duck_mini_v2/robot.urdf`. The generated bones retain
the URDF joint names, use their joint axes, and enforce the servo limits.

Animate the armature, then select its Action. Set `duck_loop`,
`duck_blend_in`, `duck_blend_out`, `duck_priority`, `duck_layer`
(`override` or `additive`), and comma-separated `duck_tags` in the Duck panel.
Choose an output directory and export. The exporter samples at 50 Hz by
default and only writes animated joints, so head clips remain head-only.

For batch/CI export:

```sh
blender motion.blend --background --python animation/blender/export_cli.py -- \
  --out animation/clips --all-actions
python animation/clips/verify_clips.py
```
