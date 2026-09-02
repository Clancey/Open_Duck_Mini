# Dock demo animation

`DockMode` keeps Open Duck Mini visibly awake while it is parked.  Call
`step(0.02)` at 50 Hz; it returns the canonical 16-joint target array.  Its ten
leg entries are always byte-identical to the supplied docked pose.  Clips which
mention a leg joint are rejected during registration.

Run `PYTHONPATH=animation python animation/demo/run_demo.py --duration 20`.
`--backend print` is dependency-free. `--backend mujoco` visualizes the
head/antenna targets in the v2 scene and degrades to print output when absent.
