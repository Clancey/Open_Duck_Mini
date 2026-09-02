# open_duck_anim

Numpy-only robot animation blending engine for the **Open Duck Mini v2**
(Phase 1 of `docs/animation_system_plan.md`).

This is a standalone distribution for the `open_duck_anim` package whose source
lives at the repository root. It is kept separate from the repository's existing
`mini-bdx` packaging (`setup.cfg`) on purpose — see the comments in
`pyproject.toml`.

## Install

From the repository root:

```bash
pip install ./packaging/open_duck_anim          # runtime (numpy only)
pip install "./packaging/open_duck_anim[dev]"   # + pytest
```

## Runtime dependencies

`numpy` only. Designed to install and run on a Raspberry Pi Zero 2W (ARM),
Python 3.9+.

## Tests

```bash
pip install "./packaging/open_duck_anim[dev]"
pytest tests/
```

## Modules

- `joint_order` — canonical 16-joint order, 14-DOF Feetech bus order, conversions.
- `clip` — `.duckanim` loading + strict validation + `DuckAnimClip`.
- `compiler` — deterministic one-way 59-float authoring JSON → `.duckanim`.
- `blend` — three-layer animation engine (background / triggered / additive).
- `transform` — absolute head pose → relative command transform.
- `limits` — joint clamp / rate limit / antenna slew safety utilities.
