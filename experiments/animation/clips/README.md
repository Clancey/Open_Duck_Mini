# Open Duck Mini v2 — Head Animation Clip Library

Expressive, **head-only** `.duckanim` clips that make the duck feel alive. Every
clip drives only the head channels (`neck_pitch`, `head_pitch`, `head_yaw`,
`head_roll`) plus the antennas; the legs are always held constant. This is the
architecture, not a limitation: the RL locomotion policy owns the legs, so a
head-masked clip is safe to play **both** in `DOCK_DEMO` and while the robot is
standing or walking. Clips that move legs are rejected by the compiler.

All clips here are validated to sit inside the **×0.5 hardware-derated** safety
envelope (`open_duck_anim/envelope.py`), so first-hardware trials — which run
derated — will play them without the runtime having to clamp anything.

## The library

Categories: **A** idle/alive loops · **B** curiosity/attention · **C**
expressive reactions · **D** walk-compatible.

| Clip | Cat | Dur | Loop | Mode | Prio | Path | What it is / when to trigger |
|------|:---:|----:|------|------|-----:|------|------------------------------|
| `idle_breathe` | A | 6.0s | wrap | any | 0 | parametric | Slow breathing-like neck bob. The **default background** "alive" loop under standing/docked. |
| `idle_scan` | A | 11.0s | wrap | any | 0 | parametric | Occasional slow head scan with holds over a breathing underlay. Long period so it never syncs with `idle_breathe`. |
| `idle_lookaround` | A | 8.0s | wrap | any | 0 | parametric | Restless micro weight-shifts and gaze wander; detuned so it never quite repeats. |
| `curious_tilt` | B | 2.6s | once | any | 10 | blender | Inquisitive head-roll tilt held briefly, with a blink. Trigger on "notices something". |
| `look_toward` | B | 2.2s | once | any | 10 | blender | Directed look toward a point of interest, held, released. Trigger to point attention. |
| `double_take` | B | 2.4s | once | stand | 12 | blender | Glance away then a quick snap-back double-take. Stand-only (snappy). Trigger for a surprise it re-checks. |
| `perk_up` | B | 1.8s | once | stand | 15 | blender | Sudden alert perk-up: head lifts, antennas raise, brief scan. Trigger on "attention caught". |
| `scan_curious` | B | 4.0s | once | any | 10 | blender | Deliberate slow survey scan side-to-side and back. Trigger to "look for" something. |
| `nod_yes` | C | 2.2s | once | any | 20 | blender | Affirmative double nod. Small enough for any mode. Trigger on yes/acknowledge. |
| `shake_no` | C | 2.2s | once | stand | 20 | blender | Negative head shake. Stand-only (yaw reads big). Trigger on no/refuse. |
| `happy_bounce` | C | 2.0s | once | stand | 18 | blender | Delighted bob with a bright antenna flick (event). Trigger on success/reward. |
| `sad_droop` | C | 3.2s | once | stand | 16 | blender | Dejected droop: head sinks, antennas fold back, slow settle with a slow blink. Trigger on failure/idle-too-long. |
| `startle` | C | 1.6s | once | stand | 30 | blender | Startled recoil then a wary settle. **Highest priority** — preempts everything. Trigger on a sudden event. |
| `walk_look_around` | D | 7.0s | wrap | walk | 5 | parametric | Gentle gaze wander to overlay **while walking**. Small, legible, seamless loop. |
| `walk_alert` | D | 2.0s | once | walk | 15 | blender | Contained "something caught my eye" alert usable mid-stride. |

`idle_alive.duckanim` (the original reference clip, 4.0s wrap, `any`, prio 0)
also lives here and is covered by the same tests.

### Priorities & arbitration

Higher `priority` wins the head layer; equal priority means the newer clip wins
(plan §6.4). The idles sit at `0` so **any** triggered reaction preempts them;
`startle` at `30` preempts other reactions. When a triggered clip finishes it
releases and the background idle blends back in.

### Modes (`requires_mode`)

`any` clips are safe standing, docked, or walking. `stand` clips have amplitude
(or snap) that only reads well/safe when not walking. `walk` clips are the
small-amplitude variants meant to overlay a gait. Trigger logic should respect
`requires_mode` — e.g. don't fire `shake_no` mid-stride; fire `walk_alert`
instead.

### Blend times

Bodies default to `blend_in_s = blend_out_s = 0.35 s` (`T_alpha`) and show
functions (antennas/eyes) to `0.10 s` (`T_beta`). Exceptions are deliberate:
idles use `0.0` (they are the always-on background), `startle` blends **in**
fast (`0.05 s`) so the recoil lands, and `sad_droop` blends slowly (`0.4 / 0.5`)
so it settles rather than snaps.

## Safety envelope (why nothing is clamped)

The derated ×0.5 per-channel ceilings are roughly `neck ±0.12`, `head_pitch
±0.27`, `head_yaw ±0.52`, `head_roll ±0.17` rad, but the **binding** constraint
is the combined budget `||c/L||₂ ≤ 0.70` (normaliser `L` on the more dangerous
side of each channel; the budget is unchanged by derating). Every clip is
authored to peak at `||c/L||₂ ≤ ~0.67`, so the runtime envelope clamp is a no-op
even at full first-hardware derating. Head `kp` is soft (8 vs 30 for legs), so
clips favour eased in/out motion over snappy steps that a laggy servo would
undershoot.

`tests/test_clip_library.py` re-checks all three properties (schema-valid,
head-masked, within the derated envelope) for **every** clip in this directory,
so a future over-amplitude or broken clip cannot ship silently.

## How to add a new clip

1. Open `experiments/animation/author_clips.py`. Near the top of `build_specs()`
   add a `ClipSpec(...)` with per-channel tracks. The curve helpers are:
   `keys([...])` (eased keyframes: `"ease_in"`, `"ease_out"`, `"smooth"`,
   `"hold"`, `"linear"`), `sine(freq, amp, phase)`, `drift(...)` (a sum of
   detuned sines for non-repeating idles), `pulse(t, width, amp)`, `const(v)`,
   and `+`/`*` composition. Head tracks are in **radians**; antenna tracks are in
   **normalised `[-1, 1]`** (the compiler does the antenna calibration — don't
   hand-roll it).
2. Set `authoring_path="blender"` for keyed/expressive clips (recorded on the
   real rig) or leave the default `"parametric"` for procedural loops.
3. Keep it inside the derated envelope. The script self-checks and refuses to
   emit a clip that would clamp (unless `--allow-clamp`). Aim for
   `||c/L||max ≤ ~0.67`.
4. Generate the clip into this directory:
   * Parametric: `python experiments/animation/author_clips.py --backend procedural --only <name> --out-dir experiments/animation/clips`
   * Blender: `"/Applications/Blender.app/Contents/MacOS/Blender" --background --python experiments/animation/author_clips.py -- --backend blender --only <name> --out-dir experiments/animation/clips`
5. Validate it in sim (stand and, if not dock/stand-only, walk):
   `python experiments/animation/phase4_integrated_sim.py --onnx HEAD_PASSTHROUGH_300M.onnx --all experiments/animation/clips --deratings 0.5,1.0 --out <artefacts-dir>`
6. Run the guard: `pytest tests/test_clip_library.py`.

If a clip fails validation or gets clamped so hard it no longer reads as the
intended emotion, **re-author it smaller** — do not ship it clamped.

## Authoring paths

Both backends feed the identical 59-float source into `open_duck_anim.compiler`,
so their output is format-identical and numerically equivalent (they differ only
by Blender's float32 round-off, ~1e-7 rad — verified with
`author_clips.py --verify-identical`):

* **blender** — the clip is keyframed on the real rig
  (`open-duck-mini.blend`, 49 bones) and recorded headless via the deterministic
  `frame_set()` recorder. Used for the keyed curiosity/reaction clips. This
  exercises the real content pipeline end-to-end.
* **parametric** — the 59-float source is generated directly from the curve
  engine (no Blender). Used for the breathing/scan/wander **loops**, which are
  naturally expressed as detuned sines and are easier to keep seamless this way.
