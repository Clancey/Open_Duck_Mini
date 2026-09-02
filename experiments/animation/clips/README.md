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

Categories: **A** idle/alive loops · **AM** emotional mood loops · **B**
curiosity/attention · **C** expressive reactions (incl. yes/no) · **CE**
emotional one-shot beats · **CS** scared/fear beats · **D** walk-compatible ·
**E** dock-only full-body (legs move — restricted, see below).

| Clip | Cat | Dur | Loop | Mode | Prio | Path | What it is / when to trigger |
|------|:---:|----:|------|------|-----:|------|------------------------------|
| `idle_breathe` | A | 6.0s | wrap | any | 0 | parametric | Slow breathing-like neck bob. The **default background** "alive" loop under standing/docked. |
| `idle_scan` | A | 11.0s | wrap | any | 0 | parametric | Occasional slow head scan with holds over a breathing underlay. Long period so it never syncs with `idle_breathe`. |
| `idle_lookaround` | A | 8.0s | wrap | any | 0 | parametric | Restless micro weight-shifts and gaze wander; detuned so it never quite repeats. |
| `mood_content` | AM | 6.5s | wrap | stand | 0 | parametric | **Happy/content mood** to sit in: high bright head carriage, light quick rhythm, small bright wander, a perky persistent tilt, frequent quick blinks. Set as the base when things are going well. |
| `mood_sad` | AM | 9.5s | wrap | stand | 0 | parametric | **Sad/dejected mood**: low slow carriage, a slight persistent roll-tilt, long pauses, barely looks around, one slow heavy blink. Set on repeated failure / long neglect. |
| `mood_sleepy` | AM | 12.0s | wrap | stand | 0 | parametric | **Sleepy/drowsy mood**: very slow drift, head settles then rouses slightly, a slow loll, long droopy eye-closes. Set on low battery / late / long idle. |
| `mood_alert` | AM | 7.5s | wrap | stand | 0 | parametric | **Alert/attentive mood**: upright and still with small sharp scans and long stillness between; wide, rare blink. Set when watching for something. |
| `mood_grumpy` | AM | 8.5s | wrap | stand | 0 | parametric | **Grumpy/annoyed mood**: a persistent cocked tilt, lowered carriage, terse sharp dismissive turn-aways, terse blinks. Set when repeatedly interrupted / poked. |
| `mood_scared` | AM | 10.0s | wrap | stand | 0 | parametric | **Scared/frightened mood**: held small and withdrawn, tense frozen stillness broken by quick darting checks, wide eyes with rare blinks. Set on a persistent threat / an ongoing scary situation (use `calm_down` to exit it). |
| `curious_tilt` | B | 2.6s | once | any | 10 | blender | Inquisitive head-roll tilt held briefly, with a blink. Trigger on "notices something". |
| `look_toward` | B | 2.2s | once | any | 10 | blender | Directed look toward a point of interest, held, released. Trigger to point attention. |
| `double_take` | B | 2.4s | once | stand | 12 | blender | Glance away then a quick snap-back double-take. Stand-only (snappy). Trigger for a surprise it re-checks. |
| `perk_up` | B | 1.8s | once | stand | 15 | blender | Sudden alert perk-up: head lifts, antennas raise, brief scan. Trigger on "attention caught". |
| `scan_curious` | B | 4.0s | once | any | 10 | blender | Deliberate slow survey scan side-to-side and back. Trigger to "look for" something. |
| `nod_yes` | C | 2.2s | once | any | 20 | blender | **Emphatic "yes"**: anticipation lift + a sharp chin-down beat, 3 nods decaying to a settle. Built from `head_pitch` (~4× headroom), not `neck_pitch`. Legible across a room. Trigger on yes/acknowledge/confirm. |
| `nod_yes_soft` | C | 1.6s | once | any | 13 | blender | **Soft polite nod**: a single gentle dip. "Noted", not "YES". A quieter answer than `nod_yes`. Trigger on a low-key acknowledgement. |
| `shake_no` | C | 2.5s | once | stand | 20 | blender | **Decisive "no"**: a wind-up + firm alternating `head_yaw` swings (±0.42) that decay and settle dead-centre. Stand-only (yaw reads big). Trigger on no/refuse/reject. |
| `shake_no_reluctant` | C | 2.8s | once | stand | 13 | blender | **Reluctant "no"**: a slower, smaller shake with the chin sinking in aversion + a slight tilt. A hesitant "...no", a different message from the firm refusal. Trigger on an unwilling decline. |
| `happy_bounce` | C | 2.0s | once | stand | 18 | blender | Delighted bob with a bright antenna flick (event). Trigger on success/reward. |
| `sad_droop` | C | 3.2s | once | stand | 16 | blender | Dejected droop: head sinks, antennas fold back, slow settle with a slow blink. Trigger on failure/idle-too-long. |
| `startle` | C | 1.6s | once | stand | 30 | blender | Startled recoil then a wary settle. **Highest priority** — preempts everything. Trigger on a sudden event. |
| `excited` | CE | 2.2s | once | stand | 22 | blender | Excited/delighted: sharp triple bob + quick wiggle + rapid double blink + bright antenna flicks. Trigger on a big reward / a favourite thing. `happy_bounce`'s louder cousin. |
| `grumpy_annoyed` | CE | 2.0s | once | stand | 21 | blender | Grumpy/annoyed: one sharp cocked turn-away + terse antenna fold. Trigger on a poke / a "no". |
| `confused_puzzled` | CE | 3.0s | once | any | 12 | blender | Confused/puzzled: the quizzical **double** head-tilt (roll one way then the other) + asymmetric antennas (one up, one down) + slow blink. Trigger on an unexpected / ambiguous input. |
| `proud_pleased` | CE | 2.6s | once | stand | 18 | blender | Proud/pleased: a dignified slow chest-puff — chin up, antennas raised and held, slow content blink. Trigger on completing something well. |
| `timid_shy` | CE | 3.0s | once | stand | 16 | blender | Timid/shy: shrink and turn away with a tilt, antennas fold, then a small shy peek back. Trigger on a stranger / being told off. |
| `disappointed` | CE | 3.0s | once | stand | 16 | blender | Disappointed: a small hopeful lift, then a slow let-down sink with a sigh and a turn-away. The anticipation beat separates it from `sad_droop`. Trigger on a near-miss / a broken promise. |
| `suspicious_wary` | CE | 3.4s | once | stand | 14 | blender | Suspicious/wary: a held cocked tilt + a slight lean-in + a slow narrow scan + wary half-back antennas. Trigger on something it doesn't trust. |
| `sleepy_yawn` | CE | 3.6s | once | stand | 17 | blender | Sleepy yawn: a big slow back-and-up stretch, a long eye-close (the yawn), antennas stretch then flop, drowsy settle. Trigger on going idle / low battery. |
| `affectionate` | CE | 2.8s | once | any | 14 | blender | Affectionate: a warm tilt-lean nuzzle with a soft bob and a slow content blink. Trigger on being greeted / petted / a familiar face. |
| `flustered` | CE | 2.2s | once | stand | 20 | blender | Flustered/embarrassed: a wavering look-away + tuck, with the rapid flutter in the **antennas + a double blink** (not the head — see below). Trigger on being caught out / an error it "notices". |
| `content_sigh` | CE | 2.8s | once | any | 12 | blender | Content sigh: a gentle lift then a slow relaxed exhale settle. A quiet positive beat. Trigger on settling down / a job well done. |
| `greeting` | CE | 2.4s | once | stand | 18 | blender | Greeting: a friendly "hello" — a warm double bob with a tilt + a bright antenna raise + a happy double blink. Trigger on a person appearing. |
| `flinch` | CS | 2.4s | once | stand | 26 | blender | Flinch: a fast aversive recoil (head snaps back and averts, ears pin, eyes wide) then a **slow tentative return**. The wary recovery is what separates it from `startle`. Trigger on a near-miss / a sudden looming thing. |
| `cower` | CS | 3.0s | once | stand | 25 | blender | Cower: shrink small — head tucked and turned away, ears pinned back and held, eyes wide, a tiny tense micro-shift. A **sustained** fear pose. Trigger on a persistent threat / being loomed over. |
| `nervous_lookaround` | CS | 3.2s | once | stand | 23 | blender | Nervous look-around: tense quick darting threat-checks over a withdrawn wary carriage, ears half-back, eyes wide. Trigger on "did I hear something?" / feeling unsafe. |
| `calm_down` | CS | 3.4s | once | stand | 19 | blender | Calm down (**recovery**): release from fear to neutral — un-tuck, face forward, ears un-pin and relax, a flurry of relieved blinks. Trigger to **exit** fear so the emotion never looks stuck; also bridges `mood_scared` → a neutral/content mood. |
| `walk_look_around` | D | 7.0s | wrap | walk | 5 | parametric | Gentle gaze wander to overlay **while walking**. Small, legible, seamless loop. |
| `walk_alert` | D | 2.0s | once | walk | 15 | blender | Contained "something caught my eye" alert usable mid-stride. |
| `dock_wiggle` | E | 3.0s | once | **dock** | 25 | dock | **Happy full-body wiggle — dock only.** Hips lead a decaying side-to-side wag, body rocks, head + antennas trail, eyes bright. The **only** clip that moves the legs. Plays **exclusively** in `DOCK_DEMO`; rejected everywhere else at compile *and* runtime. **Deliberately triggered only — never in the idle service.** |

`idle_alive.duckanim` (the original reference clip, 4.0s wrap, `any`, prio 0)
also lives here and is covered by the same tests.

### The emotional palette (categories AM and CE)

The duck can now **be** in a mood and **do** emotional beats, not just react.

* **Mood loops (`mood_*`, category AM)** are the real unlock: seamless `wrap`
  loops at `priority 0` that the duck can *sit in* for minutes, so it can be
  happy / sad / sleepy / alert / grumpy / **scared** as an ambient state. Each
  reads distinctly at a glance from **posture and rhythm alone** — higher vs
  lower head carriage, a persistent tilt, fast-and-bright vs slow-and-heavy
  timing, and blink cadence. They are `requires_mode = "stand"` (standing **or**
  docked, not walking): a mood is a *resting* state, and the walking gait swamps
  its subtle motion (the phase-4 head-follow check confirmed the subtlest moods
  do not read over a gait), so while walking the neutral idles + `walk_look_around`
  take over. Antennas are held flat at rest in every mood loop (owner decision,
  enforced by the guard test); the blink cadence carries the eye expression.
  Set the mood from application state; a triggered reaction (any `priority > 0`)
  preempts it and it blends back when the reaction finishes.
* **One-shot beats (category CE)** extend the triggered reactions across a full
  emotional range. They are distinguished from one another by **energy and
  timing** as much as direction — `excited` and `grumpy_annoyed` use similar
  amplitudes; sharpness, rhythm and recovery time separate them. Antennas *are*
  used here (brief crisp flicks/folds read as ears and are momentary, not the
  sustained loop buzz the owner objected to).

### Saying yes and no (category C) — communicative, not decorative

`nod_yes` / `shake_no` are treated as **first-class communicative gestures**: the
duck should answer a question legibly from across a room. Two things make them read:

* **Built from the axis with headroom.** A nod is `head_pitch` (which uses only
  ~¼ of its derated range), **not** `neck_pitch` (which is effectively maxed).
  An earlier note that `nod_yes` was "fighting the envelope" was a mis-diagnosis —
  it had conflated the nod with the maxed neck axis. `nod_yes` now peaks at 0.24
  rad of `head_pitch` (‖c/L‖ 0.65, no clamp). A shake is `head_yaw` (huge
  headroom), now ±0.42 rad for a decisive, unambiguous read.
* **Properly shaped, not a sine.** A nod has a small **anticipation** lift, a
  **sharp down-beat** with a gentler recovery, and **three nods of decaying
  amplitude** that settle. A shake has a wind-up then firm **alternating swings
  that decay** and return dead-centre. The swing rate is held ~1.3 Hz so the soft
  kp=8 head servo still tracks it (validated: `shake_no` weighted head-follow
  corr 0.95).
* **Intensity variants carry different messages.** `nod_yes_soft` (a single
  gentle dip) says "noted", not "YES". `shake_no_reluctant` (slower, smaller,
  chin sinking in aversion) is a hesitant "...no", a genuinely different message
  from the firm refusal. These read as distinct because timing/energy, not just
  direction, is what the viewer reads.

### Scared — a spike, a state, and a way out (category CS + `mood_scared`)

"Acting scared" is more than `startle` (a 1.6 s bidirectional spike). Fear is
built here as **all three**:

* **A state to sit in** — `mood_scared` (AM, `wrap`, prio 0): held small and
  withdrawn, tense frozen stillness broken by quick darting checks, wide eyes
  with rare blinks. Stillness punctuated by sharp checks reads as fear far better
  than continuous motion — so most of the loop is *still*, with just two quick
  darts. Antennas are silent (loop rule); the tense micro-tremor rides on the
  sub-follow-floor pitch/neck so it reads as tension without asking the servo to
  chase it.
* **Spikes/beats** — `flinch` (fast aversive recoil, **slow** tentative return —
  the wary recovery is what makes it fear, not startle), `cower` (a sustained
  shrink-small pose, ears pinned, held), `nervous_lookaround` (tense darting
  threat-checks). Antennas earn their keep here: a **fast pin-BACK** (ears
  flattened) reads unmistakably as fear, and eyes go WIDE (the `wide` event).
* **A way out** — `calm_down` releases fear back to neutral (un-tuck, face
  forward, ears un-pin, a flurry of relieved blinks). **Emotions that can only be
  entered look broken**; this is the believable exit, and it also bridges
  `mood_scared` → a neutral or content mood.

### Eyes are an emotional channel

Blink behaviour carries real emotional weight and the runtime supports three
cues (see `runtime/pi/idle_service.py`, `EyeDriver`):

* the per-frame `eyes` track (0 = closed, 1 = open) — a **longer closed window
  is a slower, heavier blink**. This is how `mood_sad` / `mood_sleepy` /
  `sleepy_yawn` get their heavy, drowsy lids and how the mood loops set their
  blink *cadence* (happy = frequent quick blinks, sad = one slow heavy blink,
  sleepy = long droopy closes, alert = rare);
* the `("eye", "wide"|"alert"|"startle"|"open", t)` event → **wide, held ~1 s**
  (fear / alert / surprise);
* the `("eye", "happy"|"double"|"double_blink", t)` event → a **rapid double
  blink** (excited / delighted / flustered).

**Runtime gaps worth closing (named, not faked):** there is no **squint /
half-closed held** cue, no **variable-speed blink** event, and no **sustained
wide / suppressed-blink** mode. `suspicious_wary` would read sharper with a
squint; `mood_sad` would benefit from a genuinely *slow* blink event rather than
a long hard close; and **fear** (`mood_scared`, `cower`, `nervous_lookaround`)
wants eyes held **wide for longer than the ~1 s the `wide` event lasts, with
blinking suppressed** — real fear is wide-eyed and *doesn't* blink, then releases
in a burst (which `calm_down` does do). Today the fear clips hold the per-frame
`eyes` track open and fire `wide` where they can, which approximates it but the
wide doesn't persist across a multi-second cower. All are currently approximated
with the per-frame `eyes` track. If you extend the runtime, the three
highest-value additions are: a `squint` (hold the lids partly closed), a
`slow_blink` (eased close/open over a settable duration), and a `wide_hold` /
`fear` mode (sustained wide with suppressed blinking until released).

### Authoring emotion: build it from roll / pitch / timing / eyes — **not** `neck_pitch`

This is the single most important lesson from building this palette, and it is
backed by measurement, not taste. Peak-to-peak channel usage across the original
16 clips, measured against the ×0.5 derated envelope:

| channel | max ptp used | derated ptp available | headroom |
|---|---|---|---|
| `neck_pitch` | 0.10 | ~0.24 | **effectively maxed** — the binding constraint, hardware-confirmed |
| `head_pitch` | 0.20 | ~0.78 | **~4× unused** |
| `head_roll` | 0.16 | ~0.50 | **~3× unused** |
| `head_yaw` | 0.80 | ~1.50 | plenty spare |

So **do not try to express emotion through `neck_pitch`.** It is the one axis
with no room, and it is exactly the axis sadness/dejection instinctively reaches
for (the head wanting to *sink*) — which is why the original `sad_droop` reads
weakly. The expressive room is in the other channels:

* **`head_roll` (tilt) is the strongest and cheapest emotional signal** and was
  barely used before. Sympathy, confusion, curiosity, quizzicality, wariness all
  live in roll. A slight **persistent** tilt (asymmetric, off-centre) instantly
  reads as feeling something — `mood_grumpy`, `mood_sad`, `suspicious_wary`,
  `confused_puzzled` all lean on it.
* **`head_pitch` is head *carriage*** — high (chin-up, `head_pitch` negative)
  reads bright/proud/alert; low reads dejected/sleepy. This is where "sinking"
  should go instead of `neck_pitch`.
* **Timing conveys emotion more than amplitude.** The same head turn reads eager,
  neutral, or reluctant purely from its acceleration curve and hold time. Happy
  and angry can share amplitudes — sharpness, rhythm and recovery time separate
  them. Lean on this hard, given the `neck_pitch` ceiling.
* **The eyes** (blink cadence + the two events above) carry a large share of the
  read for almost no motion cost.

Two hardware realities shape the timing:

* **Head `kp` is a soft 8**, so the head servo lags and undershoots fast motion.
  `head_pitch` under-reaches ~0.08 rad at the top of its range against gravity.
  **Author for a soft, slightly laggy servo:** ease in and out, let poses settle,
  don't rely on crisp snaps landing exactly. Fast oscillations (≥~2.5 Hz) simply
  do not track — the phase-4 head-follow check *fails* clips that put fast jitter
  on the head. Put rapid/nervous energy in the **antennas and eyes** (which move
  fast) and keep the head clean and slower. `flustered` is authored exactly this
  way: a slow wavering look-away on the head, all the "flustered" speed in the
  antenna flicks + a double blink.
* **Asymmetry and imperfection read as alive.** Avoid symmetric arcs, exact
  repeats, and perfectly centred rest poses. Hold poses and let them settle with
  a tiny drift (reads as thinking/feeling); a pose that snaps and freezes reads as
  broken. A small counter-move before the main move (anticipation) and a slight
  settle after (overshoot) is what separates animation from interpolation.


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
so it settles rather than snaps. The emotional **mood loops** blend in over
`0.5 s` so switching moods is a soft cross-fade rather than a jump.

## Antennas: never in idle loops (explicit owner decision)

**Please read this before adding any antenna motion to a looping/idle clip.**

The antennas are open-loop 9g-class hobby servos on GPIO (D13 left, D12 right).
On the physical robot the owner reported they are **audibly noisy**: PWM hobby
servos buzz and chatter in proportion to how **often** they are driven — not
merely how far they travel. On a small desk/dock robot that continuous buzz is
intrusive and *undermines* the "alive" effect rather than adding to it. This is
mechanical/acoustic, not a software bug.

**Owner decision (watching the robot):** *"for the idle animations, I don't want
to use the antennas. They are very noisy."* Taken literally and completely: a
dock idle loop may run for many minutes continuously on a desk, so **any**
antenna motion inside a loop is effectively continuous noise. An earlier pass
only *reduced* idle antenna motion — that was not enough. The rule now is
absolute.

So the library treats antennas as **punctuation, not a heartbeat**:

* **Zero in every looping/idle clip.** Every clip with `loop_mode: "wrap"` (and
  any background/idle layer, `priority` 0) holds the antennas at a **flat
  constant** at the neutral rest value for the whole clip — the compiled
  `show_functions.antenna_left` / `antenna_right` tracks are genuinely constant,
  so the runtime issues no changing antenna command and the servos are never
  asked to move. This covers `idle_breathe`, `idle_scan`, `idle_lookaround`,
  `idle_alive`, and `walk_look_around`. There is no "gentle drift", no "single
  small lift", no micro-motion — exactly zero. **Do not "improve" this by adding
  idle antenna motion back in.** The guard test below makes it a hard failure.
* **Motion reserved for brief triggered gestures.** Antennas move **only** in the
  short `once` reactions where the gesture *carries* the read and is over in a
  moment: the `startle` snap, the `happy_bounce` flick, the `sad_droop` fold, the
  `perk_up`/`walk_alert` raise. A short crisp flick reads as delight/alarm and
  then stops; it is not the sustained buzz the owner objected to. These are
  deliberately kept.
* **Slew-capped in depth.** The global antenna slew cap (`DEFAULT_ANTENNA_SLEW`
  in `open_duck_anim/limits.py`) was lowered from an arbitrary `8.0` to `4.0`
  normalised units/s (a full `[-1,1]` traversal now takes ~0.5 s instead of
  ~0.25 s) so that *any* clip — including future ones — cannot drive the
  antennas harshly regardless of what its tracks request. The shipped clips are
  authored to sit **within** this cap (peak authored slew ~3.8 units/s), so the
  runtime limiter is a no-op on them today; it only bites on pathological or
  future over-driven motion. `tests/test_limits.py` pins the constant.

If a future author wants livelier antennas, add a *brief, purposeful* gesture to
a specific **triggered** reaction (a `once` clip) and keep the peak slew under
the cap — **never** add antenna motion to a looping/idle clip. The head — not the
antennas — is what makes the duck feel alive.

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

## Dock-only full-body clips (category E) — why they are restricted

Every clip above this section is **head-masked**: it drives only the head and
antennas and holds the legs constant. That is load-bearing. The RL locomotion
policy owns the legs whenever the robot is standing or walking, so a head-masked
clip is safe to play in **any** mode — docked, standing, or layered over a walk.
Animating the legs breaks that guarantee: naive additive leg motion will
destabilise a balancing biped, and leg motion during locomotion would need
residual RL with a bounded scale, not raw clip playback. So full-body clips are
confined to the one place where the legs are **not** load-bearing and no policy
is running: `DOCK_DEMO`, when the duck is docked or cradled and the dock carries
its weight.

`dock_wiggle` is the first (and, for now, only) category-E clip.

**The restriction is enforced twice, by construction, not by convention:**

* **Compile time** (`open_duck_anim/clip.py`). `layer_mask="full_body"` is
  accepted **only** with `requires_mode="dock"`. A full-body clip declared
  `any`, `stand`, or `walk` is rejected by the validator with an error naming
  the illegal leg channels — the dangerous clip is *unauthorable*, not merely
  discouraged.
* **Run time** (`open_duck_anim/blend.py`). The engine's mode × channel
  capability matrix refuses to *start* a full-body clip in any mode other than
  `DOCK_DEMO` (the trigger is dropped and counted). If the mode leaves `DOCK`
  **while a full-body clip is already playing**, the clip is taken through the
  normal controlled-abort (release) path — the legs ease back toward the dock
  hold pose and are then handed back to the policy — never a snap. The engine
  only ever emits `leg_targets` in `DOCK`; in any other mode `leg_targets` is
  `None` and the legs belong to the policy.

### What the legs are allowed to do, and why it is safe

There is no *measured* balance envelope for the legs the way there is for the
head — because balance is not a constraint when the weight is on the dock. But
joint limits, velocity limits, and **mechanical** safety still are, and the legs
can reach poses the head never could. So the leg envelope
(`open_duck_anim/leg_envelope.py`) is deliberately conservative:

* **Per-joint range** is sourced from the MJCF (`open_duck_mini_v2.xml`) and the
  motion is expressed as a bounded **deflection from the dock hold pose**, then
  intersected with the MJCF `jnt_range`.
* **Hip yaw and hip roll lead**; knee and ankle barely move. That is a
  self-collision / cable-strain choice: a large knee or ankle excursion is what
  could fold a shin or foot into the chassis or wrap a servo cable. Twisting at
  the hips keeps the legs sweeping in a safe cone. Authored deflection caps
  (×0.5 hardware-derated): hip yaw 0.10 rad, hip roll 0.06, hip pitch 0.05,
  knee 0.04, ankle 0.04.
* **×0.5 hardware derating**, the same convention the head envelope uses for
  first-hardware trials, so the runtime clamp is a no-op on a derated first run.
* The `max_motor_velocity = 5.24 rad/s` bus-target rate limit is applied to the
  final leg targets exactly as it is to the head.
* The **head channels of a full-body clip still respect the measured head
  envelope** — `dock_wiggle`'s head part is built from `head_roll`/`head_yaw`/
  `head_pitch` and timing, keeping the binding `neck_pitch` axis near zero.

### Self-collision check (done, not assumed)

`experiments/animation/phase4_dock_fullbody_sim.py` (a dock sibling to the
balancing `phase4_integrated_sim.py`) plays the compiled clip through the engine
in `DOCK` and, per control tick, measures the **signed distance** between every
pair of geoms on non-adjacent links with MuJoCo's `mj_geomDistance`. The shipped
MJCF only marks the two foot pads collidable and its CAD meshes are built to
touch without interpenetrating, so a naive contact count would be vacuous;
signed distance is not. The neutral hold pose is taken as a baseline (six pairs
of nested head/trunk meshes already overlap there by design), and the clip is
asserted to (a) never drive a clear pair to within a 5 mm contact floor and
(b) never deepen a design-overlap by more than 3 mm. A sensitivity self-test
perturbs the hips ±0.45 rad and confirms the metric moves by ~0.15 m, proving it
is live. Result for `dock_wiggle`: **PASS** — the wiggle introduces no new
approach; the closest genuine clearance (trunk ↔ knee) stays ~13 mm and every
leg/head channel is inside both the MJCF range and the 5.24 rad/s velocity cap.

Run it with the mujoco venv:

```
OPEN_DUCK_ANIM_HOME=<repo> \
  <phase4-venv>/bin/python experiments/animation/phase4_dock_fullbody_sim.py
```

### It must be deliberately triggered — never automatic

`dock_wiggle` is **not** in the `duck-idle` service's candidate lists and must
not be added to them. `duck-idle` is head-and-show only by design; a full-body
clip firing unattended on a robot that might not actually be docked is exactly
the failure this whole architecture avoids. Play it only as a deliberate,
attended action, with the robot **genuinely docked or cradled** and a hand near
the switch.

### Authoring a dock full-body clip

Use the separate entry point `experiments/animation/author_dock_clips.py` (kept
apart from `author_clips.py` so head-only authoring cannot accidentally emit leg
motion). `DockClipSpec` fixes `layer_mask="full_body"` and `requires_mode="dock"`
and the script self-checks the head against the derated **head** envelope and the
legs against the derated **leg** envelope (and the rate limit) before it writes:

```
python experiments/animation/author_dock_clips.py --only dock_wiggle \
  --out-dir experiments/animation/clips
```
