# Episodic standing full-body wiggle — training findings (negative result)

**Status: attempted, diagnosed, NOT trainable on the current training stack. Enhancement, not a blocker — the owner already has a working docked wiggle (`experiments/animation/clips/dock_wiggle.duckanim`).**

This is the handover for an attempt to train a Disney-style *episodic* policy
(arXiv:2501.05204) so the Open Duck Mini V2 performs a **standing full-body happy
wiggle while balancing** — the architecturally-correct way to animate legs on a
balancing biped. It is a **preview-quality proof-of-pipeline** effort (~300M steps
≈ 1.2 h on the RTX 3090), not a show-grade clip.

The code fixes are correct per the paper and are **committed regardless of the
training outcome** (they are prerequisites for *any* future episodic run). Four
training attempts all failed the same way. This document records precisely why,
so the next person does not repeat it.

Code lives on the **`episodic` branch of `Clancey/Open_Duck_Playground`** (never
`apirrone/*`). Local clone used this session:
`~/.copilot/session-state/9d7d4839-.../files/episodic/playground-episodic`.

---

## TL;DR

- **The three paper-divergence fixes to `episodic.py` are done, unit-tested (5/5),
  and committed.** Monotonic phase with N=50 Gaussian bases; φ=1 forced
  termination; command removed from the observation (**episodic obs = 142**, walking
  stays 101). These are genuinely useful and should not be lost.
- **The ONNX export bug (`'dict' object has no attribute 'policy'`) is FIXED and
  proven** — in-loop and offline, valid `obs[1,142] → continuous_actions[1,14]`.
- **The real root cause of the four failures is an *action-distribution runaway*,
  not terminations, not reward scale, and not the reference motion.** In every run
  the policy's action standard deviation runs away by 1–5 orders of magnitude
  (`policy_dist_max_std` climbs from ~0.3 to as high as **28,980**) while the mean
  saturates at ±1; sampled actions diverge, `action_rate` cost climbs, episode
  length erodes, and the robot topples. This is reference-independent: it recurs
  even with a balance-feasible 0.4-amplitude reference.
- **An earlier root-cause in `animation_system_plan.md` (Phase 6 / R19) blamed a
  "kinematically self-contradictory reference." Direct recomputation shows that is
  wrong for the shipped `standing_wiggle.json`** — its velocity fields are the
  correct derivatives of its own pose trajectory. That section has been corrected.
- **What episodic actually needs:** a **bounded / tanh-squashed action distribution
  with explicit std/KL control** (a training-stack change to Open_Duck_Playground's
  PPO, research-grade), paired with an **amplitude curriculum** (knob already
  implemented). Not hyperparameter tuning.

---

## 1. The three paper-divergence fixes (committed, verified)

The `episodic` branch was, before this work, "a periodic environment wearing an
episodic name." All three fixes are on `Clancey/Open_Duck_Playground@episodic`
(commit `7eda615` plus follow-ups), with unit tests in
`playground/tests/test_phase_encoding.py` (5/5 passing).

| # | Paper requirement | Before | After |
|---|---|---|---|
| **1. Phase encoding** | §V ¶2 + Appendix A ¶2: φ **monotonically increasing** on [0,1], encoded with **N=50 Gaussian basis functions** `exp(−(φ−φ_i)²/(2σ²))`, φ_i equally spaced — "highly local in time." | `imitation_i % nb_steps` (cyclic) encoded as a single `[cos 2πi/N, sin 2πi/N]` pair. A sin/cos pair cannot represent a one-shot clip's local structure. | Monotonic φ = `imitation_i / (clip_len−1)` ∈ [0,1]; 50 Gaussian bases, φ_i on a uniform grid. Verified monotonic 0→1, bases correctly placed and local. |
| **2. Forced end transition** | §V ¶2 + §V-A ¶2: phase rate = 1/clip-duration; **φ=1 must terminate the episode** ("Once the motion ends, a transition is forced"). | `step > 500` resampled the command and reset — a rollout-length hack unrelated to the clip. | `clip_done = imitation_i ≥ clip_len−1`; `done = _get_termination \| clip_done`. Verified φ=1 terminates exactly once at clip end. |
| **3. No command in obs** | Eq. 4 `x_t = f^epis(f_t, φ_t)` has **no `g_t` argument**; §VI-B: "no additional user input until the episodic motion finishes." | A command was still sampled and concatenated into the observation — pure noise in the advantage estimate. | Command removed from the episodic observation. Verified absent. |

**Additional, paper-aligned changes on the branch:**
- **Stronger terminations (§V-B):** terminate on *head or torso in contact with the
  ground* **and** *head–torso self-collision* (matters for an aggressive standing
  wiggle), in addition to the original gravity-z fall check.
- **Eq. 13 phase-windowed reward boost:** `w̃(φ) = w_0 + I[φ_start<φ<φ_end]·w_extra`
  applied to angular-velocity tracking over the shake window
  (`shake_start=0.25, shake_end=0.85`), per Appendix A ¶3's "excited motion."
- **Reset-yaw consistency fix** and a **reward rebalance** toward balance.
- **Amplitude-curriculum knob** (`REFERENCE_AMPLITUDE_SCALE`,
  `episodic_reference_motion.py`) — see §4.

### Observation contract (important)

Removing the command **changes the episodic observation size**, which is fine for a
*separate* episodic policy but makes it **NOT interchangeable** with the walking
policy:

| policy | obs | actions |
|---|---|---|
| walking (unchanged) | `[1, 101]` | `[1, 14]` |
| **episodic (this work)** | **`[1, 142]`** | `[1, 14]` |

The runtime must **dispatch on observation length** to pick the right policy. The
walking policy and its 101/14 contract are untouched.

---

## 2. Diagnosis findings

All diagnosis below is from **CPU instrumentation and the TensorBoard logs of the
four runs** (no GPU was used for the diagnosis). The user's two standing hypotheses
— "terminations fire too early" and "the reference is too static to imitate" — were
both tested directly and **both are falsified**. The evidence points at the PPO
action distribution instead.

### 2a. Mean episode length and termination reasons — NOT the fault

- **Zero-action baseline survives the entire clip: 150/150 frames.** A do-nothing
  policy stands through the whole reference without tripping any termination.
- **Reference geometry is benign.** Replaying the home pose plus all 150 reference
  frames trips **0/150** terminations. Minimum head–torso distance is **≥0.196 m**
  vs the `0.10 m` self-collision threshold — a **2× margin**. The reference never
  commands a self-colliding or ground-contacting pose.
- **The trained policy's short episodes are a *symptom*, not a cause.** Mean episode
  length is ~26 steps and it falls via `torso_ground` / tilt — i.e. it topples
  *itself*. **Self-collision never fires. φ=1 never fires early.** The new
  terminations are not killing the episode prematurely.

**Verdict:** terminations are not the fault. Hypothesis falsified with evidence.

### 2b. Per-term reward breakdown

- Run `...234715` (the notorious **−540, flat** run the logs keep surfacing) had a
  genuine bug: the **`imitation` term dominated at −19,780 → −28,170** — unbounded
  negative, swamping every other term. The user's "one term dominating at a huge
  magnitude" instinct was **correct for that run**.
- **That was fixed** (bounded `exp`-based imitation sub-rewards in `[0, w]`,
  `custom_rewards.py`). Under the current config the imitation term is bounded to
  ~**+124** and the total reward starts **positive (~+8.7)**. **Fixing the reward
  scale did not fix convergence** — which is what pointed the investigation away
  from reward shaping and toward the policy distribution.

> **Note on stale logs:** the single append-only log
> `/mnt/user/open_duck/logs/episodic-wiggle-train.log` spans *all* runs, so the
> pre-fix `−555` / ONNX-error lines persist and were repeatedly re-read as if
> current. Always confirm a **live container + newest checkpoint dir** before
> trusting a cited `STEP` line.

### 2c. Is the reference motion fit for purpose? — YES on consistency; modest in amplitude

The reference `standing_wiggle.json` (150 frames × 59 floats, 50 fps, 3.0 s) was
recomputed against its own pose trajectory. The convention-free anchor is decisive:

| field | check | result |
|---|---|---|
| **joints_vel** (`35:51`) | vs finite-difference of `joints_pos` (`7:23`) | **corr 1.000, max abs err 0.000** — exact |
| **lin_vel** (`29:32`) | stored vs `d(root_pos)/dt` | both **exactly zero** — the root genuinely does not translate, so `lin_vel = 0` is *correct*, not a bug |
| **ang_vel** (`32:35`) | stored vs world-frame `2·q̇·q*` from `root_quat` | **corr 1.000 on x and z** (0.375 / 0.126), y ≈ 0 — no transposition |

So the reference **is** kinematically self-consistent. It is a **modest,
joint-driven wiggle** (largest motion `head_roll` ≈ 0.32 rad; torso does not
translate; contacts genuinely `[1,1]` throughout, deliberately asserted, not the
old hardcoded default). That is a legitimate authoring choice for a standing wiggle
— the imitation target has real, imitable content, not noise.

> **This corrects an earlier root-cause.** `animation_system_plan.md` (Phase 6 / R19)
> previously blamed "angular-velocity x/z axes transposed" and "linear velocity
> zeroed while the root moves." Neither holds for the shipped file: `joints_vel`
> matches exactly (a frame-convention-free check), the root does **not** move (so
> `lin_vel=0` is right), and `ang_vel` matches a **world-frame** derivation at
> corr 1.000. The earlier "transposed" finding came from deriving angular velocity
> in a **different frame/quaternion-order convention** than the one the file uses —
> a derivation artifact, not a file defect. One real open item remains: **nail down
> whether the env compares this world-frame `ang_vel` against a body-frame gyro**
> (benign while near-upright, but worth reconciling before the next run).

**Verdict:** the reference is fit for purpose on the dimension that was blamed. Its
only limitation is expressiveness (joint-led, no torso translation), which the
amplitude experiment (§4) shows is *not* what is stopping training.

### 2d. The actual root cause — action-distribution runaway

Extracted from the TensorBoard logs of every run. The signature is identical
everywhere: **`policy_dist_max_std` explodes, `action_rate` cost climbs,
`avg_episode_length` erodes, reward declines** — while `v_loss` often collapses
toward 0 (the *value* function fits fine; the *policy* diverges).

| run | `policy_dist_max_std` | `action_rate` cost | `avg_episode_length` | `episode_reward` | note |
|---|---|---|---|---|---|
| tb11 | 0.30 → **30.2** | 228 → 526 | 61 → 42 | 27.9 → 10.1 | v_loss 4.0e5 → 310 |
| tb12 | 0.43 → 19 → **NaN** | → 0 | 62 → **1.9** | 28.5 → 0.96 | full divergence to NaN |
| **tb13** | 0.32 → **28,980** | 461 → **1183** | 61.6 → 32.3 | 19.5 → −3.5 | entropy_loss → −1.4e8; v_loss → 0.0098 |
| tb120m | 0.27 → 10.4 | 180 → 369 | 47.6 → 33.5 | 20.9 → 8.7 | 120M-step run |
| tb234715 | 1.6 → 39.8 | 183 → 416 | 48.3 → 40.4 | −380 → −542 | + the imitation-scale bug (§2b) |
| **tb_amp (amp 0.4)** | 0.42 → **107.6** | 464 → **1250** | 61.9 → 47.3 | 19.6 → −4.0 | balance-feasible reference — *still* runs away |

The runaway persists across **every lever swept**: amplitude {1.0, 0.4}, learning
rate {3e-4, 2e-4, 1e-4}, entropy coefficient {+0.005 … −0.003}, clip {0.1, 0.2},
imitation weight {1.0, 0.5}, action_rate cost {−0.5, −1.0}, reward_scaling {0.1, 1.0}.

**Interpretation:** Open_Duck_Playground's Brax PPO uses an **unbounded Gaussian
action distribution**. On this hard, one-shot imitation objective the policy learns
that ever-larger actions (mean saturating at the ±1 action clip, std growing without
bound) momentarily reduce loss; there is no squashing or std regularization to stop
it, so the sampled actions diverge, the robot flails and topples, and episode length
erodes. `v_loss → 0` with `policy_dist_max_std → 10^4` is the textbook fingerprint:
the critic is fine; the actor's distribution is unstable. This is **structural to
the training stack**, which is exactly why no reward/phase/termination/reference
change fixed it.

---

## 3. ONNX export — FIXED and proven

**Symptom:** `[warn] ONNX export failed at step 0: 'dict' object has no attribute 'policy'`
at every checkpoint, so a full run produced nothing deployable.

**Cause:** `export_onnx.py` assumed the params were an object/namedtuple with a
`.policy` attribute (the shape the joystick runner hands it), but the episodic
runner's in-loop path passes a **plain dict** with different nesting.

**Fix (`playground/common/export_onnx.py`):**
- Guard with `hasattr(policy_params, "policy")` and handle the dict shape; the
  normalizer path handles both in-memory and dict params.
- Removed a **redundant second** `tf2onnx.convert.from_keras(..., output_path="ONNX.onnx")`
  call that hardcoded a filename and littered the working directory (commit `4c62501`).

**Proven working:**
- **In-loop:** a short 6M-step run with in-loop export enabled wrote valid
  `obs[1,142] → continuous_actions[1,14]` models at **step 0 and the 6M checkpoint**,
  no `.policy` error, validated locally with `onnxruntime`. It coexists with training
  memory (`TF_FORCE_GPU_ALLOW_GROWTH=true` + `XLA_PYTHON_CLIENT_PREALLOCATE=false` +
  `num_envs=4096`) without OOM.
- **Offline:** `export_episodic_onnx.py --checkpoint <dir> --obs_size 142` recovers
  **every** checkpoint on disk, including the exact `...234715` run
  (`episodic_234715_recovered.onnx`). **Any completed episodic run is exportable
  after the fact** — the export path is no longer a run-losing risk.

> The `np.cast` line tf2onnx emits under NumPy 2.x is a **cosmetic rewriter
> warning**, not a failure — the file is written regardless. (A local `numpy<2`
> venv never even prints it.)

---

## 4. What episodic would actually need (and what it would cost)

Being specific, per the "the phase encoding was wrong" bar, not "needs more tuning":

**X. A bounded / tanh-squashed action distribution with explicit std or KL control.
This is the core fix.** The observed failure is the actor's std diverging
(`policy_dist_max_std` to ~28,980) while the mean saturates at ±1. Replacing the
unbounded Gaussian with a **tanh-squashed normal** (SAC-style), or adding a **hard
std cap / KL penalty / std regularization** to the PPO update, removes the degree of
freedom that runs away. This is a **change to Open_Duck_Playground's PPO/policy
head**, not a hyperparameter — research-grade plumbing, but well-understood.

**Y. An amplitude curriculum ramped 0 → 1 across training.** The knob is
**implemented** (`REFERENCE_AMPLITUDE_SCALE` + `episodic_reference_motion.py`
blends each frame toward the static frame-0). On its own it does not fix the
runaway (proven: amp 0.4 still diverged), but paired with **X** it lets the policy
learn to *balance* before it must *track*, which is the right ordering for a
standing full-body clip.

**Supporting, cheaper items:**
- **Reconcile the `ang_vel` frame convention** (world vs body/gyro) between the
  reference and the env's measured base angular velocity before the next run —
  benign while upright, but should be nailed down.
- **Build a reference-velocity-consistency validator** as a pre-training gate. This
  reference passed it, but it is cheap insurance for every future clip (recompute
  `lin_vel`/`ang_vel`/`joints_vel` from the pose trajectory, reject on disagreement).
- **Authoring rule:** a clip authored for one mode (docked) is not automatically a
  valid reference for another (standing) — re-derive root/velocity channels and
  re-validate amplitudes for the target mode.

**Cost.** With **X** in place, a *preview* remains cheap — Disney's own preview
threshold is ~300M steps ≈ **1.2 h** on the RTX 3090 at ~43k steps/s. Getting **X**
right is the real spend (a training-stack change + a stabilization run or two).
Disney's **show-grade** figure is **19.7B steps ≈ ~80 h** on our 3090 — that just
buys far more of the same once training is stable.

---

## 5. Sim/hardware status

**Sim evaluation only. No hardware.** No policy reached a recognisable upright
wiggle, so there is nothing worth evaluating on the robot, and a standing full-body
policy on a freshly-reset duck with uncalibrated joint offsets is a **supervised**
matter regardless. **The physical duck was not touched** (it is running the owner's
idle service). A future supervised hardware session would require, at minimum: a
policy that is *upright and recognisable in sim first*; verified joint-offset
calibration; the 142-obs runtime dispatch wired and tested; a person on the e-stop;
and a validated double-support entry/exit envelope with a trained recovery segment.

## Artifacts (kept out of git)

TensorBoard logs and `.onnx` files for all runs live under the session state dir
(`.../files/episodic/artifacts/`) and on the tower
(`/mnt/user/open_duck/checkpoints/`), not in this repo.
