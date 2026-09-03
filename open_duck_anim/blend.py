"""Three-layer animation engine (plan §6.1-§6.4).

Layers (Disney §VI-A, plan §6.1):

1. **Background loop** — always-on idle (blinks, antenna idle), weight 1.
2. **Triggered clips** — at most one *owns* the head/show layer at a time; blended
   over the background with body weight ``alpha`` (``T_alpha`` ramp) and show
   weight ``beta`` (``T_beta`` ramp).
3. **Joystick layer** — an additive offset on the head channels (plan §6.3).

Key contracts implemented here:

* **Clock ownership (plan §6.1 / S1).** Phase advances from an elapsed monotonic
  timestamp ``t_now`` passed in by the caller — never an internal timer, never by
  counting frames. :meth:`Engine.evaluate` is a function of (state, ``t_now``,
  triggers). Skipped frames on an overrun jump the phase to the correct time and
  any discrete ``events`` crossed during the gap fire **exactly once**, in order.
* **Arbitration (plan §6.4 / S4).** ``priority`` decides ownership (higher wins,
  ties → newer). A preempting clip's blend-in starts from the **current blended
  output**, not the background — modelled by keeping the outgoing clip as a
  *releasing* layer and composing layers sequentially, so at the switch instant
  the composite is unchanged (no snap).
* **Ramps (plan §6.4).** Linear ``interp`` for joint angles (no slerp, DF3).
  Overlapping blend-in/out are clamped so a clip reaches full weight for at least
  one frame (:func:`clamp_blend_times`).
* **Loop modes (plan §5.2).** ``wrap`` / ``once`` / ``clamp``.

Design choices where the plan is silent:

* ``eyes`` (a discrete blink state) is not blended; the dominant layer (show
  weight ≥ 0.5) with an ``eyes`` track wins, else the background value.
* Discrete ``events`` fire from the always-on background loop and from the
  current *active* (non-releasing) owner clip; a clip that is preempted or
  cancelled has its pending events cancelled (plan §6.5-a).
"""

from dataclasses import dataclass, field
from typing import List, Optional
import math

import numpy as np

from .clip import DuckAnimClip, DiscreteEvent
from .joint_order import HW_ORDER_14, HEAD_SLICE_16, INIT_POS_14, LEG_INDICES_16
from .transform import TRAINING_LOW, TRAINING_HIGH, NOMINAL_HEAD_POSE
from .envelope import HeadEnvelope, DEFAULT_ENVELOPE
from .leg_envelope import LegDockEnvelope, DERATED_LEG_ENVELOPE
from .torso_envelope import (
    TorsoEnvelope,
    DEFAULT_TORSO_ENVELOPE,
    posture_to_command_offsets,
)
from .limits import JointRateLimiter, MAX_MOTOR_VELOCITY

# Timing constants (plan §6.6).
CTRL_DT = 0.02       # 50 Hz single clock
T_ALPHA = 0.35       # body blend
T_BETA = 0.10        # show blend
# Upper clamp on the elapsed dt used for the head slew guard (reviewer E6). A
# stalled control loop, a debugger pause, or a dropped tick can make the real
# wall-clock dt arbitrarily large; feeding that straight into ``slew_limit*dt``
# would authorise an effectively unlimited single-tick jump and silently defeat
# the rate guard. We cap it at a few control periods so a late tick degrades to
# "one generous but bounded step" rather than "no limit at all".
MAX_SLEW_DT = 5.0 * CTRL_DT  # 0.10 s == 5 ticks

# Modes (plan §6.2). Strings (Python 3.9-friendly; no Enum in the hot path).
MODE_DOCK = "dock"
MODE_STAND = "stand"
MODE_WALK = "walk"
_VALID_MODES = (MODE_DOCK, MODE_STAND, MODE_WALK)

# Leg DOF indices within the 14-DOF hardware order (everything but head 5..8).
_HEAD_NAMES = {"neck_pitch", "head_pitch", "head_yaw", "head_roll"}
LEG_HW_INDICES = tuple(i for i, n in enumerate(HW_ORDER_14) if n not in _HEAD_NAMES)
_DOCK_LEG_HOLD = INIT_POS_14[list(LEG_HW_INDICES)].copy()
_DOCK_LEG_HOLD.setflags(write=False)


@dataclass
class Triggers:
    """Trigger snapshot read atomically at the top of a tick (plan §6.1)."""

    clips: List[DuckAnimClip] = field(default_factory=list)
    joystick_offset: Optional[np.ndarray] = None
    cancel: bool = False


@dataclass
class TickShow:
    """Per-tick show-function output (plan §6.4 ``ShowOutput``)."""

    antenna_l: float
    antenna_r: float
    eyes: int
    events: List[DiscreteEvent] = field(default_factory=list)


@dataclass
class EngineOutput:
    """Return value of :meth:`Engine.evaluate` (plan §6.4)."""

    head_command_offsets: np.ndarray            # (4,) added to commands[3:7]
    show: TickShow
    leg_targets: Optional[np.ndarray] = None    # (10,) DOCK_DEMO only, else None
    head_targets: Optional[np.ndarray] = None   # (4,) DOCK_DEMO only, else None
    # (3,) [torso_height_delta, grav_x, grav_y] added to the standing policy's
    # torso command (commands[7:10]). STAND mode only (the torso-command standing
    # policy); None in WALK (deployed policy has no torso command) and DOCK (body
    # animated via ``leg_targets``). See :mod:`open_duck_anim.torso_envelope`.
    posture_command_offsets: Optional[np.ndarray] = None


def _ramp(x: float) -> float:
    """Clamp a linear ramp fraction to ``[0, 1]``."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x


def clamp_blend_times(blend_in: float, blend_out: float, duration: float, fps: int):
    """Clamp overlapping blend-in/out so full weight holds ≥1 frame (plan §6.4).

    A clip must reach weight 1 for at least one control frame, otherwise a
    wall-clock ``t_now`` will generally sample either side of the single instant
    where the ramps cross. We therefore require
    ``blend_in + blend_out <= duration - 1/fps`` and, when that is violated,
    scale both down to fit within ``duration - 1/fps``. Returns
    ``(blend_in', blend_out')``.
    """
    total = blend_in + blend_out
    one_frame = 1.0 / fps
    if total <= 0.0 or total <= duration - one_frame:
        return blend_in, blend_out
    avail = max(duration - one_frame, 0.0)
    scale = avail / total
    return blend_in * scale, blend_out * scale


@dataclass
class _ClipState:
    """Internal per-clip playback state."""

    clip: DuckAnimClip
    t_start: float
    bi_body: float
    bo_body: float
    bi_show: float
    bo_show: float
    releasing: bool = False
    t_release: float = 0.0
    w_body_at_release: float = 1.0
    w_show_at_release: float = 1.0
    # Event watermark: cumulative frame position already fired. Initialised well
    # below 0 (not -1e-9) so a frame-0 event fires on the first tick — in
    # float64 ``-1e-9 + 1e-9 == 0.0`` exactly, which would swallow frame 0 (C1).
    last_event_cum: float = -1.0

    # --- release control ---
    def begin_release(self, t: float) -> None:
        """Convert this layer into a *releasing* crossfade starting at ``t`` (H1).

        The outgoing weight is captured at ``t`` (while still active) so the new
        clip blends in from the current output (no snap, plan §6.4). The release
        blend-out is floored to the standard crossfade constants ``T_ALPHA``
        (body) / ``T_BETA`` (show): a clip authored with ``blend_out_s == 0``
        must NOT vanish in a single tick (which would be a multi-rad step, ~6×
        ``max_motor_velocity``). These are exactly what ``T_ALPHA``/``T_BETA``
        are for.
        """
        self.w_body_at_release = self.weight_body(t)
        self.w_show_at_release = self.weight_show(t)
        self.releasing = True
        self.t_release = t
        self.bo_body = max(self.bo_body, T_ALPHA)
        self.bo_show = max(self.bo_show, T_BETA)

    # --- phase / weights ---
    def _weight(self, t: float, bi: float, bo: float, w_at_release: float) -> float:
        clip = self.clip
        if self.releasing:
            if bo <= 0.0:
                return 0.0
            frac = (bo - (t - self.t_release)) / bo
            return _ramp(frac) * w_at_release
        te = t - self.t_start
        w_in = 1.0 if bi <= 0.0 else _ramp(te / bi)
        if clip.loop_mode == "once":
            if bo <= 0.0:
                w_out = 1.0 if te < clip.duration_s else 0.0
            else:
                w_out = _ramp((clip.duration_s - te) / bo)
            return min(w_in, w_out)
        # wrap / clamp: no natural ramp-out (only on preempt/cancel).
        return w_in

    def weight_body(self, t: float) -> float:
        return self._weight(t, self.bi_body, self.bo_body, self.w_body_at_release)

    def weight_show(self, t: float) -> float:
        return self._weight(t, self.bi_show, self.bo_show, self.w_show_at_release)

    def is_expired(self, t: float) -> bool:
        """True once the clip can no longer contribute and should be pruned."""
        if self.releasing:
            return (t - self.t_release) >= max(self.bo_body, self.bo_show)
        if self.clip.loop_mode == "once":
            te = t - self.t_start
            return te >= self.clip.duration_s + max(self.bo_body, self.bo_show)
        return False

    def _local_frame(self, t: float):
        """Return ``(i0, i1, frac)`` for linear interpolation at time ``t``."""
        clip = self.clip
        F = clip.frame_count
        cum = (t - self.t_start) * clip.fps
        if clip.loop_mode == "wrap":
            local = cum % F
            i0 = int(math.floor(local))
            frac = local - i0
            i0 %= F
            i1 = (i0 + 1) % F
        else:  # once / clamp: clamp to last frame
            local = cum
            if local < 0.0:
                local = 0.0
            i0 = int(math.floor(local))
            if i0 > F - 1:
                i0 = F - 1
            frac = local - i0
            if frac < 0.0:
                frac = 0.0
            if frac > 1.0:
                frac = 1.0
            i1 = min(i0 + 1, F - 1)
        return i0, i1, frac

    def sample_head(self, t: float) -> np.ndarray:
        i0, i1, frac = self._local_frame(t)
        j = self.clip.joints
        h0 = j[i0, HEAD_SLICE_16]
        h1 = j[i1, HEAD_SLICE_16]
        return h0 * (1.0 - frac) + h1 * frac

    def sample_legs(self, t: float) -> np.ndarray:
        """Sample the ten leg joints (LEG_NAMES order) at ``t`` (dock full-body).

        Only meaningful for ``layer_mask == "full_body"`` clips; head-masked
        clips carry the legs at their neutral hold, so the compositor never
        blends their legs (they would just reproduce the hold).
        """
        i0, i1, frac = self._local_frame(t)
        j = self.clip.joints
        l0 = j[i0, LEG_INDICES_16]
        l1 = j[i1, LEG_INDICES_16]
        return l0 * (1.0 - frac) + l1 * frac

    def sample_antennas(self, t: float):
        i0, i1, frac = self._local_frame(t)
        sh = self.clip.show
        al = sh.antenna_l[i0] * (1.0 - frac) + sh.antenna_l[i1] * frac
        ar = sh.antenna_r[i0] * (1.0 - frac) + sh.antenna_r[i1] * frac
        return float(al), float(ar)

    def sample_eyes(self, t: float) -> Optional[int]:
        eyes = self.clip.show.eyes
        if eyes.size == 0:
            return None
        i0, _, _ = self._local_frame(t)
        return int(eyes[i0])

    def collect_events(self, t: float) -> List[DiscreteEvent]:
        """Fire events crossed in ``(last_cum, cur_cum]`` exactly once (plan §6.1).

        The watermark is monotonic: if ``t`` moves backwards (``cur < prev``) we
        fire nothing and do NOT rewind the watermark (M1), so a discrete event
        can never re-fire when the caller's clock steps back or repeats a tick.
        """
        clip = self.clip
        cur = (t - self.t_start) * clip.fps
        prev = self.last_event_cum
        if cur < prev:
            # Time moved backwards: never rewind the watermark, fire nothing.
            return []
        events = clip.show.events
        if not events:
            self.last_event_cum = cur
            return []
        F = clip.frame_count
        fired = []
        for ev in events:  # events are pre-sorted by frame at load time
            f = ev.frame
            if clip.loop_mode == "wrap":
                k = int(math.ceil((prev - f) / F - 1e-9))
                if k < 0:
                    k = 0
                while True:
                    occ = f + k * F
                    if occ > cur + 1e-9:
                        break
                    if occ > prev + 1e-9:
                        fired.append((occ, ev))
                    k += 1
            else:
                occ = float(f)
                if prev + 1e-9 < occ <= cur + 1e-9:
                    fired.append((occ, ev))
        self.last_event_cum = cur
        fired.sort(key=lambda x: x[0])
        return [ev for _, ev in fired]


class Engine:
    """The three-layer animation engine (plan §6).

    Construct with an optional always-on ``background`` clip, then call
    :meth:`evaluate` exactly once per control tick with a monotonic ``t_now``.
    """

    def __init__(
        self,
        background: Optional[DuckAnimClip] = None,
        max_layers: int = 4,
        head_joint_limits: Optional[tuple] = None,
        head_envelope: Optional[HeadEnvelope] = None,
        leg_envelope: Optional[LegDockEnvelope] = None,
        torso_envelope: Optional[TorsoEnvelope] = None,
    ) -> None:
        self.background = background
        self.max_layers = max_layers
        self._layers: List[_ClipState] = []
        # Persistent background playback state (M2/M4): created once so its event
        # watermark survives across ticks (a throwaway per-tick state would reset
        # it and never fire background events) and to avoid a per-tick allocation.
        self._bg_state: Optional[_ClipState] = None
        # Head joint-limit table for DOCK direct joint targets (H2). These are
        # PHYSICAL head joint limits and are deliberately distinct from the
        # training COMMAND ranges (plan §6.3): DOCK head targets are absolute
        # joint angles, not relative commands, so they must be clamped to joint
        # limits, not command ranges. The plan publishes no explicit head
        # ``jnt_range`` numbers, so we default to the (tighter, safe) training
        # ranges as a conservative stand-in; pass ``head_joint_limits=(low,high)``
        # to supply the real MJCF ``jnt_range``.
        if head_joint_limits is None:
            low = np.array(TRAINING_LOW, dtype=np.float64)
            high = np.array(TRAINING_HIGH, dtype=np.float64)
        else:
            low = np.asarray(head_joint_limits[0], dtype=np.float64)
            high = np.asarray(head_joint_limits[1], dtype=np.float64)
            if low.shape != (4,) or high.shape != (4,):
                raise ValueError("head_joint_limits must be two length-4 arrays")
            if np.any(high < low):
                raise ValueError("head_joint_limits: every high must be >= its low")
        self._head_low = low
        self._head_high = high
        # Preallocated hot-path buffers (plan §6.7). ``_head_buf`` is the working
        # head accumulator (composited in place each tick); ``_scratch`` holds a
        # single weighted-sample temporary. The per-tick output array returned to
        # the caller is a fresh copy, so mutating these buffers is safe.
        self._head_buf = np.zeros(4, dtype=np.float64)
        self._scratch = np.zeros(4, dtype=np.float64)
        # D13/R16 safety envelope on the additive head offset (plan §6.5).
        # SAFETY-CRITICAL DEFAULT (reviewer E3): the envelope is OPT-OUT, not
        # opt-in. A bare ``Engine()`` — and even ``Engine(head_envelope=None)`` —
        # enforces ``DEFAULT_ENVELOPE`` so that no default/forgotten caller can
        # drive the ADDITIVE head path to the toppling extremes measured in S0.1.
        # To disable enforcement you must pass the explicit, greppable sentinel
        # ``HeadEnvelope.unbounded()`` (used by pure compositor-math tests and
        # authoring tools that never touch hardware). Enforcement is stateful
        # (slew needs the previous ENFORCED offset and the real elapsed dt), so
        # the Engine — which already owns the per-tick clock and state — is its
        # natural home.
        self.head_envelope = head_envelope if head_envelope is not None else DEFAULT_ENVELOPE
        self._prev_head_offsets: Optional[np.ndarray] = None
        self._last_t: Optional[float] = None
        # Latched fault flag (reviewer E1). Set True whenever a non-finite head
        # command (NaN/Inf from a bad joystick/telemetry sample or upstream math)
        # had to be substituted by the envelope's last line of defence. It never
        # clears on its own — the caller inspects ``head_fault`` and decides how
        # to recover (e.g. hold, e-stop) and calls ``reset()`` deliberately.
        self._head_fault: bool = False
        # --- DOCK full-body leg path (plan §6.2 dock capability) --------------
        # Legs are animated ONLY in DOCK_DEMO. Like the head envelope, the leg
        # envelope is OPT-OUT and defaults to the DERATED dock envelope so a bare
        # ``Engine()`` cannot drive the legs past the conservative first-hardware
        # deflections. Pass an explicit ``LegDockEnvelope`` (e.g. full-rate) to
        # override. The engine pre-clamps and pre-rate-limits the leg targets it
        # emits; the runtime re-applies MJCF jnt_range + the 5.24 rad/s limit on
        # the final bus as defence-in-depth.
        self.leg_envelope = leg_envelope if leg_envelope is not None else DERATED_LEG_ENVELOPE
        self._leg_rate = JointRateLimiter(MAX_MOTOR_VELOCITY)
        self._prev_leg_targets: Optional[np.ndarray] = None
        self._last_leg_t: Optional[float] = None
        # --- STAND full-body posture path (torso height/orientation command) ---
        # Sustained postural emotion (sad sag, proud puff, alert tall) is a torso
        # COMMAND on the standing policy, not an animated joint track. Like the
        # head and leg paths the torso envelope is OPT-OUT and enforced by
        # default; it defaults to the ENFORCING (not derated) envelope to match
        # the head, but note (torso_envelope.py) these limits are UNSWEPT
        # kinematic placeholders and MUST be swept against the trained standing
        # checkpoint before hardware. Pass ``TorsoEnvelope.unbounded()`` for pure
        # compositor-math tests. Enforced only in STAND (below).
        self.torso_envelope = (
            torso_envelope if torso_envelope is not None else DEFAULT_TORSO_ENVELOPE
        )
        self._prev_posture: Optional[np.ndarray] = None
        self._last_posture_t: Optional[float] = None
        # Observability (plan §6.5): count of full-body triggers refused because
        # the mode was not DOCK, and a latched flag set whenever an in-flight
        # full-body clip had to be controlled-aborted by a mode transition. Both
        # let a caller/telemetry notice that the dock-only guarantee bit.
        self._dropped_fullbody_triggers: int = 0
        self._fullbody_mode_aborts: int = 0

    @property
    def head_fault(self) -> bool:
        """True once a non-finite head command was substituted (reviewer E1).

        Latches until :meth:`reset` is called. See plan §6.5. A caller running
        real hardware should treat a rising edge as a telemetry/authoring fault
        and take a deliberate recovery action rather than trusting head output.
        """
        return self._head_fault

    @property
    def dropped_fullbody_triggers(self) -> int:
        """Count of full-body triggers refused for running outside DOCK (§6.2).

        Rises whenever a ``layer_mask == "full_body"`` clip is triggered while
        the mode is not DOCK. The engine refuses to start it (the dangerous case
        is unplayable, not merely discouraged); a caller can watch this to notice
        a mis-scheduled full-body clip.
        """
        return self._dropped_fullbody_triggers

    @property
    def fullbody_mode_aborts(self) -> int:
        """Count of in-flight full-body clips controlled-aborted by a mode change.

        Rises whenever the mode leaves DOCK while a full-body clip is playing:
        the clip is released through the normal crossfade (never a snap) and the
        animated leg channels stop being emitted (legs return to the policy).
        """
        return self._fullbody_mode_aborts

    def reset(self) -> None:
        """Clear stateful head-path state and the latched fault (reviewer E1).

        Provides an explicit recovery path so a single poisoned sample cannot
        latch head output for the Engine's lifetime. Drops the slew reference
        (``_prev_head_offsets``) and the clock so the next tick re-seeds cleanly
        from the true command, and clears :attr:`head_fault`.
        """
        self._prev_head_offsets = None
        self._last_t = None
        self._head_fault = False
        self._prev_leg_targets = None
        self._last_leg_t = None
        self._prev_posture = None
        self._last_posture_t = None

    # --- trigger handling -----------------------------------------------------
    def _owner(self) -> Optional[_ClipState]:
        for st in reversed(self._layers):
            if not st.releasing:
                return st
        return None

    def _start_clip(self, clip: DuckAnimClip, t: float) -> None:
        bi_body, bo_body = clamp_blend_times(
            clip.blend_in_s, clip.blend_out_s, clip.duration_s, clip.fps
        )
        bi_show, bo_show = clamp_blend_times(
            clip.show_blend_in_s, clip.show_blend_out_s, clip.duration_s, clip.fps
        )
        # Preempt the current owner: it becomes a releasing layer starting from
        # its current weight, so the new clip blends in from the current output.
        for st in self._layers:
            if not st.releasing:
                st.begin_release(t)
        self._layers.append(
            _ClipState(
                clip=clip, t_start=t,
                bi_body=bi_body, bo_body=bo_body,
                bi_show=bi_show, bo_show=bo_show,
            )
        )
        # Cap the number of simultaneous layers (drop the oldest releasing).
        while len(self._layers) > self.max_layers:
            self._layers.pop(0)

    def _apply_triggers(self, triggers: Triggers, t: float, mode: str) -> None:
        if triggers.cancel:
            for st in self._layers:
                if not st.releasing:
                    st.begin_release(t)
        for clip in triggers.clips:
            # RUNTIME capability gate (plan §6.2). A full-body clip animates the
            # legs and is ONLY playable in DOCK. Refuse to even start it in any
            # other mode — belt-and-braces with the compile-time validator, so a
            # clip mis-scheduled at runtime (wrong mode) cannot move the legs of a
            # standing/walking robot. Count the refusal for observability.
            if clip.layer_mask == "full_body" and mode != MODE_DOCK:
                self._dropped_fullbody_triggers += 1
                continue
            owner = self._owner()
            if owner is None or clip.priority >= owner.clip.priority:
                # higher wins; equal → newer wins (plan §6.4).
                self._start_clip(clip, t)
            # lower priority: ignored (does not take ownership).

        # Mode-transition-mid-clip (plan §6.2/§6.5). If the mode has left DOCK
        # while a full-body clip is still playing, degrade SAFELY: release it
        # through the normal crossfade (never a snap). The animated leg channels
        # simply stop being emitted below (leg_targets is None outside DOCK, so
        # the legs return to the policy with no engine-side step), and the head/
        # show part fades out via the standard blend-out.
        if mode != MODE_DOCK:
            for st in self._layers:
                if st.clip.layer_mask == "full_body" and not st.releasing:
                    st.begin_release(t)
                    self._fullbody_mode_aborts += 1

    # --- background sampling --------------------------------------------------
    def _sample_background(self, t: float):
        """Sample the always-on background loop into ``_head_buf`` (M2/M4).

        Uses a single persistent ``_ClipState`` so its event watermark survives
        across ticks. Returns ``(head_buf, antenna_l, antenna_r, eyes)`` where
        ``head_buf`` is the engine's working accumulator (the caller composites
        the triggered layers into it in place).
        """
        if self.background is None:
            self._head_buf[:] = 0.0
            return self._head_buf, 0.0, 0.0, 0
        if self._bg_state is None:
            self._bg_state = _ClipState(
                clip=self.background, t_start=t,
                bi_body=0.0, bo_body=0.0, bi_show=0.0, bo_show=0.0,
            )
        st = self._bg_state
        self._head_buf[:] = st.sample_head(t)
        al, ar = st.sample_antennas(t)
        eyes = st.sample_eyes(t)
        return self._head_buf, al, ar, (0 if eyes is None else eyes)

    # --- main evaluation ------------------------------------------------------
    def evaluate(self, t_now: float, mode: str, triggers: Optional[Triggers] = None) -> EngineOutput:
        """Evaluate the engine for one control tick (plan §6.4).

        Pure O(lookup): one elapsed-time→frame computation and a small number of
        linear blends over ≤16 floats. No solving.
        """
        if mode not in _VALID_MODES:
            raise ValueError("mode must be one of %r, got %r" % (_VALID_MODES, mode))
        if triggers is None:
            triggers = Triggers()

        self._apply_triggers(triggers, t_now, mode)

        # Layer 1: background (composited in place into the persistent buffer).
        comp_head, comp_al, comp_ar, comp_eyes = self._sample_background(t_now)

        # Background discrete events fire too (M2): idle loops may carry cues.
        fired_events: List[DiscreteEvent] = []
        if self._bg_state is not None:
            fired_events.extend(self._bg_state.collect_events(t_now))

        # Layer 2: triggered clips composed in order (releasing first → active).
        # DOCK-only leg accumulator: starts at the dock hold and blends in each
        # full-body layer's legs weighted by its BODY weight (so leg motion
        # crossfades exactly like the head, and a releasing full-body clip eases
        # its legs back to the hold instead of snapping). Only built in DOCK;
        # outside DOCK the legs are policy-owned and never emitted.
        comp_legs = None
        if mode == MODE_DOCK:
            comp_legs = _DOCK_LEG_HOLD.copy()
        # STAND posture accumulator: torso command offsets, blended by BODY weight
        # exactly like the head, so a mood's sag eases in over T_alpha and eases
        # back to neutral when a neutral-posture clip preempts it. Starts from the
        # background clip's (usually neutral) posture. Cheap (sin of 3 scalars per
        # layer); only clamped + emitted in STAND (below).
        if self.background is not None:
            comp_posture = posture_to_command_offsets(self.background.posture)
        else:
            comp_posture = np.zeros(3, dtype=np.float64)
        owner = self._owner()
        for st in self._layers:
            wb = st.weight_body(t_now)
            ws = st.weight_show(t_now)
            if wb > 0.0:
                ch = st.sample_head(t_now)
                # comp = comp*(1-wb) + ch*wb, in place (M4: no new head array).
                comp_head *= (1.0 - wb)
                np.multiply(ch, wb, out=self._scratch)
                comp_head += self._scratch
                # Posture blends with the same body weight. Every clip carries a
                # posture (neutral for head-only clips), so this correctly eases a
                # non-neutral mood in AND eases it back out under a neutral clip.
                cp = posture_to_command_offsets(st.clip.posture)
                comp_posture *= (1.0 - wb)
                comp_posture += cp * wb
                if comp_legs is not None and st.clip.layer_mask == "full_body":
                    cl = st.sample_legs(t_now)
                    comp_legs *= (1.0 - wb)
                    comp_legs += cl * wb
            if ws > 0.0:
                cal, car = st.sample_antennas(t_now)
                comp_al = comp_al * (1.0 - ws) + cal * ws
                comp_ar = comp_ar * (1.0 - ws) + car * ws
                eyes = st.sample_eyes(t_now)
                if eyes is not None and ws >= 0.5:
                    comp_eyes = eyes
            # Only the active owner fires events; releasing clips are cancelled.
            if st is owner:
                fired_events.extend(st.collect_events(t_now))

        # Layer 3: joystick additive head offset (plan §6.3), added in place.
        # Finiteness guard (reviewer E1): a NaN/Inf joystick or telemetry sample
        # summed here would poison ``comp_head`` and, once stored as the slew
        # reference, latch head output for the Engine's lifetime. Substitute a
        # non-finite offset with zero (drop just this contribution) and latch the
        # fault so the caller can react; the envelope below is a second line of
        # defence for non-finite values arriving via the authored path.
        if triggers.joystick_offset is not None:
            joy = np.asarray(triggers.joystick_offset, dtype=np.float64)
            if not np.all(np.isfinite(joy)):
                self._head_fault = True
                joy = np.where(np.isfinite(joy), joy, 0.0)
            comp_head += joy

        # ``head_command_offsets`` is a RELATIVE delta (authored head pose −
        # nominal, plus joystick). Per plan §6.3 the clamp to the training ranges
        # applies to the ABSOLUTE command after ``base_command + delta``, which
        # the caller composes downstream (see ``transform.pose_to_command``). We
        # must therefore return the UNCLAMPED delta here (H2). Subtracting the
        # (zero) nominal also yields a fresh array the caller owns.
        head_offsets = comp_head - NOMINAL_HEAD_POSE

        # D13/R16 safety envelope on the additive offset (plan §6.5). Applied
        # only in the BALANCING modes (stand/walk), where head_command_offsets is
        # the balance-critical additive command. In DOCK the legs are docked (not
        # balancing) and the head is driven via absolute ``head_targets``, so the
        # balance envelope does not apply and — critically (reviewer E7) — the
        # slew reference is FROZEN so a DOCK→STAND transition slews from the true
        # last emitted STAND command, not a fictitious offset advanced during
        # DOCK. ``HeadEnvelope.unbounded()`` is a pure passthrough (bypass) used
        # to disable enforcement deliberately.
        if mode != MODE_DOCK:
            # Elapsed dt for the slew guard. Fall back to CTRL_DT on the first
            # tick / after a backwards clock, and clamp the UPPER bound (reviewer
            # E6) so a stalled loop or debugger pause cannot authorise an
            # unbounded jump by inflating ``slew_limit*dt``.
            if self._last_t is None or t_now <= self._last_t:
                dt = CTRL_DT
            else:
                dt = min(t_now - self._last_t, MAX_SLEW_DT)
            head_offsets, fault = self.head_envelope.clamp(
                head_offsets,
                prev_command_head=self._prev_head_offsets,
                dt=dt,
                return_fault=True,
            )
            if fault:
                self._head_fault = True
            # Refuse to store a non-finite slew reference (reviewer E1): clamp
            # already sanitises, but guard defensively so a bug upstream cannot
            # latch. If somehow non-finite, drop the reference (re-seed next tick)
            # rather than poison every future tick via ``prev + clip(c-prev)``.
            if np.all(np.isfinite(head_offsets)):
                self._prev_head_offsets = head_offsets.copy()
            else:
                self._prev_head_offsets = None
                self._head_fault = True
            self._last_t = t_now

        # STAND full-body posture command (torso height/orientation). Enforced
        # ONLY in STAND — the torso-command standing policy is the only policy
        # that consumes a torso command. In WALK the deployed passthrough policy
        # has no torso command; in DOCK the body is animated via ``leg_targets``.
        # Outside STAND the slew reference is FROZEN (like the head's in DOCK) so a
        # WALK/DOCK→STAND transition slews from the true last emitted posture, not
        # one advanced while the command was inert.
        posture_command_offsets = None
        if mode == MODE_STAND:
            if self._last_posture_t is None or t_now <= self._last_posture_t:
                dt_p = CTRL_DT
            else:
                dt_p = min(t_now - self._last_posture_t, MAX_SLEW_DT)
            posture_command_offsets, p_fault = self.torso_envelope.clamp(
                comp_posture,
                prev_command_torso=self._prev_posture,
                dt=dt_p,
                return_fault=True,
            )
            if p_fault:
                self._head_fault = True
            if np.all(np.isfinite(posture_command_offsets)):
                self._prev_posture = posture_command_offsets.copy()
            else:
                self._prev_posture = None
                self._head_fault = True
            self._last_posture_t = t_now
        else:
            # Drop the reference so a later STAND re-entry re-seeds from neutral.
            self._prev_posture = None
            self._last_posture_t = None

        show = TickShow(
            antenna_l=float(np.clip(comp_al, -1.0, 1.0)),
            antenna_r=float(np.clip(comp_ar, -1.0, 1.0)),
            eyes=int(comp_eyes),
            events=fired_events,
        )

        leg_targets = None
        head_targets = None
        if mode == MODE_DOCK:
            # DOCK_DEMO: head driven as DIRECT absolute joint targets, clamped to
            # PHYSICAL head joint limits (H2). Legs held at the dock hold unless a
            # full-body clip is animating them (comp_legs), in which case the
            # animated targets are clamped to the conservative leg dock envelope
            # (jnt_range ∩ small deflection box: self-collision / cable-strain
            # guard, plan §6.2) and then RATE-LIMITED at max_motor_velocity so the
            # emitted leg targets are already spec-compliant. The runtime re-clamps
            # + re-rate-limits the final bus as defence-in-depth.
            head_targets = np.clip(comp_head, self._head_low, self._head_high)
            leg_raw = comp_legs if comp_legs is not None else _DOCK_LEG_HOLD
            leg_clamped = self.leg_envelope.clamp(leg_raw)
            # Leg velocity clamp. Seed the reference from the dock hold on the
            # first DOCK tick / after any non-DOCK excursion (re-entering DOCK
            # re-seeds from the hold, never from a stale target), and clamp the
            # upper bound on dt like the head slew so a stalled loop cannot
            # authorise an unbounded leg step.
            if self._prev_leg_targets is None or self._last_leg_t is None or t_now <= self._last_leg_t:
                leg_targets = leg_clamped.copy()
            else:
                dt_leg = min(t_now - self._last_leg_t, MAX_SLEW_DT)
                leg_targets = self._leg_rate.limit(self._prev_leg_targets, leg_clamped, dt_leg)
            self._prev_leg_targets = leg_targets.copy()
            self._last_leg_t = t_now
        else:
            # Outside DOCK the legs are policy-owned; drop the leg reference so a
            # later DOCK re-entry re-seeds from the hold (no cross-mode leg step).
            self._prev_leg_targets = None
            self._last_leg_t = None

        # Prune expired layers in place (M4: no per-tick list rebuild).
        i = len(self._layers) - 1
        while i >= 0:
            if self._layers[i].is_expired(t_now):
                del self._layers[i]
            i -= 1

        return EngineOutput(
            head_command_offsets=head_offsets,
            show=show,
            leg_targets=leg_targets,
            head_targets=head_targets,
            posture_command_offsets=posture_command_offsets,
        )
