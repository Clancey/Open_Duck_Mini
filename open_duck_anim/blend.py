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
from .joint_order import HW_ORDER_14, HEAD_SLICE_16, INIT_POS_14
from .transform import TRAINING_LOW, TRAINING_HIGH, NOMINAL_HEAD_POSE

# Timing constants (plan §6.6).
CTRL_DT = 0.02       # 50 Hz single clock
T_ALPHA = 0.35       # body blend
T_BETA = 0.10        # show blend

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

    def _apply_triggers(self, triggers: Triggers, t: float) -> None:
        if triggers.cancel:
            for st in self._layers:
                if not st.releasing:
                    st.begin_release(t)
        for clip in triggers.clips:
            owner = self._owner()
            if owner is None or clip.priority >= owner.clip.priority:
                # higher wins; equal → newer wins (plan §6.4).
                self._start_clip(clip, t)
            # lower priority: ignored (does not take ownership).

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

        self._apply_triggers(triggers, t_now)

        # Layer 1: background (composited in place into the persistent buffer).
        comp_head, comp_al, comp_ar, comp_eyes = self._sample_background(t_now)

        # Background discrete events fire too (M2): idle loops may carry cues.
        fired_events: List[DiscreteEvent] = []
        if self._bg_state is not None:
            fired_events.extend(self._bg_state.collect_events(t_now))

        # Layer 2: triggered clips composed in order (releasing first → active).
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
        if triggers.joystick_offset is not None:
            comp_head += np.asarray(triggers.joystick_offset, dtype=np.float64)

        # ``head_command_offsets`` is a RELATIVE delta (authored head pose −
        # nominal, plus joystick). Per plan §6.3 the clamp to the training ranges
        # applies to the ABSOLUTE command after ``base_command + delta``, which
        # the caller composes downstream (see ``transform.pose_to_command``). We
        # must therefore return the UNCLAMPED delta here (H2). Subtracting the
        # (zero) nominal also yields a fresh array the caller owns.
        head_offsets = comp_head - NOMINAL_HEAD_POSE

        show = TickShow(
            antenna_l=float(np.clip(comp_al, -1.0, 1.0)),
            antenna_r=float(np.clip(comp_ar, -1.0, 1.0)),
            eyes=int(comp_eyes),
            events=fired_events,
        )

        leg_targets = None
        head_targets = None
        if mode == MODE_DOCK:
            # DOCK_DEMO: legs held (load-relieving dock posture); head driven as
            # DIRECT absolute joint targets. These are joint angles, so they are
            # clamped to the PHYSICAL head joint limits (H2) — NOT the training
            # command ranges. jnt_range clamping for the full 14-DOF bus targets
            # is applied downstream by open_duck_anim.limits.
            leg_targets = _DOCK_LEG_HOLD.copy()
            head_targets = np.clip(comp_head, self._head_low, self._head_high)

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
        )
