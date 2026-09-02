"""Tests for envelope.py — the empirical safe head envelope (plan §6.5, D13/R16).

Covers the enforcement contract and every reviewer-flagged guard:

* per-channel deflection clamping *at* and *beyond* each limit, both signs;
* slew limiting across ticks, applied LAST so the per-channel rate cap holds
  even during a large reversal, with varying and very large dt (E4, E6);
* the combined multi-axis L2 budget, using the SAME dangerous-side normaliser
  the harness measured the budget with (E2), binding even when every individual
  channel is within its own limit;
* a within-envelope signal passing through completely unchanged;
* non-finite (NaN/Inf) input in every channel: sanitised, flagged, no latch (E1);
* the hardware-derating helper deriving factor (not factor**2) end-to-end (E5);
* batched / >1-D input rejected (E12) and input arrays never aliased (E13).
"""

import numpy as np
import pytest

from open_duck_anim import envelope as env
from open_duck_anim.envelope import (
    HeadEnvelope,
    clamp_head_envelope,
    DEFAULT_ENVELOPE,
    DEFLECTION_LOW,
    DEFLECTION_HIGH,
    HEAD_CHANNELS,
    SLEW_LIMIT,
    COMBINED_L2_BUDGET,
    HARDWARE_DERATING,
)


# --- oracles ---------------------------------------------------------------
def _safe_mag_vec(low, high):
    """Independent duplicate of the harness ``_safe_mag_vec`` (E2 oracle).

    L_i = min(|low_i|, high_i): the magnitude of the TIGHTER (more dangerous)
    side of each channel. The combined budget was measured against this
    normaliser (envelope_sweep.py:_safe_mag_vec), so enforcement MUST use the
    same one or the scalar budget is applied against looser denominators than it
    was calibrated with.
    """
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    return np.minimum(np.abs(low), high)


def _combined_norm(e, c):
    """Combined norm ``||c / L||_2`` using the CORRECTED dangerous-side L (E2)."""
    c = np.asarray(c, dtype=np.float64)
    L = _safe_mag_vec(e.low, e.high)
    L = np.where(L > 1e-12, L, 1e-12)
    return float(np.sqrt(np.sum((c / L) ** 2)))


def _single_axis(j, value):
    v = np.zeros(4)
    v[j] = value
    return v


def _unconstrained():
    """Envelope with the combined budget effectively disabled (per-channel only)."""
    return HeadEnvelope(low=DEFAULT_ENVELOPE.low, high=DEFAULT_ENVELOPE.high,
                        slew_limit=SLEW_LIMIT, l2_budget=1e9)


# === E2: enforcement normaliser must agree with the harness ================
def test_enforcement_normaliser_matches_harness_oracle():
    """DEFAULT_ENVELOPE._L must equal the harness dangerous-side normaliser."""
    oracle = _safe_mag_vec(DEFAULT_ENVELOPE.low, DEFAULT_ENVELOPE.high)
    assert np.allclose(DEFAULT_ENVELOPE._L, oracle)
    # And it is the *tighter* side, not the commanded side, for the asymmetric
    # channels: neck_pitch (0.16, not 0.31) and head_yaw (0.29, not 1.50).
    assert DEFAULT_ENVELOPE._L[0] == pytest.approx(0.16)
    assert DEFAULT_ENVELOPE._L[2] == pytest.approx(0.29)


def test_enforcement_scale_matches_measured_budget_not_commanded_side():
    """A vector the OLD commanded-side rule allowed must now be scaled down.

    Reviewer E2 reproduction: [0.093, 0.234, 0.450, 0.150] has ||c/L_min||2 =
    1.71 (3.1x the 0.55 measured budget) but only ~0.6 under the commanded-side
    normaliser. The corrected enforcement must pull it back to the budget.
    """
    c = np.array([0.093, 0.234, 0.450, 0.150])
    assert _combined_norm(DEFAULT_ENVELOPE, c) > 1.5  # far over budget
    out = clamp_head_envelope(c)
    assert _combined_norm(DEFAULT_ENVELOPE, out) == pytest.approx(
        COMBINED_L2_BUDGET, abs=1e-9
    )


# --- per-channel deflection clamping (at and beyond, both signs) -----------
@pytest.mark.parametrize("j", range(4))
def test_single_axis_clamped_to_high(j):
    e = _unconstrained()
    out = e.clamp(_single_axis(j, 100.0))
    assert out[j] == pytest.approx(DEFLECTION_HIGH[j])
    assert np.all(out[np.arange(4) != j] == 0.0)


@pytest.mark.parametrize("j", range(4))
def test_single_axis_clamped_to_low(j):
    e = _unconstrained()
    out = e.clamp(_single_axis(j, -100.0))
    assert out[j] == pytest.approx(DEFLECTION_LOW[j])


@pytest.mark.parametrize("j", range(4))
def test_exactly_at_limit_passes(j):
    e = _unconstrained()
    hi = e.clamp(_single_axis(j, DEFLECTION_HIGH[j]))
    lo = e.clamp(_single_axis(j, DEFLECTION_LOW[j]))
    assert hi[j] == pytest.approx(DEFLECTION_HIGH[j])
    assert lo[j] == pytest.approx(DEFLECTION_LOW[j])


def test_beyond_limit_never_exceeds_any_channel():
    e = _unconstrained()
    out = e.clamp(np.array([9.0, 9.0, 9.0, 9.0]))
    assert np.all(out <= DEFLECTION_HIGH + 1e-12)
    out = e.clamp(np.array([-9.0, -9.0, -9.0, -9.0]))
    assert np.all(out >= DEFLECTION_LOW - 1e-12)


def test_asymmetric_dangerous_side_bounds():
    """neck_pitch and head_yaw have a much tighter negative side; clamping to
    the dangerous side must use that tighter magnitude (E11)."""
    e = _unconstrained()
    # neck_pitch: +0.31 allowed, -0.16 is the dangerous side.
    assert e.clamp(_single_axis(0, 5.0))[0] == pytest.approx(0.31)
    assert e.clamp(_single_axis(0, -5.0))[0] == pytest.approx(-0.16)
    # head_yaw: +1.50 allowed, -0.29 dangerous.
    assert e.clamp(_single_axis(2, 5.0))[2] == pytest.approx(1.50)
    assert e.clamp(_single_axis(2, -5.0))[2] == pytest.approx(-0.29)


# --- slew limiting (applied LAST — E4) -------------------------------------
def test_slew_caps_step_at_dt():
    """head_pitch 0->0.31 is within its L2 budget (0.31/0.78=0.40<0.6) so only
    the slew guard binds: the step is capped at slew_limit*dt."""
    dt = 0.02
    prev = np.zeros(4)
    out = clamp_head_envelope([0.0, 0.31, 0.0, 0.0], prev_command_head=prev, dt=dt)
    max_step = SLEW_LIMIT * dt  # 0.1048
    assert out[1] == pytest.approx(max_step)


def test_slew_rate_cap_holds_during_large_reversal():
    """E4 postcondition: the slew guard is applied LAST, so NO per-channel rate
    exceeds the cap, even for a big commanded reversal that also trips the L2
    down-scale. Reproduces the reviewer's -13.9 rad/s case and asserts it can no
    longer happen."""
    dt = 0.02
    prev = np.array([0.0, 0.0, 0.9, 0.0])  # near full yaw
    out = clamp_head_envelope([9.0, 9.0, 9.0, 9.0], prev_command_head=prev, dt=dt)
    rate = np.abs(out - prev) / dt
    assert np.all(rate <= SLEW_LIMIT + 1e-9), rate


def test_slew_converges_over_ticks():
    """Feeding back the enforced value converges to a within-budget target."""
    dt = 0.02
    target = np.array([0.0, 0.15, 0.0, 0.0])  # head_pitch, within budget
    prev = np.zeros(4)
    max_step = SLEW_LIMIT * dt
    n = 0
    while not np.allclose(prev, target, atol=1e-6) and n < 1000:
        prev = clamp_head_envelope(target, prev_command_head=prev, dt=dt)
        n += 1
    assert np.allclose(prev, target, atol=1e-6)
    assert n >= int(0.15 / max_step)


def test_slew_skipped_when_no_prev():
    """No prev → no slew guard (first tick). A within-budget command passes."""
    out = clamp_head_envelope([0.0, 0.15, 0.0, 0.0], prev_command_head=None)
    assert out[1] == pytest.approx(0.15)


def test_slew_varying_dt_multi_tick():
    """Slew across ticks with different dt each tick, each step within cap (E11)."""
    prev = np.zeros(4)
    target = np.array([0.0, 0.7, 0.0, 0.0])  # head_pitch high side
    for dt in (0.02, 0.05, 0.01, 0.1, 0.005):
        out = clamp_head_envelope(target, prev_command_head=prev, dt=dt)
        assert np.all(np.abs(out - prev) <= SLEW_LIMIT * dt + 1e-12)
        prev = out


def test_slew_bad_dt_raises():
    with pytest.raises(ValueError):
        clamp_head_envelope([0.1, 0, 0, 0], prev_command_head=np.zeros(4), dt=0.0)


# --- combined multi-axis L2 budget -----------------------------------------
def test_combined_budget_constant_is_measured_value():
    """Pin the empirically-adopted combined budget (envelope.py provenance).

    0.55 is the largest L2 budget whose adversarial ENFORCED-path validation keeps
    STAND worst-case tilt <= the 8.6 deg Phase 4 bound (measured 2026-09-01,
    experiments/animation/envelope_sweep.py --experiment validate; 0.60 hit
    8.75 deg). If this changes, re-derive and update the provenance comment.
    """
    assert COMBINED_L2_BUDGET == 0.55


def test_combined_budget_scales_down_over_budget_vector():
    c = np.array([DEFLECTION_HIGH[0], 0.0, DEFLECTION_HIGH[2], 0.0])
    out = clamp_head_envelope(c)
    assert _combined_norm(DEFAULT_ENVELOPE, out) == pytest.approx(
        COMBINED_L2_BUDGET, abs=1e-9
    )
    # direction preserved (uniform scale)
    assert out[0] / out[2] == pytest.approx(c[0] / c[2])


def test_combined_budget_binds_when_all_channels_within_limits():
    """The L2 budget must bind even when EVERY individual channel is within its
    own deflection limit (E11) — the whole point of a combined constraint."""
    # Each channel at 0.55 of its dangerous-side limit: individually legal,
    # jointly ||c/L||2 = 0.55*2 = 1.1 > the budget.
    L = _safe_mag_vec(DEFAULT_ENVELOPE.low, DEFAULT_ENVELOPE.high)
    c = 0.55 * L  # positive → within high on every channel
    assert np.all(c <= DEFLECTION_HIGH + 1e-12)  # each channel legal alone
    assert _combined_norm(DEFAULT_ENVELOPE, c) > COMBINED_L2_BUDGET
    out = clamp_head_envelope(c)
    assert _combined_norm(DEFAULT_ENVELOPE, out) == pytest.approx(
        COMBINED_L2_BUDGET, abs=1e-9
    )


def test_combined_budget_in_budget_passes_unchanged():
    c = np.array([0.02, 0.05, 0.05, 0.05])
    assert _combined_norm(DEFAULT_ENVELOPE, c) < COMBINED_L2_BUDGET
    out = clamp_head_envelope(c)
    assert np.allclose(out, c)


def test_combined_budget_uses_dangerous_side():
    """Per-channel clamp uses each channel's own signed limit; the negative neck
    side (-0.16) is the tighter, dangerous one."""
    neg = clamp_head_envelope(_single_axis(0, -1.0), envelope=_unconstrained())
    assert neg[0] == pytest.approx(DEFLECTION_LOW[0])


# --- within-envelope signal passes through unchanged -----------------------
def test_within_envelope_unchanged_with_slew():
    dt = 0.02
    prev = np.array([0.05, 0.10, 0.05, 0.05])
    cmd = prev + np.array([0.001, 0.001, 0.001, 0.001])
    assert _combined_norm(DEFAULT_ENVELOPE, cmd) < COMBINED_L2_BUDGET
    out = clamp_head_envelope(cmd, prev_command_head=prev, dt=dt)
    assert np.allclose(out, cmd)


# === E1: non-finite input sanitised, flagged, and does NOT latch ============
@pytest.mark.parametrize("j", range(4))
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_channel_sanitised_and_flagged(j, bad):
    c = np.array([0.1, 0.1, 0.1, 0.1])
    c[j] = bad
    out, fault = clamp_head_envelope(c, return_fault=True)
    assert fault is True
    assert np.all(np.isfinite(out))
    # the combined guard is NOT disabled by the NaN (it would be if NaN reached
    # np.clip / the norm): output stays within the budget.
    assert _combined_norm(DEFAULT_ENVELOPE, out) <= COMBINED_L2_BUDGET + 1e-9


def test_nonfinite_does_not_latch_across_clean_ticks():
    """One poisoned sample must not poison later clean ticks (E1). Feeding the
    enforced output back as prev, a clean command recovers immediately."""
    dt = 0.02
    prev = np.zeros(4)
    # poisoned tick
    prev, fault = clamp_head_envelope([np.nan, 0.0, 0.0, 0.0],
                                      prev_command_head=prev, dt=dt,
                                      return_fault=True)
    assert fault and np.all(np.isfinite(prev))
    # subsequent clean ticks converge to a normal within-budget command
    target = np.array([0.0, 0.1, 0.0, 0.0])
    for _ in range(50):
        prev, fault = clamp_head_envelope(target, prev_command_head=prev, dt=dt,
                                          return_fault=True)
        assert fault is False
        assert np.all(np.isfinite(prev))
    assert np.allclose(prev, target, atol=1e-6)


def test_nonfinite_prev_does_not_propagate():
    """A poisoned prev (slew reference) must not turn a clean command into NaN."""
    out, fault = clamp_head_envelope([0.0, 0.1, 0.0, 0.0],
                                     prev_command_head=[np.nan, 0.0, 0.0, 0.0],
                                     dt=0.02, return_fault=True)
    assert np.all(np.isfinite(out))


# --- derating (E5: factor, not factor**2) ----------------------------------
def test_derated_scales_limits_but_not_budget():
    """E5: derated() scales low/high but MUST leave l2_budget unchanged — the
    budget is L-normalised, so scaling L already derates the absolute constraint
    by ``factor``; scaling the budget too would apply factor**2."""
    d = DEFAULT_ENVELOPE.derated()  # default factor 0.5
    assert np.allclose(d.low, DEFAULT_ENVELOPE.low * HARDWARE_DERATING)
    assert np.allclose(d.high, DEFAULT_ENVELOPE.high * HARDWARE_DERATING)
    assert d.l2_budget == COMBINED_L2_BUDGET  # NOT scaled
    assert d.slew_limit == DEFAULT_ENVELOPE.slew_limit


def test_derated_end_to_end_is_factor_not_factor_squared():
    """clamp(big, derated(f)) ≈ f * clamp(big) for a saturating command (E5).

    If derated() double-applied the factor, the ratio would be f**2 (=0.25 for
    f=0.5) instead of f."""
    f = 0.5
    big = np.array([9.0, 9.0, 9.0, 9.0])
    full = clamp_head_envelope(big)
    der = clamp_head_envelope(big, envelope=DEFAULT_ENVELOPE.derated(f))
    assert np.allclose(der, f * full, atol=1e-9)


def test_derated_is_stricter():
    d = DEFAULT_ENVELOPE.derated(0.5)
    c = _single_axis(1, 5.0)  # head_pitch, saturating
    full = clamp_head_envelope(c, envelope=DEFAULT_ENVELOPE)
    der = clamp_head_envelope(c, envelope=d)
    assert der[1] < full[1]


def test_derated_bad_factor():
    with pytest.raises(ValueError):
        DEFAULT_ENVELOPE.derated(0.0)
    with pytest.raises(ValueError):
        DEFAULT_ENVELOPE.derated(1.5)


# === unbounded() escape hatch (E3) =========================================
def test_unbounded_is_pure_passthrough():
    u = HeadEnvelope.unbounded()
    assert u.bypass is True
    raw = np.array([5.0, -5.0, 9.0, -9.0])
    assert np.allclose(u.clamp(raw), raw)
    # even non-finite passes verbatim — it is a DELIBERATE full opt-out.
    out = u.clamp([np.nan, 0.0, 0.0, 0.0])
    assert np.isnan(out[0])


# --- construction guards ---------------------------------------------------
def test_bad_shape_raises():
    with pytest.raises(ValueError):
        HeadEnvelope(low=np.zeros(3), high=np.ones(3))


def test_limits_must_bracket_zero():
    with pytest.raises(ValueError):
        HeadEnvelope(low=np.array([0.1, -0.1, -0.1, -0.1]),
                     high=np.array([0.2, 0.1, 0.1, 0.1]))


def test_high_below_low_raises():
    with pytest.raises(ValueError):
        HeadEnvelope(low=np.array([-0.1, -0.1, -0.1, -0.1]),
                     high=np.array([-0.2, 0.1, 0.1, 0.1]))


def test_bad_slew_budget():
    with pytest.raises(ValueError):
        HeadEnvelope(slew_limit=0.0)
    with pytest.raises(ValueError):
        HeadEnvelope(l2_budget=-1.0)


# === E13: constructor copies low/high and freezes them =====================
def test_constructor_copies_and_freezes_limits():
    low = np.array([-0.16, -0.78, -0.29, -0.5])
    high = np.array([0.31, 0.78, 1.5, 0.5])
    e = HeadEnvelope(low=low, high=high)
    low[0] = -99.0
    high[2] = 99.0
    assert e.low[0] == pytest.approx(-0.16)  # unaffected by caller mutation
    assert e.high[2] == pytest.approx(1.5)
    with pytest.raises(ValueError):  # read-only
        e.low[0] = 0.0
    with pytest.raises(ValueError):
        e.high[0] = 0.0


# === E12: batched / >1-D input rejected ====================================
def test_batched_input_rejected():
    with pytest.raises(ValueError):
        clamp_head_envelope(np.zeros((2, 4)))
    with pytest.raises(ValueError):
        clamp_head_envelope(np.zeros((1, 4)))
    with pytest.raises(ValueError):
        clamp_head_envelope(np.zeros((4, 1)))


# --- input coercion --------------------------------------------------------
def test_non_numeric_rejected():
    with pytest.raises(TypeError):
        clamp_head_envelope(["a", "b", "c", "d"])


def test_wrong_length_rejected():
    with pytest.raises(ValueError):
        clamp_head_envelope([0.1, 0.2, 0.3])


def test_list_input_accepted():
    out = clamp_head_envelope([0.0, 0.0, 0.0, 0.0])
    assert isinstance(out, np.ndarray)
    assert out.shape == (4,)


def test_out_buffer_written():
    buf = np.empty(4)
    out = clamp_head_envelope([0.05, 0.05, 0.05, 0.05], out=buf)
    assert out is buf


def test_channels_constant():
    assert HEAD_CHANNELS == ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
    assert env.DEFLECTION_LOW.shape == (4,)
