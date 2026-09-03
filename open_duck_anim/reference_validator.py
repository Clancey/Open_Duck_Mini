"""Kinematic-consistency validator for the 59-float reference-motion format.

Motivation
==========
Every reference clip in this project has historically been checked *numerically*
(tilt bounds, joint limits, velocity limits, head-follow correlation) but the
derived velocity/contact fields were never checked against the pose trajectory
they claim to describe. That gap let a training reference ship with

* its angular-velocity **x/z axes transposed** relative to the quaternion
  trajectory,
* a **linear-velocity channel that is identically zero**, and
* leg joints that barely move (a "full-body" motion that is really a head wiggle),

none of which any numeric bound could see — and it cost four GPU training runs.

This module recomputes every *derived* field straight from the *pose* fields and
fails when they disagree. It is deliberately numpy-only and Python 3.9-compatible
so it lives in the reusable ``open_duck_anim`` library and runs in CI and in the
authoring path.

Frame layout (59 floats, the ``EpisodicReferenceMotion`` order)
===============================================================
================  =========  ==========================================
slice             indices    field
================  =========  ==========================================
root_pos           0:3       world root position (m)
root_quat          3:7       root orientation, **scipy ``as_quat()`` order
                             (XYZW, scalar-last)**
joints_pos         7:23      16 joint angles (rad), ``JOINT_ORDER_16``
left_toe_pos      23:26      left toe position, body frame (m)
right_toe_pos     26:29      right toe position, body frame (m)
world_lin_vel     29:32      world linear velocity (m/s)  == d(root_pos)/dt
world_ang_vel     32:35      world angular velocity (rad/s), see convention
joints_vel        35:51      16 joint velocities (rad/s) == d(joints_pos)/dt
left_toe_vel      51:54      left toe velocity (m/s)
right_toe_vel     54:57      right toe velocity (m/s)
foot_contacts     57:59      [left, right] contact flags in {0, 1}
================  =========  ==========================================

Angular-velocity convention (stated once, explicitly)
=====================================================
``root_quat`` is XYZW (scipy ``Rotation.as_quat()``). The **stale** ``qw, qx,
qy, qz`` comments in some generators are wrong — do not trust them. The world
angular velocity between consecutive frames ``q0 -> q1`` is the rotation vector
of the *world-frame* relative rotation, divided by ``dt``::

    q_rel = q1 (x) conj(q0)           # Hamilton product, XYZW
    omega = rotvec(q_rel) / dt        # matches scipy:
                                      #   (R.from_quat(q1) * R.from_quat(q0).inv()).as_rotvec() / dt

The quaternion helpers below reproduce scipy to ~1e-16 and are the single source
of truth for what "the angular velocity implied by the quaternion trajectory"
means in this validator.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .joint_order import JOINT_ORDER_16

FRAME_SIZE_59 = 59

# --- named slices into a 59-float frame --------------------------------------
ROOT_POS = slice(0, 3)
ROOT_QUAT = slice(3, 7)
JOINTS_POS = slice(7, 23)
LEFT_TOE_POS = slice(23, 26)
RIGHT_TOE_POS = slice(26, 29)
WORLD_LIN_VEL = slice(29, 32)
WORLD_ANG_VEL = slice(32, 35)
JOINTS_VEL = slice(35, 51)
LEFT_TOE_VEL = slice(51, 54)
RIGHT_TOE_VEL = slice(54, 57)
FOOT_CONTACTS = slice(57, 59)

# --- default tolerances / thresholds -----------------------------------------
# Velocities are finite differences, so the disagreement we care about is a
# *systematic* one (wrong axis, zeroed channel, wrong dt), not float noise.
# These absolute tolerances are generous relative to that.
DEFAULT_LIN_VEL_ATOL = 1e-3       # m/s
DEFAULT_ANG_VEL_ATOL = 5e-3       # rad/s
DEFAULT_JOINTS_VEL_ATOL = 5e-3    # rad/s
# A "full-body" motion whose leg joints move less than this (peak-to-peak, rad)
# is almost certainly authored for the wrong base (e.g. a docked clip reused as a
# standing reference). ~0.5 deg.
DEFAULT_LEG_MOTION_MIN_PTP = 0.02


class ReferenceValidationError(ValueError):
    """Raised by :func:`validate_reference` / :func:`validate_reference_file`
    when ``raise_on_error`` is set and at least one ERROR-severity issue is
    found."""


class Issue:
    """A single validator finding.

    ``severity`` is ``"error"`` or ``"warning"``. ``field`` names the frame
    field involved (e.g. ``"world_ang_vel"``). ``message`` is human-readable.
    """

    __slots__ = ("severity", "field", "message")

    def __init__(self, severity: str, field: str, message: str) -> None:
        self.severity = severity
        self.field = field
        self.message = message

    @property
    def is_error(self) -> bool:
        return self.severity == "error"

    def __repr__(self) -> str:
        return "Issue(%s, %s, %r)" % (self.severity, self.field, self.message)

    def __str__(self) -> str:
        return "[%s] %s: %s" % (self.severity.upper(), self.field, self.message)


class ValidationResult:
    """Aggregate result of validating one reference clip."""

    def __init__(self, fps: float, n_frames: int) -> None:
        self.fps = fps
        self.n_frames = n_frames
        self.issues: List[Issue] = []

    # -- construction ---------------------------------------------------------
    def add_error(self, field: str, message: str) -> None:
        self.issues.append(Issue("error", field, message))

    def add_warning(self, field: str, message: str) -> None:
        self.issues.append(Issue("warning", field, message))

    # -- queries --------------------------------------------------------------
    @property
    def errors(self) -> List[Issue]:
        return [i for i in self.issues if i.is_error]

    @property
    def warnings(self) -> List[Issue]:
        return [i for i in self.issues if not i.is_error]

    @property
    def ok(self) -> bool:
        """True iff there are no ERROR-severity issues (warnings are allowed)."""
        return not self.errors

    def summary(self) -> str:
        head = "reference: %d frames @ %.3g fps — %d error(s), %d warning(s)" % (
            self.n_frames,
            self.fps,
            len(self.errors),
            len(self.warnings),
        )
        lines = [head] + ["  " + str(i) for i in self.issues]
        return "\n".join(lines)

    def raise_if_error(self) -> None:
        if not self.ok:
            raise ReferenceValidationError(self.summary())


# --------------------------------------------------------------------------- #
# Quaternion helpers (numpy-only; XYZW; match scipy to ~1e-16).
# --------------------------------------------------------------------------- #
def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two XYZW quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def quat_conj(a: np.ndarray) -> np.ndarray:
    """Conjugate (== inverse for a unit quaternion), XYZW."""
    return np.array([-a[0], -a[1], -a[2], a[3]])


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle, rad) of a single XYZW quaternion.

    Chooses the shortest rotation (``w >= 0``) so the vector is continuous for
    the small inter-frame rotations we differentiate.
    """
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n == 0.0:
        return np.zeros(3)
    q = q / n
    if q[3] < 0.0:
        q = -q
    v = q[:3]
    w = q[3]
    s = np.linalg.norm(v)
    if s < 1e-12:
        # near-identity: rotvec ~= 2 * vector part
        return 2.0 * v
    angle = 2.0 * np.arctan2(s, w)
    return v / s * angle


def angular_velocity_from_quats(quats: np.ndarray, dt: float) -> np.ndarray:
    """World-frame angular velocity (N, 3) implied by an (N, 4) XYZW trajectory.

    ``omega[i]`` is the rate over the interval ``(i-1 -> i)`` and is assigned to
    frame ``i`` (backward difference); ``omega[0]`` is copied from ``omega[1]``
    so the array aligns 1:1 with the stored ``world_ang_vel`` field, which the
    upstream generator likewise fills by backward difference (the first frame is
    a duplicate/seed).
    """
    quats = np.asarray(quats, dtype=np.float64)
    n = quats.shape[0]
    out = np.zeros((n, 3), dtype=np.float64)
    for i in range(1, n):
        rel = quat_mul(quats[i], quat_conj(quats[i - 1]))
        out[i] = quat_to_rotvec(rel) / dt
    if n >= 2:
        out[0] = out[1]
    return out


# --------------------------------------------------------------------------- #
# Finite differences matching the generator (backward difference, seed frame 0).
# --------------------------------------------------------------------------- #
def _backward_diff(x: np.ndarray, dt: float) -> np.ndarray:
    """Backward finite difference along axis 0, with frame 0 seeded from frame 1.

    The reference generator computes ``vel[i] = (x[i] - x[i-1]) / dt`` and emits
    the first velocity only once ``prev`` is initialised, so frame 0's stored
    velocity equals frame 1's. We mirror that so a *correct* clip validates with
    zero residual rather than an artificial frame-0 discrepancy.
    """
    d = np.zeros_like(x)
    if x.shape[0] >= 2:
        d[1:] = (x[1:] - x[:-1]) / dt
        d[0] = d[1]
    return d


def _axis_swap_hint(stored: np.ndarray, derived: np.ndarray) -> Optional[str]:
    """If ``stored`` looks like ``derived`` with two axes swapped, name them.

    Returns a message like ``"x<->z axes appear transposed"`` or ``None``. Uses
    a simple correlation/peak comparison across the three axes, which is exactly
    the signature of the standing_wiggle bug (a quaternion written in the wrong
    component order so its implied ang-vel lands on a different axis than the
    stored value).
    """
    labels = ("x", "y", "z")
    # Only meaningful if there is real signal to compare.
    if np.linalg.norm(stored) < 1e-9 or np.linalg.norm(derived) < 1e-9:
        return None

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        a = a - a.mean()
        b = b - b.mean()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float(np.abs(a @ b) / (na * nb))

    for i, j in ((0, 1), (0, 2), (1, 2)):
        # stored axis i tracks derived axis j AND stored axis j tracks derived
        # axis i, while the diagonal (i->i) does not.
        cross = min(corr(stored[:, i], derived[:, j]), corr(stored[:, j], derived[:, i]))
        diag = max(corr(stored[:, i], derived[:, i]), corr(stored[:, j], derived[:, j]))
        if cross > 0.9 and cross > diag + 0.3:
            return "%s<->%s axes appear transposed" % (labels[i], labels[j])
    return None


def validate_reference(
    frames: Sequence[Sequence[float]],
    fps: float,
    *,
    motion_type: str = "unknown",
    lin_vel_atol: float = DEFAULT_LIN_VEL_ATOL,
    ang_vel_atol: float = DEFAULT_ANG_VEL_ATOL,
    joints_vel_atol: float = DEFAULT_JOINTS_VEL_ATOL,
    leg_motion_min_ptp: float = DEFAULT_LEG_MOTION_MIN_PTP,
    raise_on_error: bool = False,
) -> ValidationResult:
    """Validate a 59-float reference clip for kinematic self-consistency.

    Args:
        frames: ``(N, 59)`` array-like of per-frame floats.
        fps: frames per second (used for every finite difference).
        motion_type: free-form label. If it contains ``"full"``/``"body"``/
            ``"walk"``/``"wiggle"`` the leg-motion degeneracy check is applied.
        *_atol: absolute tolerances for each recomputed field.
        leg_motion_min_ptp: minimum peak-to-peak leg-joint motion (rad) expected
            of a full-body motion before a degeneracy warning is raised.
        raise_on_error: if True, raise :class:`ReferenceValidationError` when any
            ERROR-severity issue is found.

    Returns a :class:`ValidationResult`.
    """
    arr = np.asarray(frames, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != FRAME_SIZE_59:
        raise ValueError(
            "frames must be (N, %d); got shape %r" % (FRAME_SIZE_59, arr.shape)
        )
    if fps <= 0:
        raise ValueError("fps must be positive, got %r" % (fps,))

    n = arr.shape[0]
    dt = 1.0 / float(fps)
    result = ValidationResult(float(fps), n)

    if n < 2:
        result.add_warning(
            "frames", "only %d frame(s); cannot check finite-difference fields" % n
        )
        if raise_on_error:
            result.raise_if_error()
        return result

    root_pos = arr[:, ROOT_POS]
    root_quat = arr[:, ROOT_QUAT]
    joints_pos = arr[:, JOINTS_POS]
    stored_lin_vel = arr[:, WORLD_LIN_VEL]
    stored_ang_vel = arr[:, WORLD_ANG_VEL]
    stored_joints_vel = arr[:, JOINTS_VEL]
    contacts = arr[:, FOOT_CONTACTS]

    # -- 0. quaternion sanity (unit norm) -------------------------------------
    qnorm = np.linalg.norm(root_quat, axis=1)
    if np.any(np.abs(qnorm - 1.0) > 1e-3):
        worst = int(np.argmax(np.abs(qnorm - 1.0)))
        result.add_error(
            "root_quat",
            "root_quat is not unit-norm (frame %d norm=%.4f); expected XYZW "
            "unit quaternions" % (worst, qnorm[worst]),
        )

    # -- 1. linear velocity == d(root_pos)/dt ---------------------------------
    derived_lin_vel = _backward_diff(root_pos, dt)
    lin_res = np.abs(stored_lin_vel - derived_lin_vel)
    lin_max = float(lin_res.max())
    root_moves = float(np.ptp(root_pos, axis=0).max()) > 1e-6
    lin_all_zero = float(np.abs(stored_lin_vel).max()) == 0.0

    if lin_max > lin_vel_atol:
        wf = int(np.unravel_index(np.argmax(lin_res), lin_res.shape)[0])
        result.add_error(
            "world_lin_vel",
            "world_lin_vel (29:32) disagrees with d(root_pos)/dt: max residual "
            "%.4g m/s at frame %d (stored=%s, derived=%s). It must be the finite "
            "difference of root_pos at %.0f fps."
            % (
                lin_max,
                wf,
                np.array2string(stored_lin_vel[wf], precision=4),
                np.array2string(derived_lin_vel[wf], precision=4),
                fps,
            ),
        )
    # Degenerate/suspicious: a motion whose linear-velocity channel is *exactly*
    # zero on every frame. If the root also moves this is an outright error;
    # otherwise it is the tell-tale "zeroed linear velocity" signature.
    if lin_all_zero:
        if root_moves:
            result.add_error(
                "world_lin_vel",
                "world_lin_vel is identically zero on all %d frames while "
                "root_pos moves (ptp=%s) — the linear-velocity field was never "
                "filled from the trajectory."
                % (n, np.array2string(np.ptp(root_pos, axis=0), precision=4)),
            )
        else:
            result.add_warning(
                "world_lin_vel",
                "world_lin_vel is identically zero on all %d frames (root_pos is "
                "static). Consistent, but confirm the reference is meant to be a "
                "pinned-root motion." % n,
            )

    # -- 2. angular velocity == omega implied by root_quat --------------------
    derived_ang_vel = angular_velocity_from_quats(root_quat, dt)
    ang_res = np.abs(stored_ang_vel - derived_ang_vel)
    ang_max = float(ang_res.max())
    if ang_max > ang_vel_atol:
        wf = int(np.unravel_index(np.argmax(ang_res), ang_res.shape)[0])
        msg = (
            "world_ang_vel (32:35) disagrees with the angular velocity implied "
            "by the root_quat (3:7) trajectory: max residual %.4g rad/s at frame "
            "%d (stored=%s, derived=%s), convention = rotvec(q_i (x) conj(q_{i-1}))"
            "/dt with q in scipy XYZW order."
            % (
                ang_max,
                wf,
                np.array2string(stored_ang_vel[wf], precision=4),
                np.array2string(derived_ang_vel[wf], precision=4),
            )
        )
        hint = _axis_swap_hint(stored_ang_vel, derived_ang_vel)
        if hint is not None:
            msg += (
                " The stored and derived series match under an axis swap: %s. "
                "This is the classic symptom of a root_quat written in the wrong "
                "component order (e.g. WXYZ) while the format is XYZW." % hint
            )
        result.add_error("world_ang_vel", msg)

    # -- 3. joint velocity == d(joints_pos)/dt --------------------------------
    derived_joints_vel = _backward_diff(joints_pos, dt)
    jv_res = np.abs(stored_joints_vel - derived_joints_vel)
    jv_max = float(jv_res.max())
    if jv_max > joints_vel_atol:
        fidx, jidx = (int(x) for x in np.unravel_index(np.argmax(jv_res), jv_res.shape))
        jname = JOINT_ORDER_16[jidx]
        result.add_error(
            "joints_vel",
            "joints_vel (35:51) disagrees with d(joints_pos)/dt: max residual "
            "%.4g rad/s at frame %d joint %d (%s). It must be the finite "
            "difference of joints_pos at %.0f fps." % (jv_max, fidx, jidx, jname, fps),
        )

    # -- 4. foot-contact plausibility -----------------------------------------
    if not np.all(np.isin(contacts, (0.0, 1.0))):
        result.add_error(
            "foot_contacts",
            "foot_contacts (57:59) must be 0/1 flags; found other values.",
        )
    else:
        both_up = np.all(contacts == 0.0, axis=1)
        n_airborne = int(both_up.sum())
        mt = motion_type.lower()
        grounded_motion = any(
            k in mt for k in ("stand", "wiggle", "idle", "dock", "nod", "shake")
        )
        # A grounded/standing motion should never have both feet off the ground.
        if grounded_motion and n_airborne > 0:
            result.add_error(
                "foot_contacts",
                "%d/%d frames have BOTH feet off the ground, implausible for a "
                "'%s' (grounded) motion." % (n_airborne, n, motion_type),
            )
        # A locomotion clip that never lifts a foot is equally implausible.
        if "walk" in mt:
            ever_swing = np.any(contacts == 0.0)
            if not ever_swing:
                result.add_warning(
                    "foot_contacts",
                    "a 'walk' motion but no frame ever lifts a foot "
                    "(foot_contacts are always [1,1]).",
                )

    # -- 5. degenerate reference: knees/ankles barely move for a full-body motion
    # The load-bearing flexion joints (knees + ankles) are the tell: a standing
    # "full-body" wiggle whose knees move ~0.004 rad is really a head/hip jiggle
    # inherited from a docked clip, not a genuine full-body motion. Hips can sway
    # a bit more without the lower body actually doing anything, so we key off the
    # knee/ankle excursion specifically (matching the standing_wiggle defect).
    mt = motion_type.lower()
    is_fullbody = any(k in mt for k in ("full", "body", "walk", "wiggle"))
    KNEE_ANKLE_16 = (3, 4, 14, 15)
    ka_ptp = np.ptp(joints_pos[:, list(KNEE_ANKLE_16)], axis=0)
    max_ka_ptp = float(ka_ptp.max())
    if is_fullbody and max_ka_ptp < leg_motion_min_ptp:
        names = ", ".join(
            "%s=%.4f" % (JOINT_ORDER_16[j], p) for j, p in zip(KNEE_ANKLE_16, ka_ptp)
        )
        result.add_warning(
            "joints_pos",
            "declared full-body motion '%s' but the knees/ankles barely move "
            "(max %.4f rad ptp; %s). A near-static lower body usually means the "
            "clip was authored for a docked/pinned robot and is degenerate as a "
            "standing/full-body reference." % (motion_type, max_ka_ptp, names),
        )

    if raise_on_error:
        result.raise_if_error()
    return result


# --------------------------------------------------------------------------- #
# File-level helpers.
# --------------------------------------------------------------------------- #
def _frames_and_fps_from_dict(data: Dict) -> Tuple[List[List[float]], float]:
    if "Frames" not in data:
        raise ValueError("reference JSON has no 'Frames' key")
    frames = data["Frames"]
    fps = data.get("FPS", data.get("fps"))
    if fps is None:
        raise ValueError("reference JSON has no 'FPS'/'fps' key")
    return frames, float(fps)


def validate_reference_file(
    path: str,
    *,
    motion_type: Optional[str] = None,
    raise_on_error: bool = False,
    **kwargs,
) -> ValidationResult:
    """Load a 59-float reference JSON file and validate it.

    ``motion_type`` defaults to the file's stem (so ``standing_wiggle.json`` is
    treated as a grounded full-body motion for the plausibility/degeneracy
    checks).
    """
    with open(path, "r") as fh:
        data = json.load(fh)
    frames, fps = _frames_and_fps_from_dict(data)
    if motion_type is None:
        import os

        motion_type = os.path.splitext(os.path.basename(path))[0]
    return validate_reference(
        frames,
        fps,
        motion_type=motion_type,
        raise_on_error=raise_on_error,
        **kwargs,
    )
