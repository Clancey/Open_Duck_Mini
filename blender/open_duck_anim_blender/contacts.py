"""Foot-contact computation (defect D3) — bpy-free.

Upstream hardcodes ``"foot_contact": [1, 1]``, which makes the imitation
reward's contact term degenerate for Blender clips (D3). This module computes
**real** contacts from foot geometry, and — because contact from a kinematic rig
is not always reliable (a purely head/antenna "show" clip has no meaningful gait
contact) — also supports an explicit *invalid* marker so the training side can
zero the contact weight instead of trusting bogus values.

Two things are emitted by the recorder (see :mod:`.export`):

* per-frame ``foot_contact = [left, right]`` (0/1) written into the 59-float
  frame (bytes 57:59), computed by :func:`compute_foot_contacts`;
* a top-level authoring-JSON flag ``FootContactValid`` (bool). When the author
  marks a clip as non-stepping / contacts-unreliable, the recorder writes
  ``[0, 0]`` for every frame AND sets ``FootContactValid = false`` — an explicit,
  greppable signal, never a silent ``[1, 1]``.
"""

from __future__ import annotations

from typing import List, Tuple

# Default: a foot whose lowest point is within this many metres of the ground
# plane is "in contact". Tuned against the real ``open-duck-mini.blend`` rig
# (verified on Blender 5.2.1): across the shipped 60-frame walk the toe bones
# travel ~29 mm vertically (min/max per foot ≈ 0/29 mm above the lowest planted
# point), with a median height of ~15 mm. The previous 10 mm band assumed a
# ~1 cm kinematic step and flagged only ~19% of the cycle as contact per foot —
# implausibly low for a walking gait, whose stance fraction is ≳50%. A 15 mm
# band (≈ half the measured toe travel ≈ the median height) yields ~48% per-foot
# contact duty while still classifying the fully-lifted swing apex (~29 mm) as
# no-contact. Tune per scene via the panel; also set ``ground_z`` to the scene's
# actual ground (this rig's toes sit at world Z ≈ -0.147 m, so the default
# ``ground_z=0`` must be overridden or every frame reads as contact).
DEFAULT_CONTACT_THRESHOLD_M: float = 0.015


def compute_foot_contacts(
    left_foot_z: float,
    right_foot_z: float,
    ground_z: float = 0.0,
    threshold: float = DEFAULT_CONTACT_THRESHOLD_M,
) -> List[int]:
    """Return ``[left_contact, right_contact]`` as 0/1 from foot heights.

    A foot is in contact when its world-space height is at or below
    ``ground_z + threshold``. This replaces the hardcoded ``[1, 1]`` (D3).

    Args:
        left_foot_z / right_foot_z: world-space Z of the foot/toe (metres).
        ground_z: world Z of the ground plane (default 0).
        threshold: contact band above the ground (metres, must be >= 0).
    """
    if threshold < 0:
        raise ValueError("threshold must be >= 0")
    cutoff = ground_z + threshold
    return [int(left_foot_z <= cutoff), int(right_foot_z <= cutoff)]


def invalid_contacts() -> List[int]:
    """The explicit non-stepping / unreliable sentinel: ``[0, 0]``.

    Paired with ``FootContactValid = false`` in the authoring JSON so a
    downstream consumer can zero ``w_contact`` (plan Appendix C) rather than
    receive a misleading ``[1, 1]``.
    """
    return [0, 0]
