"""Small, dependency-free URDF reader used by the Blender add-on."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree


@dataclass(frozen=True)
class UrdfJoint:
    name: str
    parent: str
    child: str
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    axis: tuple[float, float, float]
    lower: float
    upper: float
    velocity: float


def _vector(element, attribute: str, default: str) -> tuple[float, float, float]:
    return tuple(float(value) for value in element.get(attribute, default).split())


def parse_urdf(path: str | Path) -> list[UrdfJoint]:
    """Return all revolute and continuous joints from *path* in URDF order."""
    root = ElementTree.parse(path).getroot()
    joints: list[UrdfJoint] = []
    for node in root.findall("joint"):
        if node.get("type") not in {"revolute", "continuous"}:
            continue
        parent = node.find("parent")
        child = node.find("child")
        if parent is None or child is None:
            raise ValueError(f"Joint {node.get('name')!r} has no parent or child link")
        origin = node.find("origin")
        axis = node.find("axis")
        limit = node.find("limit")
        if limit is None and node.get("type") == "revolute":
            raise ValueError(f"Revolute joint {node.get('name')!r} has no limit")
        joints.append(
            UrdfJoint(
                name=node.get("name", ""),
                parent=parent.get("link", ""),
                child=child.get("link", ""),
                origin_xyz=_vector(origin, "xyz", "0 0 0") if origin is not None else (0.0, 0.0, 0.0),
                origin_rpy=_vector(origin, "rpy", "0 0 0") if origin is not None else (0.0, 0.0, 0.0),
                axis=_vector(axis, "xyz", "1 0 0") if axis is not None else (1.0, 0.0, 0.0),
                lower=float(limit.get("lower", "-3.141592653589793")) if limit is not None else -3.141592653589793,
                upper=float(limit.get("upper", "3.141592653589793")) if limit is not None else 3.141592653589793,
                velocity=float(limit.get("velocity", "0")) if limit is not None else 0.0,
            )
        )
    return joints
