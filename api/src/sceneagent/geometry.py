"""Camera pose + frustum visibility helpers.

Scene convention (matches the pipeline):
  - Right-handed world, Z-up.
  - A camera "pose" is a dict: {"position": [x, y, z], "yaw_deg": float}.
    Yaw is around the world Z axis in degrees; 0 deg points along +X.
  - FOV is specified as vertical field-of-view in degrees.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

CameraPose = dict[str, Any]


def pose_from_yaw(position: list[float], yaw_deg: float) -> CameraPose:
    """Build a camera pose dict from a position + yaw.

    Parameters
    ----------
    position : 3-vector
    yaw_deg  : rotation around +Z, degrees. 0 deg = looking along +X.

    Returns a dict with keys: position, yaw_deg, forward, right, up.
    ``forward`` is a unit vector in world coordinates.
    """
    yaw_rad = math.radians(yaw_deg)
    forward = [math.cos(yaw_rad), math.sin(yaw_rad), 0.0]
    # With up=+Z and forward in XY plane, right = forward x up (right-handed).
    right = [forward[1], -forward[0], 0.0]
    up = [0.0, 0.0, 1.0]
    return {
        "position": list(position),
        "yaw_deg": float(yaw_deg),
        "forward": forward,
        "right": right,
        "up": up,
    }


def _as_pose(pose: CameraPose) -> CameraPose:
    """Normalize the pose dict — accept {position, yaw_deg} or full form."""
    if "forward" in pose:
        return pose
    return pose_from_yaw(pose["position"], float(pose.get("yaw_deg", 0.0)))


def is_point_visible_from_pose(
    point: list[float] | tuple[float, float, float] | np.ndarray,
    pose: CameraPose,
    fov_deg: float = 60.0,
    max_distance: float = 15.0,
    aspect_ratio: float = 16.0 / 9.0,
) -> bool:
    """Frustum-visibility test.

    Accepts an abbreviated pose ``{"position": [...], "yaw_deg": ...}`` or a
    full pose (with ``forward``).  ``fov_deg`` is the vertical FOV.
    """
    pose = _as_pose(pose)
    p = np.asarray(point, dtype=float).reshape(3)
    cam = np.asarray(pose["position"], dtype=float).reshape(3)
    fwd = np.asarray(pose["forward"], dtype=float).reshape(3)
    fwd = fwd / (np.linalg.norm(fwd) + 1e-9)

    delta = p - cam
    dist = float(np.linalg.norm(delta))
    if dist <= 1e-6:
        return True
    if dist > max_distance:
        return False

    direction = delta / dist
    cos_forward = float(np.dot(direction, fwd))
    if cos_forward <= 0.0:
        return False  # behind the camera

    # Half-angles: vertical is fov_deg/2, horizontal expands by aspect.
    half_v = math.radians(fov_deg) / 2.0
    half_h = math.atan(math.tan(half_v) * aspect_ratio)
    # A point is in the frustum if its angular offset from forward, along both
    # axes, is below the corresponding half-angle.
    up = np.asarray(pose["up"], dtype=float)
    right = np.asarray(pose["right"], dtype=float)
    # Project direction onto right/up axes; 'cos_forward' serves as the
    # denominator for horizontal/vertical angles.
    dx = float(np.dot(direction, right))
    dy = float(np.dot(direction, up))
    ang_h = math.atan2(abs(dx), cos_forward)
    ang_v = math.atan2(abs(dy), cos_forward)
    return ang_h <= half_h and ang_v <= half_v


def nearest_trajectory_pose(
    trajectory: list[dict[str, Any]], timestamp: float
) -> CameraPose:
    """Return the camera pose whose ``timestamp`` is closest to the given time."""
    if not trajectory:
        # Default pose at origin looking along +X.
        return pose_from_yaw([0.0, 0.0, 1.7], 0.0)
    best = min(
        trajectory,
        key=lambda e: abs(float(e.get("timestamp", 0.0)) - float(timestamp)),
    )
    pos = best.get("position", [0.0, 0.0, 1.7])
    yaw = float(best.get("yaw_deg", 0.0))
    return pose_from_yaw(list(pos), yaw)
