"""Generate 30 camera poses distributed around the scene bbox.

Reads data/scene/demo/labels.json (list of {ins_id, label, bounding_box[8 corners]}),
computes the scene AABB, and writes pipeline/render_views/camera_poses.json with
30 poses: three rings of 10 cameras at different heights, pointing toward the
scene centroid with small random target jitter so nearby objects are covered.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

LABELS = Path("data/scene/demo/labels.json")
OUT = Path("pipeline/render_views/camera_poses.json")


def scene_bounds() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = json.loads(LABELS.read_text())
    pts = []
    for o in data:
        bb = o.get("bounding_box")
        if not bb:
            continue
        for p in bb:
            pts.append([p["x"], p["y"], p["z"]])
    arr = np.asarray(pts, dtype=np.float32)
    return arr.min(0), arr.max(0), (arr.min(0) + arr.max(0)) / 2


def ring(center: np.ndarray, radius: float, height: float, n: int, yaw0: float = 0.0):
    """n camera poses on a ring at the given height, each looking slightly below centerline."""
    out = []
    for i in range(n):
        a = yaw0 + 2 * math.pi * i / n
        pos = [float(center[0] + radius * math.cos(a)),
               float(center[1] + radius * math.sin(a)),
               float(height)]
        tgt = [float(center[0]), float(center[1]), float(center[2])]
        out.append({"position": pos, "lookAt": tgt, "yaw_deg": math.degrees(a)})
    return out


def main():
    bmin, bmax, center = scene_bounds()
    extent = bmax - bmin
    diag = float(np.linalg.norm(extent[:2]))
    r_outer = diag * 0.7
    r_mid = diag * 0.45
    r_inner = max(diag * 0.15, 0.5)

    eye = float(center[2])  # roughly half the vertical extent
    low_h = float(bmin[2] + 1.2)
    high_h = float(bmin[2] + min(extent[2] * 0.8, 2.6))

    poses = []
    # Ring 1 — outside perimeter, eye level, looking in
    poses.extend(ring(center, r_outer, low_h, 10, yaw0=0.0))
    # Ring 2 — mid radius, slightly higher
    poses.extend(ring(center, r_mid, high_h, 10, yaw0=math.pi / 10))
    # Ring 3 — inside the scene, low, targets jittered to cover interesting regions
    for i in range(10):
        a = 2 * math.pi * i / 10
        pos = [float(center[0] + r_inner * math.cos(a)),
               float(center[1] + r_inner * math.sin(a)),
               float(eye)]
        # Target: push outward toward objects
        tx = float(center[0] + extent[0] * 0.35 * math.cos(a + math.pi))
        ty = float(center[1] + extent[1] * 0.35 * math.sin(a + math.pi))
        tz = float(center[2] - 0.2)
        poses.append({"position": pos, "lookAt": [tx, ty, tz], "yaw_deg": math.degrees(a + math.pi)})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(poses, indent=2))
    print(f"wrote {len(poses)} poses to {OUT}")
    print(f"scene center {center.tolist()}, extent {extent.tolist()}")


if __name__ == "__main__":
    main()
