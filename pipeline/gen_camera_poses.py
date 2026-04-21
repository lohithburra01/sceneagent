"""Generate object-density-driven camera poses for a splat scene.

v2 (over the simple 6×6 grid version): instead of sampling a grid
that lands some cameras facing walls, we sample camera positions
*around* the densest clusters of non-structural GT centroids:

  - K-means cluster all non-structural GT centroids into N_VIEWS
    centers. This naturally puts more cameras where there are more
    objects.
  - For each centroid, place the camera 2.0 m away on a horizontal
    radius (yaw chosen by sweep across cluster), at floor + 1.6 m
    eye height, looking back at the centroid.
  - Reject any camera that lands outside the scene bbox (cameras
    are clamped inward).

labels.json is read only for centroids ("where the stuff is") — class
names are not used at inference. A user-supplied splat would replace
this with point-cloud DBSCAN on the Gaussians themselves.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

LABELS = Path("data/scene/demo/labels.json")
OUT = Path("pipeline/output/camera_poses.json")
N_VIEWS = 30
EYE_HEIGHT = 1.6
CAM_DISTANCE = 2.0
STRUCTURAL = {
    "wall", "floor", "ceiling", "suspended_ceiling", "suspended ceiling",
    "window", "door", "stairs", "column",
}


def _bbox(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return arr.min(0), arr.max(0)


def _centroids(gt: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (object_centroids, scene_min, scene_max)."""
    all_pts = []
    centers = []
    for o in gt:
        bb = o.get("bounding_box")
        if not bb:
            continue
        a = np.asarray([[p["x"], p["y"], p["z"]] for p in bb])
        all_pts.append(a)
        label = (o.get("label") or "").lower().replace("-", "_").replace(" ", "_")
        if label in STRUCTURAL:
            continue
        centers.append(a.mean(0))
    bmin, bmax = _bbox(np.concatenate(all_pts, axis=0))
    return np.asarray(centers), bmin, bmax


def main():
    gt = json.loads(LABELS.read_text())
    centroids, bmin, bmax = _centroids(gt)
    print(f"non-structural centroids: {len(centroids)} | scene bbox {bmin.tolist()} → {bmax.tolist()}")

    # Cluster centroids into N_VIEWS clusters (more cameras where more objects)
    km = KMeans(n_clusters=min(N_VIEWS, len(centroids)), n_init=10, random_state=0).fit(centroids)
    targets = km.cluster_centers_
    weights = np.bincount(km.labels_, minlength=len(targets))

    # Sweep yaw across views to keep diverse angles
    yaws = np.linspace(0, 2 * np.pi, len(targets), endpoint=False)

    poses = []
    eye_z = bmin[2] + EYE_HEIGHT
    for i, (t, w, yaw) in enumerate(zip(targets, weights, yaws)):
        cam_xy = t[:2] + CAM_DISTANCE * np.array([np.cos(yaw), np.sin(yaw)])
        # clamp inside bbox so we don't end up outside the room
        cam_xy = np.clip(cam_xy, bmin[:2] + 0.3, bmax[:2] - 0.3)
        pos = np.array([cam_xy[0], cam_xy[1], eye_z], dtype=np.float64)

        # if cluster centroid is below eye level (low table), look down a bit;
        # if it's at ceiling (chandelier), look up. Keep z literal.
        look_at = t.tolist()

        poses.append({
            "index": i,
            "position": pos.tolist(),
            "look_at": look_at,
            "up": [0.0, 0.0, 1.0],
            "fov_y_deg": 60.0,
            "width": 800,
            "height": 600,
            "cluster_size": int(w),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(poses, indent=2))
    print(f"wrote {len(poses)} object-density poses to {OUT}")
    print(f"cluster sizes (objects per camera): min={weights.min()} max={weights.max()} mean={weights.mean():.1f}")


if __name__ == "__main__":
    main()
