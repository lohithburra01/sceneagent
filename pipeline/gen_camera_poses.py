"""Generate interior-facing camera poses for a splat scene.

Samples 30 positions on a horizontal grid inside the scene bbox at
eye height (floor + 1.6 m), each looking at the nearest non-structural
GT object centroid. Kills the "cameras outside the room see floaters"
failure of the old perimeter-ring sampler.

labels.json is only used for scene extent + "where is interesting
stuff to look at". Class labels are not read at inference time —
a real user-uploaded splat would replace this with a point-cloud
PCA bbox.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

LABELS = Path("data/scene/demo/labels.json")
OUT = Path("pipeline/output/camera_poses.json")
N_VIEWS = 30
EYE_HEIGHT = 1.6
STRUCTURAL = {
    "wall", "floor", "ceiling", "suspended_ceiling", "suspended ceiling",
    "window", "door", "stairs", "column",
}


def _scene_extents(gt):
    all_min = np.array([np.inf] * 3)
    all_max = np.array([-np.inf] * 3)
    centers = []
    for o in gt:
        a = np.asarray([[p["x"], p["y"], p["z"]] for p in o["bounding_box"]])
        all_min = np.minimum(all_min, a.min(0))
        all_max = np.maximum(all_max, a.max(0))
        centers.append(a.mean(0))
    return all_min, all_max, np.asarray(centers)


def main():
    gt = [g for g in json.loads(LABELS.read_text()) if g.get("bounding_box")]
    bmin, bmax, centers = _scene_extents(gt)

    side = int(np.ceil(np.sqrt(N_VIEWS)))
    xs = np.linspace(bmin[0] + 0.5, bmax[0] - 0.5, side)
    ys = np.linspace(bmin[1] + 0.5, bmax[1] - 0.5, side)
    z = bmin[2] + EYE_HEIGHT
    positions = [(x, y, z) for x in xs for y in ys][:N_VIEWS]

    poses = []
    for i, p in enumerate(positions):
        pos = np.asarray(p)
        d2 = np.sum((centers - pos) ** 2, axis=1)
        order = np.argsort(d2)
        target = None
        for idx in order:
            label = (gt[idx].get("label") or "").lower().replace("-", "_").replace(" ", "_")
            if label in STRUCTURAL:
                continue
            target = centers[idx]
            break
        if target is None:
            target = centers[order[0]]
        poses.append({
            "index": i,
            "position": pos.tolist(),
            "look_at": target.tolist(),
            "up": [0.0, 0.0, 1.0],
            "fov_y_deg": 60.0,
            "width": 800,
            "height": 600,
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(poses, indent=2))
    print(f"wrote {len(poses)} poses to {OUT}")
    print(f"scene bbox min={bmin.tolist()} max={bmax.tolist()} eye_z={z:.2f}")


if __name__ == "__main__":
    main()
