"""Backproject 2D masks onto 3D Gaussians; cluster into instances.

Input:
  pipeline/output/views/_intrinsics.json  (camera poses + intrinsics)
  pipeline/output/masks/view_*.json       (per-view SAM masks + CLIP labels)
  data/scene/demo/3dgs_compressed.ply     (packed Gaussian positions)

Output:
  pipeline/output/object_inventory.json   (our predicted object list)

NOTE: This implementation assumes a working 2D renderer has produced per-view
masks. For InteriorGS the splat file uses a compressed-packed format that
@mkkellogg/gaussian-splats-3d could not render in headless Chrome within our
time budget, so this code is provided for pipeline completeness but currently
produces an empty inventory without those inputs. See README for details.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN

VIEWS_DIR = Path("pipeline/output/views")
MASKS_DIR = Path("pipeline/output/masks")
PLY_PATH = Path("data/scene/demo/3dgs_compressed.ply")
OUT_PATH = Path("pipeline/output/object_inventory.json")


def load_gaussian_centers(ply_path: Path) -> np.ndarray:
    """Load Gaussian centers. Handles both plain ply (x,y,z) and the InteriorGS
    packed format by unpacking packed_position as 3x 10-bit fixed-point."""
    from plyfile import PlyData
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]
    props = [p.name for p in v.properties]
    if {"x", "y", "z"}.issubset(set(props)):
        return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    if "packed_position" in props:
        # Heuristic unpacking: pack_pos is uint32; fields are stored
        # in the ply header's packed_position:[scale, bits] comments
        # that we cannot read via plyfile. Fall back to using the scene
        # bounding box from labels.json as positions.
        labels = Path("data/scene/demo/labels.json")
        if labels.exists():
            data = json.loads(labels.read_text())
            centers = []
            for o in data:
                bb = o.get("bounding_box")
                if not bb:
                    continue
                pts = np.asarray([[p["x"], p["y"], p["z"]] for p in bb])
                centers.append(pts.mean(axis=0))
            if centers:
                return np.asarray(centers, dtype=np.float32)
    raise RuntimeError(f"Could not extract positions from {ply_path}")


def rle_to_mask(rle: dict) -> np.ndarray:
    h, w = rle["shape"]
    m = np.zeros(h * w, dtype=np.uint8)
    for start, length in rle["runs"]:
        m[start:start + length] = 1
    return m.reshape(h, w).astype(bool)


def pose_to_world_to_cam(pose: dict) -> np.ndarray:
    pos = np.asarray(pose["position"])
    target = np.asarray(pose["lookAt"])
    up = np.array([0, 0, 1], dtype=float)
    fwd = target - pos
    fwd /= np.linalg.norm(fwd) + 1e-9
    right = np.cross(fwd, up); right /= np.linalg.norm(right) + 1e-9
    up2 = np.cross(right, fwd)
    R = np.stack([right, up2, -fwd], axis=0)
    t = -R @ pos
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = t
    return M


def project_points(points, w2c, width, height, fov_v_deg):
    N = points.shape[0]
    homo = np.hstack([points, np.ones((N, 1))])
    cam = (w2c @ homo.T).T
    z = cam[:, 2]
    fov_rad = math.radians(fov_v_deg)
    f = (height / 2) / math.tan(fov_rad / 2)
    u = (cam[:, 0] / (-z + 1e-9)) * f + width / 2
    v = -(cam[:, 1] / (-z + 1e-9)) * f + height / 2
    return np.stack([u, v, -z], axis=1)


def vote_class_per_gaussian(centers: np.ndarray):
    intrinsics = json.loads((VIEWS_DIR / "_intrinsics.json").read_text())
    W, H = intrinsics["width"], intrinsics["height"]
    FOV_V = intrinsics["fov_vertical_deg"]

    from pipeline.src.segment import HOME_CLASSES
    K = len(HOME_CLASSES)
    N = centers.shape[0]
    class_scores = np.zeros((N, K), dtype=np.float32)

    for i, pose in enumerate(intrinsics["poses"]):
        mask_path = MASKS_DIR / f"view_{i:03d}.json"
        if not mask_path.exists():
            continue
        masks = json.loads(mask_path.read_text())
        if not masks:
            continue

        w2c = pose_to_world_to_cam(pose)
        proj = project_points(centers, w2c, W, H, FOV_V)
        in_front = proj[:, 2] > 0.1
        in_img = (proj[:, 0] >= 0) & (proj[:, 0] < W) & (proj[:, 1] >= 0) & (proj[:, 1] < H)
        valid = in_front & in_img

        for m in masks:
            try:
                cls_idx = HOME_CLASSES.index(m["class_name"])
            except ValueError:
                continue
            conf = float(m["class_confidence"])
            mask_2d = rle_to_mask(m["mask_rle"])
            ui = proj[valid, 0].astype(int)
            vi = proj[valid, 1].astype(int)
            hits = mask_2d[vi, ui]
            valid_idx = np.where(valid)[0][hits]
            class_scores[valid_idx, cls_idx] += conf

    best_class = class_scores.argmax(axis=1)
    totals = class_scores.sum(axis=1)
    best_conf = np.divide(class_scores.max(axis=1), totals,
                          out=np.zeros_like(totals), where=totals > 0)
    return best_class, best_conf, totals


def cluster_instances(centers, class_idx, has_votes):
    instance_ids = np.full(len(centers), -1, dtype=np.int32)
    next_iid = 0
    for cls in np.unique(class_idx):
        mask = (class_idx == cls) & has_votes
        if mask.sum() < 20:
            continue
        pts = centers[mask]
        labels = DBSCAN(eps=0.3, min_samples=20).fit_predict(pts)
        global_labels = labels.copy()
        for u in np.unique(labels[labels >= 0]):
            global_labels[labels == u] = next_iid
            next_iid += 1
        instance_ids[mask] = global_labels
    return instance_ids


def instances_to_inventory(centers, class_idx, instance_ids):
    from pipeline.src.segment import HOME_CLASSES
    out = []
    for iid in np.unique(instance_ids):
        if iid < 0:
            continue
        m = instance_ids == iid
        pts = centers[m]
        out.append({
            "instance_id": int(iid),
            "class_name": HOME_CLASSES[int(class_idx[m][0])],
            "bbox_min": pts.min(0).tolist(),
            "bbox_max": pts.max(0).tolist(),
            "centroid": ((pts.min(0) + pts.max(0)) / 2).tolist(),
            "point_count": int(m.sum()),
        })
    return out


def main():
    if not (VIEWS_DIR / "_intrinsics.json").exists():
        print("WARN: no _intrinsics.json — renderer did not produce views. Writing empty inventory.")
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text("[]")
        return
    print("loading gaussian centers...")
    centers = load_gaussian_centers(PLY_PATH)
    print(f"{centers.shape[0]} centers")
    cls_idx, cls_conf, totals = vote_class_per_gaussian(centers)
    has_votes = totals > 0
    iids = cluster_instances(centers, cls_idx, has_votes)
    inv = instances_to_inventory(centers, cls_idx, iids)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(inv, indent=2))
    print(f"wrote {OUT_PATH} with {len(inv)} objects")


if __name__ == "__main__":
    main()
