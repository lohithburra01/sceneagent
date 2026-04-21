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
    """Load Gaussian centers.

    Preferred path: read pipeline/output/decoded_splat.npz produced by
    pipeline.src.decode_splat (handles the PlayCanvas compressed-packed format).
    Falls back to a plain .ply (x,y,z) read if the npz doesn't exist.
    """
    npz_path = Path("pipeline/output/decoded_splat.npz")
    if npz_path.exists():
        data = np.load(npz_path)
        return data["centers"].astype(np.float32)
    from plyfile import PlyData
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]
    props = [p.name for p in v.properties]
    if {"x", "y", "z"}.issubset(set(props)):
        return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    raise RuntimeError(
        f"{ply_path} uses the compressed format; run `python -m pipeline.src.decode_splat` first"
    )


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


# Classes we never want to turn into instances. They're structural (always
# present) or noise from CLIP's residential bias on our sparse renders.
STRUCTURAL_CLASSES = {"wall", "floor", "ceiling", "suspended_ceiling"}
NOISE_CLASSES = {"shower", "bathtub", "toilet", "washing_machine", "dryer",
                 "stove", "refrigerator", "oven", "microwave", "sink", "bed",
                 "pillow", "ottoman", "television", "bookshelf", "nightstand",
                 "fireplace", "mirror", "rug", "curtain", "shelf", "cabinet"}

# Per-mask vote weight uses mask_area normalised against the median, so a big
# ceiling mask doesn't drown out small wine-bottle masks. Weight = min(1, median_area / mask_area)
# clipped so tiny masks don't get unbounded influence.
MIN_CLIP_CONFIDENCE = 0.24   # filter CLIP predictions below this (very weak signal)


def _mask_path_for(i: int) -> Path:
    """Match the new renderer's 2-digit stem, fall back to the old 3-digit."""
    a = MASKS_DIR / f"view_{i:02d}.json"
    if a.exists():
        return a
    return MASKS_DIR / f"view_{i:03d}.json"


def _load_vocab() -> list[str]:
    """Prefer the new open INTERIOR_VOCAB; fall back to the legacy HOME_CLASSES."""
    try:
        from pipeline.src.vocab_interior import INTERIOR_VOCAB
        return INTERIOR_VOCAB
    except Exception:
        from pipeline.src.segment import HOME_CLASSES as _HC
        return list(_HC)


def vote_class_per_gaussian(centers: np.ndarray):
    intrinsics = json.loads((VIEWS_DIR / "_intrinsics.json").read_text())
    W, H = intrinsics["width"], intrinsics["height"]
    FOV_V = intrinsics["fov_vertical_deg"]

    HOME_CLASSES = _load_vocab()
    K = len(HOME_CLASSES)
    N = centers.shape[0]
    class_scores = np.zeros((N, K), dtype=np.float32)

    # First pass: collect all mask areas to compute a median for normalisation.
    all_areas = []
    for i, pose in enumerate(intrinsics["poses"]):
        mp = _mask_path_for(i)
        if not mp.exists():
            continue
        for m in json.loads(mp.read_text()):
            shape = m["mask_rle"]["shape"]
            area = sum(run[1] for run in m["mask_rle"]["runs"])
            if area > 0:
                all_areas.append(area)
    if not all_areas:
        return np.zeros(N, dtype=np.int32), np.zeros(N), np.zeros(N)
    median_area = float(np.median(all_areas))
    print(f"median mask area: {median_area:.0f} px (over {len(all_areas)} masks)")

    for i, pose in enumerate(intrinsics["poses"]):
        mask_path = _mask_path_for(i)
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
            cls = m["class_name"]
            if cls in STRUCTURAL_CLASSES or cls in NOISE_CLASSES:
                continue
            try:
                cls_idx = HOME_CLASSES.index(cls)
            except ValueError:
                continue
            conf = float(m["class_confidence"])
            if conf < MIN_CLIP_CONFIDENCE:
                continue
            # Mask-area-normalised vote weight: small masks are worth as much
            # as median; big masks are attenuated (dividing their per-pixel
            # vote so total mask contribution stays ~= median_area * conf).
            area = sum(run[1] for run in m["mask_rle"]["runs"])
            if area <= 0:
                continue
            weight = conf * median_area / area

            mask_2d = rle_to_mask(m["mask_rle"])
            ui = proj[valid, 0].astype(int)
            vi = proj[valid, 1].astype(int)
            hits = mask_2d[vi, ui]
            valid_idx = np.where(valid)[0][hits]
            class_scores[valid_idx, cls_idx] += weight

    best_class = class_scores.argmax(axis=1)
    totals = class_scores.sum(axis=1)
    best_conf = np.divide(class_scores.max(axis=1), totals,
                          out=np.zeros_like(totals), where=totals > 0)
    return best_class, best_conf, totals


def cluster_instances(centers, class_idx, has_votes):
    """DBSCAN per class, tighter params matched to the typical GT object size
    (~0.3m). Skip structural + noise classes."""
    HOME_CLASSES = _load_vocab()
    instance_ids = np.full(len(centers), -1, dtype=np.int32)
    next_iid = 0
    for cls in np.unique(class_idx):
        cls_name = HOME_CLASSES[int(cls)]
        if cls_name in STRUCTURAL_CLASSES or cls_name in NOISE_CLASSES:
            continue
        mask = (class_idx == cls) & has_votes
        if mask.sum() < 20:
            continue
        pts = centers[mask]
        labels = DBSCAN(eps=0.15, min_samples=25).fit_predict(pts)
        global_labels = labels.copy()
        for u in np.unique(labels[labels >= 0]):
            global_labels[labels == u] = next_iid
            next_iid += 1
        instance_ids[mask] = global_labels
    return instance_ids


def instances_to_inventory(centers, class_idx, instance_ids):
    HOME_CLASSES = _load_vocab()
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
