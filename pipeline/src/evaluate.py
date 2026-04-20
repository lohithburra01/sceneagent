"""Evaluate our predicted object inventory vs InteriorGS labels.json.

InteriorGS labels.json is a flat list of {ins_id, label, bounding_box:[8 xyz-corners]}.
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import Counter

import numpy as np

OURS_PATH = Path("pipeline/output/object_inventory.json")
GT_PATH = Path("data/scene/demo/labels.json")
OUT_PATH = Path("pipeline/output/metrics.json")


def bbox_from_corners(corners):
    a = np.asarray([[p["x"], p["y"], p["z"]] for p in corners])
    return a.min(0).tolist(), a.max(0).tolist()


def bbox_iou(a_min, a_max, b_min, b_max) -> float:
    a_min = np.asarray(a_min); a_max = np.asarray(a_max)
    b_min = np.asarray(b_min); b_max = np.asarray(b_max)
    inter_min = np.maximum(a_min, b_min)
    inter_max = np.minimum(a_max, b_max)
    inter = np.prod(np.maximum(inter_max - inter_min, 0))
    vol_a = np.prod(a_max - a_min)
    vol_b = np.prod(b_max - b_min)
    union = vol_a + vol_b - inter
    return float(inter / union) if union > 0 else 0.0


def normalize_class(s: str) -> str:
    return s.lower().replace("-", "_").replace(" ", "_")


def main():
    ours = json.loads(OURS_PATH.read_text()) if OURS_PATH.exists() else []
    gt_raw = json.loads(GT_PATH.read_text())
    gt = []
    for obj in gt_raw:
        bb = obj.get("bounding_box")
        if not bb:
            continue
        bmin, bmax = bbox_from_corners(bb)
        gt.append({
            "instance_id": obj.get("ins_id"),
            "class_name": normalize_class(obj.get("label", "unknown")),
            "bbox_min": bmin, "bbox_max": bmax,
        })

    tp = fn = fp = 0
    class_tp, class_fn, class_fp = Counter(), Counter(), Counter()
    matched_gt = set()

    for o in ours:
        oc = normalize_class(o["class_name"])
        best_iou = 0.0; best_gt = None
        for g in gt:
            if g["instance_id"] in matched_gt:
                continue
            iou = bbox_iou(o["bbox_min"], o["bbox_max"], g["bbox_min"], g["bbox_max"])
            if iou > best_iou:
                best_iou, best_gt = iou, g
        if best_gt and best_iou >= 0.25:
            if normalize_class(best_gt["class_name"]) == oc:
                tp += 1; class_tp[oc] += 1
                matched_gt.add(best_gt["instance_id"])
            else:
                fp += 1; class_fp[oc] += 1
        else:
            fp += 1; class_fp[oc] += 1

    for g in gt:
        if g["instance_id"] not in matched_gt:
            fn += 1
            class_fn[g["class_name"]] += 1

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    per_class = {}
    for c in set(class_tp) | set(class_fn) | set(class_fp):
        p = class_tp[c] / (class_tp[c] + class_fp[c]) if (class_tp[c] + class_fp[c]) else 0
        r = class_tp[c] / (class_tp[c] + class_fn[c]) if (class_tp[c] + class_fn[c]) else 0
        per_class[c] = {"precision": p, "recall": r, "tp": class_tp[c], "fp": class_fp[c], "fn": class_fn[c]}

    result = {
        "num_predicted": len(ours),
        "num_ground_truth": len(gt),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "iou_threshold": 0.25,
        "per_class": per_class,
        "note": (
            "InteriorGS 0003_839989 (wine bar). Our splat-render step (Puppeteer + "
            "@mkkellogg/gaussian-splats-3d headless) could not load the compressed-"
            "packed .ply format within the time budget, so no per-view masks were "
            "generated and backproject.py produced an empty inventory. Pipeline "
            "code (segment.py / backproject.py / evaluate.py) is implemented and "
            "can be run once a compatible 2D rasterizer is plugged in."
        ) if len(ours) == 0 else None,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    print(f"Predicted: {result['num_predicted']}  GT: {result['num_ground_truth']}")
    print(f"Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")
    print(f"TP: {tp}  FP: {fp}  FN: {fn}")


if __name__ == "__main__":
    main()
