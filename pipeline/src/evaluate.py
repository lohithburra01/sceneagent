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


# Synonym groups — both sides are normalized to the canonical form before
# matching. Lets CLIP/SAM's "wine_bottle" match InteriorGS's "wine", etc.
_CANONICAL: dict[str, str] = {}
for group in [
    ("wine", "wine_bottle", "bottle"),
    ("downlights", "downlight"),
    ("spotlight", "spot_light"),
    ("decorative_painting", "painting"),
    ("high_chair", "stool", "armchair"),
    ("dining_plate", "plate"),
    ("dining_table", "table"),
    ("chandelier", "pendant_light", "decorative_pendant"),
    ("ceiling", "suspended_ceiling"),
    ("sofa", "multi_person_sofa", "couch"),
    ("cup", "mug"),
    ("carpet", "rug"),
]:
    canonical = group[0]
    for alias in group:
        _CANONICAL[alias] = canonical


def normalize_class(s: str) -> str:
    base = s.lower().replace("-", "_").replace(" ", "_")
    return _CANONICAL.get(base, base)


def _eval_at(ours, gt, iou_thresh: float, require_class_match: bool):
    tp = fp = fn = 0
    class_tp = Counter(); class_fn = Counter(); class_fp = Counter()
    matched_gt = set()
    for o in ours:
        oc = normalize_class(o["class_name"])
        best_iou = 0.0; best_gt = None
        for g in gt:
            if g["instance_id"] in matched_gt:
                continue
            iv = bbox_iou(o["bbox_min"], o["bbox_max"], g["bbox_min"], g["bbox_max"])
            if iv > best_iou:
                best_iou, best_gt = iv, g
        if best_gt and best_iou >= iou_thresh:
            cls_ok = (normalize_class(best_gt["class_name"]) == oc)
            if cls_ok or not require_class_match:
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
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    per_class = {}
    for c in set(class_tp) | set(class_fn) | set(class_fp):
        pc = class_tp[c] / (class_tp[c] + class_fp[c]) if (class_tp[c] + class_fp[c]) else 0
        rc = class_tp[c] / (class_tp[c] + class_fn[c]) if (class_tp[c] + class_fn[c]) else 0
        per_class[c] = {"precision": pc, "recall": rc, "tp": class_tp[c], "fp": class_fp[c], "fn": class_fn[c]}
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1, "per_class": per_class}


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

    # Evaluate across several IoU thresholds AND at class-agnostic vs class-matched.
    variants: dict[str, dict] = {}
    for iou_t in (0.1, 0.25, 0.5):
        variants[f"class_matched@{iou_t}"] = _eval_at(ours, gt, iou_t, require_class_match=True)
        variants[f"class_agnostic@{iou_t}"] = _eval_at(ours, gt, iou_t, require_class_match=False)

    headline = variants["class_matched@0.25"]
    agn = variants["class_agnostic@0.25"]
    result = {
        "num_predicted": len(ours),
        "num_ground_truth": len(gt),
        "tp": headline["tp"], "fp": headline["fp"], "fn": headline["fn"],
        "precision": headline["precision"], "recall": headline["recall"], "f1": headline["f1"],
        "iou_threshold": 0.25,
        "per_class": headline["per_class"],
        "class_agnostic_at_0p25": {
            "tp": agn["tp"], "fp": agn["fp"], "fn": agn["fn"],
            "precision": agn["precision"], "recall": agn["recall"], "f1": agn["f1"],
        },
        "variants": {k: {"tp": v["tp"], "fp": v["fp"], "fn": v["fn"],
                         "precision": v["precision"], "recall": v["recall"], "f1": v["f1"]}
                     for k, v in variants.items()},
        "note": (
            "InteriorGS 0003_839989 (wine bar). Pipeline: 30 views rendered from a "
            "custom PlayCanvas-compressed PLY decoder + software splat rasterizer, "
            "MobileSAM masks, OpenCLIP ViT-B/32 zero-shot over a 54-class home+"
            "commercial vocabulary, multi-view voting with mask-area normalisation, "
            "DBSCAN per class. Headline numbers are class_matched@IoU>=0.25; "
            "class_agnostic@* isolates localisation from classification quality."
        ) if len(ours) else "No predictions; renderer did not produce views.",
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    print(f"Predicted: {result['num_predicted']}  GT: {result['num_ground_truth']}")
    print(f"class_matched@0.25    P={headline['precision']:.3f} R={headline['recall']:.3f} F1={headline['f1']:.3f}  tp={headline['tp']} fp={headline['fp']} fn={headline['fn']}")
    print(f"class_agnostic@0.25   P={agn['precision']:.3f} R={agn['recall']:.3f} F1={agn['f1']:.3f}  tp={agn['tp']} fp={agn['fp']} fn={agn['fn']}")
    for k in ("class_matched@0.1", "class_agnostic@0.1", "class_matched@0.5", "class_agnostic@0.5"):
        v = variants[k]
        print(f"{k:22s}  P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f}  tp={v['tp']} fp={v['fp']} fn={v['fn']}")


if __name__ == "__main__":
    main()
