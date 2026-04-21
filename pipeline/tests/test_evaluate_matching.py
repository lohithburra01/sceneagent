"""Hungarian bipartite matching vs greedy matching.

The greedy implementation in evaluate.py iterates predictions in list
order and picks the best remaining GT for each. When a wrong-class
prediction overlaps two GTs, it can claim the one the right-class
prediction needs, stranding both. Hungarian matching avoids that.
"""
import numpy as np

from pipeline.src.evaluate import _eval_at


def _cube(cx, cy, cz, r=0.5):
    return [cx - r, cy - r, cz - r], [cx + r, cy + r, cz + r]


def test_hungarian_avoids_greedy_theft():
    """Two GTs, two predictions, all same class.

    Pred_0 straddles both GTs but best-matches B (IoU 0.57).
    Pred_1 only overlaps B (IoU 0.9).
    Greedy takes Pred_0 first, claims B, leaves Pred_1 stranded → TP=1.
    Hungarian finds the globally optimal Pred_0→A, Pred_1→B → TP=2.
    """
    a_min, a_max = _cube(0.0, 0, 0, r=0.5)
    b_min, b_max = _cube(1.2, 0, 0, r=0.5)
    gt = [
        {"instance_id": "A", "class_name": "chair",
         "bbox_min": a_min, "bbox_max": a_max},
        {"instance_id": "B", "class_name": "chair",
         "bbox_min": b_min, "bbox_max": b_max},
    ]
    ours = [
        {"class_name": "chair", "bbox_min": [0.3, -0.5, -0.5],
         "bbox_max": [1.5, 0.5, 0.5]},
        {"class_name": "chair", "bbox_min": [0.8, -0.5, -0.5],
         "bbox_max": [1.7, 0.5, 0.5]},
    ]
    r = _eval_at(ours, gt, iou_thresh=0.1, require_class_match=False)
    assert r["tp"] == 2, r


def test_structural_classes_are_dropped_from_eval():
    # One wall GT, one wall prediction that matches perfectly. After
    # filtering, both sides are empty -> zero TP/FP/FN, not +1 TP.
    wall_min, wall_max = [-5, -5, 0], [5, 5, 3]
    gt = [{"instance_id": "W", "class_name": "wall",
           "bbox_min": wall_min, "bbox_max": wall_max}]
    ours = [{"class_name": "wall",
             "bbox_min": wall_min, "bbox_max": wall_max}]
    r = _eval_at(ours, gt, iou_thresh=0.25, require_class_match=True)
    assert r["tp"] == 0 and r["fp"] == 0 and r["fn"] == 0, r
