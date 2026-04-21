"""Read-only endpoint surfacing the latest pipeline/output/metrics.json.

Used by the web frontend to render the F1 footer chip on the sidebar.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["metrics"])

# In dev the API runs at repo root; in Docker the metrics file is baked into
# the image at /app/pipeline/output. Try both.
CANDIDATES = [
    Path("pipeline/output/metrics.json"),
    Path("/app/pipeline/output/metrics.json"),
]


@router.get("/scenes/{slug}/metrics")
async def scene_metrics(slug: str) -> dict[str, Any]:
    # Currently we only have one scene's metrics file; slug is accepted for
    # future multi-scene support but not yet used to disambiguate.
    _ = slug
    for p in CANDIDATES:
        if p.exists():
            try:
                data = json.loads(p.read_text())
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"metrics unreadable: {exc}")
            # Return only the fields the frontend cares about. Keep it
            # honest — no massaging.
            return {
                "f1": float(data.get("f1", 0.0)),
                "precision": float(data.get("precision", 0.0)),
                "recall": float(data.get("recall", 0.0)),
                "tp": int(data.get("tp", 0)),
                "fp": int(data.get("fp", 0)),
                "fn": int(data.get("fn", 0)),
                "num_predicted": int(data.get("num_predicted", 0)),
                "num_ground_truth": int(data.get("num_ground_truth", 0)),
                "iou_threshold": float(data.get("iou_threshold", 0.25)),
            }
    raise HTTPException(status_code=404, detail="metrics.json not found")
