"""All detected objects for a scene — fed by the segmentation pipeline,
read straight from scene_objects.

Different from /hotspots: hotspots are the subset that have been matched
to a lister note. detections is every object the pipeline produced
(typically 100s), so the listing UI can show a true inventory + per-
object bbox + confidence.
"""

from __future__ import annotations

import math
from typing import Any

from fastapi import APIRouter, HTTPException

from ..db import pool

router = APIRouter(tags=["detections"])


@router.get("/scenes/{slug}/detections")
async def list_detections(slug: str) -> list[dict[str, Any]]:
    p = pool()
    async with p.acquire() as con:
        scene = await con.fetchrow("SELECT id FROM scenes WHERE slug = $1", slug)
        if scene is None:
            raise HTTPException(status_code=404, detail=f"scene {slug!r} not found")
        rows = await con.fetch(
            """
            SELECT
              id, instance_id, class_name, room_label,
              centroid, bbox_min, bbox_max, source
            FROM scene_objects
            WHERE scene_id = $1
            """,
            scene["id"],
        )

    # Confidence proxy: log-normalize the (effective) Gaussian count via the
    # bbox volume × density, but we don't have point_count in the DB. Use bbox
    # diagonal magnitude as a stand-in until the schema gets a real
    # confidence column. Larger objects = more visual evidence ≈ higher
    # confidence; clipped to [0.4, 0.99] so the UI shows a useful range.
    out: list[dict[str, Any]] = []
    for r in rows:
        bmin = list(r["bbox_min"])
        bmax = list(r["bbox_max"])
        diag = math.sqrt(sum((bmax[i] - bmin[i]) ** 2 for i in range(3)))
        # rough mapping: 0.1m → ~0.4, 1m → ~0.7, 5m+ → ~0.95
        conf = 0.4 + 0.55 * min(1.0, math.log10(max(diag, 0.05) * 10) / 1.7)
        out.append({
            "id": str(r["id"]),
            "instance_id": int(r["instance_id"]),
            "class_name": r["class_name"],
            "room_label": r["room_label"],
            "centroid": list(r["centroid"]),
            "bbox_min": bmin,
            "bbox_max": bmax,
            "source": r["source"],
            "confidence": conf,
        })
    # Stable order so UI list doesn't shuffle between calls
    out.sort(key=lambda d: (d["class_name"], d["instance_id"]))
    return out
