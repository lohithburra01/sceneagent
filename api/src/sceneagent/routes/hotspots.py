"""Hotspot read endpoint — joins notes + scene_objects."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

from ..db import pool

router = APIRouter(tags=["hotspots"])


@router.get("/scenes/{slug}/hotspots")
async def list_hotspots(
    slug: str,
    category: Optional[str] = Query(default=None),
) -> list[dict[str, Any]]:
    """Return all hotspots for a scene, joined with note + object metadata."""
    p = pool()
    async with p.acquire() as con:
        scene = await con.fetchrow("SELECT id FROM scenes WHERE slug = $1", slug)
        if scene is None:
            raise HTTPException(status_code=404, detail=f"scene {slug!r} not found")
        scene_id = scene["id"]

        clauses = ["n.scene_id = $1"]
        args: list[Any] = [scene_id]
        if category is not None:
            args.append(category)
            clauses.append(f"n.category = ${len(args)}")
        where = " AND ".join(clauses)

        rows = await con.fetch(
            f"""
            SELECT
              h.id              AS hotspot_id,
              h.note_id         AS note_id,
              h.object_id       AS object_id,
              h.match_confidence AS match_confidence,
              h.position        AS position,
              h.auto_accepted   AS auto_accepted,
              n.text            AS note_text,
              n.video_timestamp AS video_timestamp,
              n.category        AS category,
              n.category_confidence AS category_confidence,
              o.instance_id     AS instance_id,
              o.class_name      AS class_name,
              o.room_label      AS room_label,
              o.centroid        AS centroid,
              o.bbox_min        AS bbox_min,
              o.bbox_max        AS bbox_max
            FROM hotspots h
            JOIN notes n ON n.id = h.note_id
            LEFT JOIN scene_objects o ON o.id = h.object_id
            WHERE {where}
            ORDER BY n.video_timestamp
            """,
            *args,
        )

    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "id": str(r["hotspot_id"]),
                "note_id": str(r["note_id"]) if r["note_id"] else None,
                "object_id": str(r["object_id"]) if r["object_id"] else None,
                "match_confidence": float(r["match_confidence"]),
                "position": list(r["position"]) if r["position"] is not None else None,
                "auto_accepted": bool(r["auto_accepted"]),
                "note_text": r["note_text"],
                "video_timestamp": float(r["video_timestamp"])
                if r["video_timestamp"] is not None
                else None,
                "category": r["category"],
                "category_confidence": float(r["category_confidence"])
                if r["category_confidence"] is not None
                else None,
                "instance_id": r["instance_id"],
                "class_name": r["class_name"],
                "room_label": r["room_label"],
                "centroid": list(r["centroid"]) if r["centroid"] is not None else None,
                "bbox_min": list(r["bbox_min"]) if r["bbox_min"] is not None else None,
                "bbox_max": list(r["bbox_max"]) if r["bbox_max"] is not None else None,
            }
        )
    return out
