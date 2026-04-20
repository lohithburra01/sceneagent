"""Note creation + batch seed-matching endpoints."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..categorizer import classify_category
from ..db import pool
from ..geometry import nearest_trajectory_pose
from ..matcher import rank_objects_for_note

logger = logging.getLogger(__name__)

router = APIRouter(tags=["notes"])


class CreateNoteBody(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    video_timestamp: float = Field(..., ge=0.0)


def _parse_trajectory(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return []
    if isinstance(raw, list):
        return raw
    return []


async def _fetch_scene(con, slug: str) -> dict[str, Any]:
    row = await con.fetchrow(
        "SELECT id, slug, camera_trajectory FROM scenes WHERE slug = $1", slug
    )
    if row is None:
        raise HTTPException(status_code=404, detail=f"scene {slug!r} not found")
    return {
        "id": row["id"],
        "slug": row["slug"],
        "camera_trajectory": _parse_trajectory(row["camera_trajectory"]),
    }


async def _fetch_objects(con, scene_id: uuid.UUID) -> list[dict[str, Any]]:
    rows = await con.fetch(
        """
        SELECT id, instance_id, class_name, room_label,
               centroid, bbox_min, bbox_max, clip_embedding
        FROM scene_objects
        WHERE scene_id = $1
        """,
        scene_id,
    )
    out: list[dict[str, Any]] = []
    for r in rows:
        emb = r["clip_embedding"]
        # pgvector codec returns a numpy array; convert to list for portability.
        if hasattr(emb, "tolist"):
            emb = emb.tolist()
        out.append(
            {
                "id": r["id"],
                "instance_id": r["instance_id"],
                "class_name": r["class_name"],
                "room_label": r["room_label"],
                "centroid": list(r["centroid"]) if r["centroid"] is not None else None,
                "bbox_min": list(r["bbox_min"]) if r["bbox_min"] is not None else None,
                "bbox_max": list(r["bbox_max"]) if r["bbox_max"] is not None else None,
                "clip_embedding": emb,
            }
        )
    return out


def _hotspot_threshold() -> float:
    # v1: permissive — auto-accept everything above 0.5, per the design spec.
    return 0.5


async def _classify_and_match(
    con,
    scene: dict[str, Any],
    note_id: uuid.UUID,
    text: str,
    video_timestamp: float,
    objects: list[dict[str, Any]],
) -> dict[str, Any]:
    """Classify the note, produce a hotspot if there is a viable match.

    Returns a dict with ``category``, ``category_confidence``, ``hotspot`` (or
    None). The note row is updated in-place with the category.
    """
    category, cat_conf = classify_category(text)
    await con.execute(
        "UPDATE notes SET category = $1, category_confidence = $2 WHERE id = $3",
        category,
        cat_conf,
        note_id,
    )

    pose = nearest_trajectory_pose(scene["camera_trajectory"], video_timestamp)
    ranked = rank_objects_for_note(text, pose, objects)

    hotspot_row: Optional[dict[str, Any]] = None
    if ranked:
        best = ranked[0]
        obj = best["object"]
        similarity = float(best["similarity"])
        # Delete any stale hotspot for this note (idempotent for seed-match).
        await con.execute("DELETE FROM hotspots WHERE note_id = $1", note_id)
        row = await con.fetchrow(
            """
            INSERT INTO hotspots
              (note_id, object_id, match_confidence, position, auto_accepted)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id, note_id, object_id, match_confidence, position, auto_accepted
            """,
            note_id,
            obj["id"],
            similarity,
            obj.get("centroid") or [0.0, 0.0, 0.0],
            similarity >= _hotspot_threshold(),
        )
        hotspot_row = {
            "id": str(row["id"]),
            "note_id": str(row["note_id"]),
            "object_id": str(row["object_id"]) if row["object_id"] else None,
            "match_confidence": float(row["match_confidence"]),
            "position": list(row["position"]),
            "auto_accepted": bool(row["auto_accepted"]),
            "object_class": obj["class_name"],
        }

    return {
        "category": category,
        "category_confidence": cat_conf,
        "hotspot": hotspot_row,
    }


@router.post("/scenes/{slug}/notes")
async def create_note(slug: str, body: CreateNoteBody) -> dict[str, Any]:
    """Insert a note, classify it, and create a hotspot for the best match."""
    p = pool()
    async with p.acquire() as con:
        scene = await _fetch_scene(con, slug)
        objects = await _fetch_objects(con, scene["id"])

        row = await con.fetchrow(
            """
            INSERT INTO notes (scene_id, text, video_timestamp)
            VALUES ($1, $2, $3)
            RETURNING id, scene_id, text, video_timestamp, created_at
            """,
            scene["id"],
            body.text,
            body.video_timestamp,
        )
        note_id = row["id"]

        result = await _classify_and_match(
            con, scene, note_id, body.text, body.video_timestamp, objects
        )

    return {
        "note_id": str(note_id),
        "text": body.text,
        "video_timestamp": body.video_timestamp,
        "category": result["category"],
        "category_confidence": result["category_confidence"],
        "hotspot": result["hotspot"],
    }


@router.post("/scenes/{slug}/notes/seed-match")
async def seed_match(slug: str) -> dict[str, Any]:
    """Match every note in the scene that doesn't yet have a hotspot."""
    p = pool()
    matched = 0
    skipped = 0
    async with p.acquire() as con:
        scene = await _fetch_scene(con, slug)
        objects = await _fetch_objects(con, scene["id"])

        notes = await con.fetch(
            """
            SELECT n.id, n.text, n.video_timestamp
            FROM notes n
            LEFT JOIN hotspots h ON h.note_id = n.id
            WHERE n.scene_id = $1 AND h.id IS NULL
            ORDER BY n.created_at
            """,
            scene["id"],
        )
        for n in notes:
            res = await _classify_and_match(
                con, scene, n["id"], n["text"], float(n["video_timestamp"]), objects
            )
            if res["hotspot"] is not None:
                matched += 1
            else:
                skipped += 1

    return {
        "scene": slug,
        "notes_seen": matched + skipped,
        "hotspots_created": matched,
        "skipped": skipped,
    }
