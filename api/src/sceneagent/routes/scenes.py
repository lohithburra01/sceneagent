"""Scene + scene_objects read endpoints."""

from __future__ import annotations

import json
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

from ..db import pool

router = APIRouter(tags=["scenes"])


def _row_to_dict(row) -> dict[str, Any]:
    """Convert an asyncpg Record to a plain dict."""
    return {k: v for k, v in row.items()}


def _json_safe(value: Any) -> Any:
    """Make DB values JSON-serializable (UUIDs → str, datetimes → isoformat)."""
    import datetime
    import uuid

    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    return value


def _scene_to_response(row) -> dict[str, Any]:
    data = _row_to_dict(row)
    # camera_trajectory comes back as a JSON string from JSONB if no codec set;
    # asyncpg 0.29 returns a parsed object by default, but handle both.
    traj = data.get("camera_trajectory")
    if isinstance(traj, str):
        try:
            data["camera_trajectory"] = json.loads(traj)
        except Exception:
            pass
    return {k: _json_safe(v) for k, v in data.items()}


def _object_to_response(row) -> dict[str, Any]:
    data = _row_to_dict(row)
    # clip_embedding is a pgvector → numpy array; don't ship it in a list endpoint.
    data.pop("clip_embedding", None)
    return {k: _json_safe(v) for k, v in data.items()}


@router.get("/scenes/{slug}")
async def get_scene(slug: str) -> dict[str, Any]:
    """Return the scene row for the given slug."""
    p = pool()
    async with p.acquire() as con:
        row = await con.fetchrow(
            """
            SELECT id, slug, title, address, splat_url, camera_trajectory,
                   processed_at
            FROM scenes
            WHERE slug = $1
            """,
            slug,
        )
    if row is None:
        raise HTTPException(status_code=404, detail=f"scene {slug!r} not found")
    return _scene_to_response(row)


@router.get("/scenes/{slug}/objects")
async def list_scene_objects(
    slug: str,
    room: Optional[str] = Query(default=None),
    class_name: Optional[str] = Query(default=None),
    limit: int = Query(default=500, ge=1, le=5000),
) -> list[dict[str, Any]]:
    """Return scene_objects filtered by optional room and/or class_name."""
    p = pool()
    async with p.acquire() as con:
        scene = await con.fetchrow("SELECT id FROM scenes WHERE slug = $1", slug)
        if scene is None:
            raise HTTPException(status_code=404, detail=f"scene {slug!r} not found")
        scene_id = scene["id"]

        # Build filter clauses dynamically so NULL args mean "no filter".
        clauses = ["scene_id = $1"]
        args: list[Any] = [scene_id]
        if room is not None:
            args.append(room)
            clauses.append(f"room_label = ${len(args)}")
        if class_name is not None:
            args.append(class_name)
            clauses.append(f"class_name = ${len(args)}")

        args.append(limit)
        where = " AND ".join(clauses)
        rows = await con.fetch(
            f"""
            SELECT id, scene_id, instance_id, class_name, room_label,
                   centroid, bbox_min, bbox_max, source
            FROM scene_objects
            WHERE {where}
            ORDER BY instance_id
            LIMIT ${len(args)}
            """,
            *args,
        )
    return [_object_to_response(r) for r in rows]
