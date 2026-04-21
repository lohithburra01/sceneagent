"""Scene-interaction tools shared by the LangGraph agent and MCP server.

Every function is ``async`` and accepts a ``scene_slug`` as its first argument
(except ``measure_distance`` which is stateless).  Returns plain JSON-able
dicts so both transports (internal agent + MCP) can serialize identically.
"""

from __future__ import annotations

import math
import uuid
from typing import Any, Optional

from ..clip_util import encode_text
from ..db import pool


async def _scene_id(con, scene_slug: str) -> uuid.UUID:
    row = await con.fetchrow("SELECT id FROM scenes WHERE slug = $1", scene_slug)
    if row is None:
        raise ValueError(f"scene {scene_slug!r} not found")
    return row["id"]


async def list_objects(
    scene_slug: str,
    room: Optional[str] = None,
    class_name: Optional[str] = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return scene_objects filtered by optional room / class_name."""
    p = pool()
    async with p.acquire() as con:
        scene_id = await _scene_id(con, scene_slug)
        clauses = ["scene_id = $1"]
        args: list[Any] = [scene_id]
        if room is not None:
            args.append(room)
            clauses.append(f"room_label = ${len(args)}")
        if class_name is not None:
            args.append(class_name)
            clauses.append(f"class_name = ${len(args)}")
        args.append(int(limit))
        where = " AND ".join(clauses)
        rows = await con.fetch(
            f"""
            SELECT id, instance_id, class_name, room_label,
                   centroid, bbox_min, bbox_max
            FROM scene_objects
            WHERE {where}
            ORDER BY instance_id
            LIMIT ${len(args)}
            """,
            *args,
        )
    out = []
    for r in rows:
        bmin = list(r["bbox_min"]) if r["bbox_min"] is not None else None
        bmax = list(r["bbox_max"]) if r["bbox_max"] is not None else None
        # Confidence proxy: log-scaled bbox diagonal — same approach as the
        # /scenes/:slug/detections endpoint, so the agent and the sidebar
        # see the same numbers.
        conf = None
        if bmin and bmax:
            import math as _m
            diag = _m.sqrt(sum((bmax[i] - bmin[i]) ** 2 for i in range(3)))
            conf = 0.4 + 0.55 * min(1.0, _m.log10(max(diag, 0.05) * 10) / 1.7)
        out.append({
            "id": str(r["id"]),
            "instance_id": r["instance_id"],
            "class_name": r["class_name"],
            "room_label": r["room_label"],
            "centroid": list(r["centroid"]) if r["centroid"] is not None else None,
            "bbox_min": bmin,
            "bbox_max": bmax,
            "confidence": conf,
        })
    # Sort by confidence descending so the agent can pick "highest" trivially
    # without re-sorting in the planner.
    out.sort(key=lambda d: -(d["confidence"] or 0))
    return out


async def list_hotspots(
    scene_slug: str, category: Optional[str] = None
) -> list[dict[str, Any]]:
    """Return hotspots joined with note text + object class."""
    p = pool()
    async with p.acquire() as con:
        scene_id = await _scene_id(con, scene_slug)
        clauses = ["n.scene_id = $1"]
        args: list[Any] = [scene_id]
        if category is not None:
            args.append(category)
            clauses.append(f"n.category = ${len(args)}")
        where = " AND ".join(clauses)
        rows = await con.fetch(
            f"""
            SELECT h.id AS hotspot_id, h.note_id AS note_id,
                   h.object_id AS object_id, h.match_confidence, h.position,
                   n.text AS note_text, n.category, n.video_timestamp,
                   o.class_name, o.room_label
            FROM hotspots h
            JOIN notes n ON n.id = h.note_id
            LEFT JOIN scene_objects o ON o.id = h.object_id
            WHERE {where}
            ORDER BY n.video_timestamp
            """,
            *args,
        )
    return [
        {
            "id": str(r["hotspot_id"]),
            "note_id": str(r["note_id"]) if r["note_id"] else None,
            "object_id": str(r["object_id"]) if r["object_id"] else None,
            "match_confidence": float(r["match_confidence"]),
            "position": list(r["position"]) if r["position"] is not None else None,
            "note_text": r["note_text"],
            "category": r["category"],
            "video_timestamp": float(r["video_timestamp"])
            if r["video_timestamp"] is not None
            else None,
            "class_name": r["class_name"],
            "room_label": r["room_label"],
        }
        for r in rows
    ]


async def find_by_description(
    scene_slug: str, text: str, limit: int = 5
) -> list[dict[str, Any]]:
    """Top-N objects by cosine similarity to the query text (pgvector `<=>`)."""
    embedding = encode_text(text)
    p = pool()
    async with p.acquire() as con:
        scene_id = await _scene_id(con, scene_slug)
        rows = await con.fetch(
            """
            SELECT id, instance_id, class_name, room_label,
                   centroid, bbox_min, bbox_max,
                   1.0 - (clip_embedding <=> $2) AS score
            FROM scene_objects
            WHERE scene_id = $1
            ORDER BY clip_embedding <=> $2
            LIMIT $3
            """,
            scene_id,
            embedding,
            int(limit),
        )
    return [
        {
            "object_id": str(r["id"]),
            "instance_id": r["instance_id"],
            "class_name": r["class_name"],
            "room_label": r["room_label"],
            "centroid": list(r["centroid"]) if r["centroid"] is not None else None,
            "bbox_min": list(r["bbox_min"]) if r["bbox_min"] is not None else None,
            "bbox_max": list(r["bbox_max"]) if r["bbox_max"] is not None else None,
            "score": float(r["score"]),
        }
        for r in rows
    ]


async def measure_distance(
    scene_slug: str,  # unused but kept for API parity with the other tools
    point_a: list[float],
    point_b: list[float],
) -> dict[str, Any]:
    """Return Euclidean distance in meters between two 3D points."""
    a = [float(x) for x in point_a]
    b = [float(x) for x in point_b]
    if len(a) != 3 or len(b) != 3:
        raise ValueError("point_a and point_b must both be 3-vectors")
    d = math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))
    return {"meters": float(d), "point_a": a, "point_b": b}


_FOCUS_TO_CATEGORIES = {
    "features": ("feature",),
    "issues": ("issue",),
    "included": ("included",),
    "specs": ("spec",),
    "info": ("info",),
    "story": ("story",),
}


async def plan_tour(
    scene_slug: str,
    focus: Optional[str] = None,
    max_stops: int = 6,
) -> list[dict[str, Any]]:
    """Build a simple tour through (optionally filtered) hotspots."""
    target_cats = _FOCUS_TO_CATEGORIES.get((focus or "").lower())
    hotspots = await list_hotspots(scene_slug)
    if target_cats:
        hotspots = [h for h in hotspots if h.get("category") in target_cats]

    stops: list[dict[str, Any]] = []
    for h in hotspots[: int(max_stops)]:
        narration = h.get("note_text") or ""
        if h.get("class_name"):
            narration = f"{h['class_name']}: {narration}".strip(": ").strip()
        stops.append(
            {
                "hotspot_id": h.get("id"),
                "object_id": h.get("object_id"),
                "position": h.get("position"),
                "dwell_seconds": 4.0,
                "narration_hint": narration,
                "highlight_hotspot_id": h.get("id"),
                "category": h.get("category"),
            }
        )
    return stops


# A machine-readable manifest so the LangGraph node can list + dispatch tools.
TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "list_objects",
        "description": "List objects in the scene (optionally filtered by room or class_name).",
        "parameters": {
            "room": "string|null",
            "class_name": "string|null",
            "limit": "integer (default 50)",
        },
    },
    {
        "name": "list_hotspots",
        "description": "List hotspots (pinned notes) optionally filtered by category (feature|included|issue|info|spec|story|other).",
        "parameters": {"category": "string|null"},
    },
    {
        "name": "find_by_description",
        "description": "Semantic search: given a free-text description, return the top objects.",
        "parameters": {"text": "string", "limit": "integer (default 5)"},
    },
    {
        "name": "measure_distance",
        "description": "Euclidean distance in meters between two 3D points.",
        "parameters": {
            "point_a": "[x,y,z]",
            "point_b": "[x,y,z]",
        },
    },
    {
        "name": "plan_tour",
        "description": "Plan a short camera tour through hotspots (optionally focused).",
        "parameters": {
            "focus": "string|null (features|issues|included|specs|info|story)",
            "max_stops": "integer (default 6)",
        },
    },
    {
        "name": "render_view",
        "description": "Return a base64 PNG preview rendered from the given world-space position.",
        "parameters": {"position": "[x,y,z]"},
    },
    {
        "name": "describe_image",
        "description": "Ask the VLM to describe the most recently rendered view relative to a question.",
        "parameters": {
            "image_base64": "string",
            "question": "string",
        },
    },
    {
        "name": "answer",
        "description": "Finalize and deliver the user-facing response.",
        "parameters": {"text": "string"},
    },
]
