"""Note → scene-object matcher.

Given a free-text note and a camera pose, rank candidate objects by:
  1. Frustum visibility from the pose (discarded if not visible).
  2. CLIP cosine similarity between the note embedding and each object's
     stored CLIP embedding.

Objects are expected to be dicts with at least:
    id           — opaque object id (UUID string or similar)
    class_name   — str
    centroid     — [x, y, z]
    clip_embedding — list[float] of length 512 (L2-normalized)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .clip_util import encode_text
from .geometry import CameraPose, is_point_visible_from_pose


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def rank_objects_for_note(
    note_text: str,
    pose: CameraPose,
    objects: list[dict[str, Any]],
    fov_deg: float = 60.0,
    max_distance: float = 15.0,
    min_similarity: float = 0.0,
    fallback_all_if_empty: bool = True,
) -> list[dict[str, Any]]:
    """Rank ``objects`` for ``note_text`` seen from ``pose``.

    Returns a list of dicts ``{object, similarity}`` sorted by similarity desc.
    Objects failing the frustum check are dropped. If nothing survives and
    ``fallback_all_if_empty`` is true, all objects are considered (useful when
    the synthesized trajectory doesn't match the real scene).
    """
    if not objects:
        return []

    note_emb = np.asarray(encode_text(note_text), dtype=np.float32)

    visible: list[dict[str, Any]] = []
    for obj in objects:
        centroid = obj.get("centroid")
        if centroid is None:
            continue
        if is_point_visible_from_pose(
            centroid, pose, fov_deg=fov_deg, max_distance=max_distance
        ):
            visible.append(obj)

    pool = visible if visible else (objects if fallback_all_if_empty else [])

    ranked: list[dict[str, Any]] = []
    for obj in pool:
        emb = obj.get("clip_embedding")
        if emb is None:
            continue
        sim = _cosine(note_emb, np.asarray(emb, dtype=np.float32))
        if sim < min_similarity:
            continue
        ranked.append({"object": obj, "similarity": sim})
    ranked.sort(key=lambda r: r["similarity"], reverse=True)
    return ranked
