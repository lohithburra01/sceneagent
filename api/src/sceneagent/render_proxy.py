"""Picks a pre-rendered splat view closest to a given world-space position.

Track M (the CV pipeline) renders 30 views into ``pipeline/output/views/`` plus
an ``_intrinsics.json`` listing the poses. At agent-tool time we can't
rasterize a splat from Python, so we serve the closest pre-rendered frame as
a proxy.

Returns a gray 640x360 fallback PNG whenever the views directory is absent
or contains no matching frames.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_FALLBACK_WIDTH = 640
_FALLBACK_HEIGHT = 360
_FALLBACK_COLOR = (128, 128, 128)


def _candidate_view_dirs() -> list[Path]:
    """Search paths for rendered views. Env override wins."""
    override = os.environ.get("SCENEAGENT_VIEWS_DIR")
    paths = []
    if override:
        paths.append(Path(override))
    paths.extend(
        [
            Path("pipeline/output/views"),
            Path("/app/data/pipeline/output/views"),
            Path("/app/data/views"),
        ]
    )
    return paths


def _load_intrinsics(views_dir: Path) -> Optional[dict[str, Any]]:
    candidate = views_dir / "_intrinsics.json"
    if not candidate.exists():
        return None
    try:
        return json.loads(candidate.read_text())
    except Exception as exc:  # pragma: no cover
        logger.warning("failed to parse %s: %s", candidate, exc)
        return None


def _fallback_png_bytes() -> bytes:
    """Produce a solid-gray PNG so callers always get a usable image."""
    try:
        from PIL import Image

        img = Image.new("RGB", (_FALLBACK_WIDTH, _FALLBACK_HEIGHT), _FALLBACK_COLOR)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:  # pragma: no cover
        # Absolute fallback: the minimal 1x1 PNG (gray pixel).
        return (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00"
            b"\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDAT\x08\x99c"
            b"\xf8\xff\xff?\x00\x05\xfe\x02\xfe\xd4\x05\xdf\x7f\x00\x00\x00\x00"
            b"IEND\xaeB`\x82"
        )


def _png_dimensions(data: bytes) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as img:
            return img.size  # (width, height)
    except Exception:
        return _FALLBACK_WIDTH, _FALLBACK_HEIGHT


def _distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(min(3, len(a), len(b)))))


def render_view(
    scene_slug: str,
    position: list[float],
    *,
    views_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Return ``{image_base64, width, height, source}`` for the closest view."""
    search_paths = [views_dir] if views_dir else _candidate_view_dirs()

    chosen_dir: Optional[Path] = None
    for p in search_paths:
        if p is not None and p.exists() and p.is_dir():
            chosen_dir = p
            break

    if chosen_dir is None:
        logger.info("render_view: no views dir; serving gray fallback")
        data = _fallback_png_bytes()
        return {
            "image_base64": base64.b64encode(data).decode("ascii"),
            "width": _FALLBACK_WIDTH,
            "height": _FALLBACK_HEIGHT,
            "source": "fallback:no-views-dir",
        }

    intrinsics = _load_intrinsics(chosen_dir)
    frames = sorted(chosen_dir.glob("view_*.png"))
    if not frames:
        logger.info("render_view: no frames in %s; serving gray fallback", chosen_dir)
        data = _fallback_png_bytes()
        return {
            "image_base64": base64.b64encode(data).decode("ascii"),
            "width": _FALLBACK_WIDTH,
            "height": _FALLBACK_HEIGHT,
            "source": "fallback:empty-views",
        }

    # Build index → position map from intrinsics (if present) so we can pick
    # the closest frame by Euclidean distance.
    best_path: Path = frames[0]
    best_dist = float("inf")
    poses = (intrinsics or {}).get("poses") or []
    if poses:
        for i, pose in enumerate(poses):
            pos = pose.get("position") or pose.get("pos") or None
            if pos is None:
                continue
            d = _distance(position, pos)
            if d < best_dist:
                # Match to the file with the corresponding index, if it exists.
                candidate = chosen_dir / f"view_{i:03d}.png"
                if candidate.exists():
                    best_dist = d
                    best_path = candidate
    else:
        # Without intrinsics we just return the first available frame.
        best_path = frames[0]
        best_dist = 0.0

    try:
        data = best_path.read_bytes()
    except Exception as exc:  # pragma: no cover
        logger.warning("failed reading %s: %s", best_path, exc)
        data = _fallback_png_bytes()
        return {
            "image_base64": base64.b64encode(data).decode("ascii"),
            "width": _FALLBACK_WIDTH,
            "height": _FALLBACK_HEIGHT,
            "source": f"fallback:read-error:{best_path.name}",
        }

    width, height = _png_dimensions(data)
    return {
        "image_base64": base64.b64encode(data).decode("ascii"),
        "width": width,
        "height": height,
        "source": best_path.name,
        "distance": float(best_dist) if best_dist != float("inf") else None,
    }
