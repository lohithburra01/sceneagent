"""FastMCP server exposing the SceneAgent tools.

Importing this module builds a :class:`FastMCP` instance. Run the server from
the command line with::

    python -m sceneagent.mcp_server

The same tool implementations are reused by the in-process LangGraph agent —
see :mod:`sceneagent.agent.tools`.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .agent import tools as _tools
from .render_proxy import render_view as _render_view

logger = logging.getLogger(__name__)

try:  # pragma: no cover — import guarded so unit-tests can load submodules
    from mcp.server.fastmcp import FastMCP
except Exception:  # pragma: no cover
    try:
        from fastmcp import FastMCP  # type: ignore[no-redef]
    except Exception as exc:
        FastMCP = None  # type: ignore[assignment]
        logger.warning("FastMCP not available: %s", exc)


def build_mcp() -> Any:
    """Construct and return the FastMCP server (or ``None`` if unavailable)."""
    if FastMCP is None:  # pragma: no cover
        return None

    mcp = FastMCP("sceneagent")

    @mcp.tool()
    async def list_objects(
        scene_slug: str,
        room: Optional[str] = None,
        class_name: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List scene_objects (optionally filtered by room or class_name)."""
        return await _tools.list_objects(scene_slug, room, class_name, limit)

    @mcp.tool()
    async def list_hotspots(
        scene_slug: str, category: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """List hotspots, optionally filtered by category."""
        return await _tools.list_hotspots(scene_slug, category)

    @mcp.tool()
    async def find_by_description(
        scene_slug: str, text: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        """Top-N objects by semantic similarity to the query text."""
        return await _tools.find_by_description(scene_slug, text, limit)

    @mcp.tool()
    async def measure_distance(
        scene_slug: str, point_a: list[float], point_b: list[float]
    ) -> dict[str, Any]:
        """Euclidean distance in meters between two 3D points."""
        return await _tools.measure_distance(scene_slug, point_a, point_b)

    @mcp.tool()
    async def plan_tour(
        scene_slug: str, focus: Optional[str] = None, max_stops: int = 6
    ) -> list[dict[str, Any]]:
        """Plan a short narrated camera tour through hotspots."""
        return await _tools.plan_tour(scene_slug, focus, max_stops)

    @mcp.tool()
    async def render_view(scene_slug: str, position: list[float]) -> dict[str, Any]:
        """Return a base64-PNG preview rendered from ``position``."""
        return _render_view(scene_slug, position)

    return mcp


# Module-level singleton so ``python -m sceneagent.mcp_server`` works and
# in-process callers can introspect the server if needed.
mcp = build_mcp()


def main() -> None:  # pragma: no cover
    if mcp is None:
        raise SystemExit("FastMCP is not installed; add 'mcp' to requirements.")
    # FastMCP exposes a blocking .run() that speaks stdio by default.
    mcp.run()


if __name__ == "__main__":  # pragma: no cover
    main()
