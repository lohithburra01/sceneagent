"""Chat endpoint (JSON + optional SSE streaming)."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..agent.graph import run_agent
from ..db import pool

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


class ChatBody(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)


async def _scene_exists(slug: str) -> bool:
    p = pool()
    async with p.acquire() as con:
        row = await con.fetchrow("SELECT 1 FROM scenes WHERE slug = $1", slug)
    return row is not None


@router.post("/scenes/{slug}/chat")
async def chat(slug: str, body: ChatBody) -> dict[str, Any]:
    """Run the agent loop once and return the final response + trace."""
    try:
        exists = await _scene_exists(slug)
    except Exception as exc:
        logger.warning("scene lookup failed: %s", exc)
        raise HTTPException(status_code=503, detail="scene database unavailable") from exc
    if not exists:
        raise HTTPException(status_code=404, detail=f"scene {slug!r} not found")

    result = await run_agent(slug, body.message)
    return {
        "scene": slug,
        "message": body.message,
        "response": result.get("response"),
        "tool_calls": result.get("tool_calls"),
    }


@router.post("/scenes/{slug}/chat/stream")
async def chat_stream(slug: str, body: ChatBody, request: Request):
    """Streaming variant. Emits one SSE event per tool call + final answer.

    For v1 the agent runs to completion, then replays each tool-call event.
    This keeps latency unchanged but gives the frontend the same UX scaffold
    (thinking dots → tool results → final message) it would get from a real
    streaming planner.
    """
    try:
        from sse_starlette.sse import EventSourceResponse
    except Exception as exc:  # pragma: no cover
        raise HTTPException(
            status_code=501, detail="sse-starlette not installed"
        ) from exc

    if not await _scene_exists(slug):
        raise HTTPException(status_code=404, detail=f"scene {slug!r} not found")

    async def events():
        try:
            result = await run_agent(slug, body.message)
        except Exception as exc:
            logger.warning("agent run failed: %s", exc)
            yield {"event": "error", "data": json.dumps({"error": str(exc)})}
            return

        for call in result.get("tool_calls", []):
            if await request.is_disconnected():
                return
            yield {
                "event": "tool_call",
                "data": json.dumps(
                    {
                        "tool": call.get("tool"),
                        "args": call.get("args"),
                    }
                ),
            }
            # Small delay so the frontend can render intermediate states.
            await asyncio.sleep(0.05)

        yield {
            "event": "answer",
            "data": json.dumps({"response": result.get("response")}),
        }

    return EventSourceResponse(events())
