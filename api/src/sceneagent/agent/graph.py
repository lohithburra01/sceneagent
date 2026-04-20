"""LangGraph single-node agent loop for SceneAgent.

The flow is deliberately simple for v1:

    START ─▶ execute ─▶ (tool?) ─▶ execute ─▶ ... ─▶ answer ─▶ END

Each iteration:
  1. Build a prompt from the system preamble, the tool manifest, and the
     conversation history.
  2. Ask Gemini to output a single JSON object ``{"tool": ..., "args": ...}``.
     ``tool == "answer"`` terminates the loop.
  3. Dispatch the chosen tool, append the observation, and loop.

When ``GEMINI_API_KEY`` isn't set (CI / tests), the agent falls back to a
rule-based planner so the HTTP surface stays functional.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from . import tools as scene_tools
from ..render_proxy import render_view as _render_view
from ..vlm import describe_image as _describe_image

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 6
_MODEL_NAME = os.environ.get("LLM_MODEL", "gemini-2.0-flash")

SYSTEM_PROMPT = """You are SceneAgent — a 3D real-estate listing concierge.
You can see the current listing only through the tools provided to you.

Available tools (always call ONE per turn; finish with tool=answer):
  list_objects(room?, class_name?, limit?)
  list_hotspots(category?)
  find_by_description(text, limit?)
  measure_distance(point_a, point_b)
  plan_tour(focus?, max_stops?)
  render_view(position)                 — returns a base64 PNG preview
  describe_image(image_base64, question) — vision description of a PNG
  answer(text)                          — final response to the user

Rules:
  - Output STRICT JSON only, no prose: {"tool": "<name>", "args": {...}}.
  - Do not invent object IDs or positions — get them from a tool first.
  - Use render_view + describe_image when the question is visual/subjective.
  - When you have enough information, call answer with your final reply.
  - Hotspot categories: feature, included, issue, info, spec, story, other.
"""


@dataclass
class AgentState:
    scene_slug: str
    user_message: str
    history: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    response: Optional[str] = None
    iterations: int = 0


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

ToolFn = Callable[..., Awaitable[Any]]


async def _tool_render_view(
    scene_slug: str, position: list[float], **_: Any
) -> dict[str, Any]:
    return _render_view(scene_slug, position)


async def _tool_describe_image(
    scene_slug: str, image_base64: str, question: str, **_: Any
) -> dict[str, Any]:  # scene_slug unused but kept for a uniform signature
    return {"description": _describe_image(image_base64, question)}


async def _tool_answer(scene_slug: str, text: str = "", **_: Any) -> dict[str, Any]:
    return {"text": text}


async def _dispatch(tool: str, scene_slug: str, args: dict[str, Any]) -> Any:
    args = args or {}
    if tool == "list_objects":
        return await scene_tools.list_objects(
            scene_slug,
            room=args.get("room"),
            class_name=args.get("class_name"),
            limit=int(args.get("limit", 50)),
        )
    if tool == "list_hotspots":
        return await scene_tools.list_hotspots(scene_slug, category=args.get("category"))
    if tool == "find_by_description":
        return await scene_tools.find_by_description(
            scene_slug,
            text=str(args.get("text", "")),
            limit=int(args.get("limit", 5)),
        )
    if tool == "measure_distance":
        return await scene_tools.measure_distance(
            scene_slug,
            point_a=list(args.get("point_a") or []),
            point_b=list(args.get("point_b") or []),
        )
    if tool == "plan_tour":
        return await scene_tools.plan_tour(
            scene_slug,
            focus=args.get("focus"),
            max_stops=int(args.get("max_stops", 6)),
        )
    if tool == "render_view":
        return await _tool_render_view(
            scene_slug, position=list(args.get("position") or [0, 0, 1.7])
        )
    if tool == "describe_image":
        return await _tool_describe_image(
            scene_slug,
            image_base64=str(args.get("image_base64", "")),
            question=str(args.get("question", "")),
        )
    if tool == "answer":
        return await _tool_answer(scene_slug, text=str(args.get("text", "")))
    raise ValueError(f"unknown tool: {tool!r}")


# ---------------------------------------------------------------------------
# Planner — Gemini if available, heuristic otherwise
# ---------------------------------------------------------------------------


def _extract_json_object(text: str) -> Optional[dict[str, Any]]:
    """Best-effort JSON-object extractor. Tolerates ```json fences."""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?", "", t).strip()
        t = re.sub(r"```$", "", t).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    # Fallback: find the first {...} block.
    match = re.search(r"\{.*\}", t, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _heuristic_plan(state: AgentState) -> dict[str, Any]:
    """No-LLM planner used when GEMINI_API_KEY is absent."""
    msg = state.user_message.lower()
    if state.iterations <= 1:
        if "issue" in msg or "problem" in msg or "wrong" in msg:
            return {"tool": "list_hotspots", "args": {"category": "issue"}}
        if "include" in msg:
            return {"tool": "list_hotspots", "args": {"category": "included"}}
        if "tour" in msg or "walk" in msg:
            focus = "features" if "feature" in msg else None
            return {"tool": "plan_tour", "args": {"focus": focus}}
        if any(k in msg for k in ("tall", "ceiling", "how big", "size", "distance", "how far")):
            return {"tool": "list_hotspots", "args": {"category": "spec"}}
        # Default: semantic search on the raw message.
        return {"tool": "find_by_description", "args": {"text": state.user_message, "limit": 5}}
    # Second turn: always answer with a summary of what we found.
    last = state.tool_calls[-1] if state.tool_calls else None
    if last is None:
        text = "I couldn't find enough information to answer that."
    else:
        obs = last.get("observation")
        if isinstance(obs, list):
            if not obs:
                text = "I didn't find anything matching that in the scene."
            else:
                sample = obs[:3]
                preview = ", ".join(
                    o.get("note_text") or o.get("class_name") or o.get("narration_hint") or "item"
                    for o in sample
                    if isinstance(o, dict)
                )
                text = f"Found {len(obs)} item(s): {preview}."
        elif isinstance(obs, dict) and "meters" in obs:
            text = f"The distance is {obs['meters']:.2f} meters."
        else:
            text = "Here's what I found based on your question."
    return {"tool": "answer", "args": {"text": text}}


def _gemini_plan(state: AgentState) -> Optional[dict[str, Any]]:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        import google.generativeai as genai
    except Exception as exc:  # pragma: no cover
        logger.warning("google-generativeai import failed: %s", exc)
        return None

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(_MODEL_NAME, system_instruction=SYSTEM_PROMPT)
        # Build transcript — summarize tool outputs so we don't blow context.
        parts: list[str] = [f"Scene slug: {state.scene_slug}", f"User: {state.user_message}"]
        for i, call in enumerate(state.tool_calls):
            obs = call.get("observation")
            # Truncate base64 PNGs so they don't flood the context window.
            if isinstance(obs, dict) and "image_base64" in obs:
                obs = {**obs, "image_base64": "<base64 elided>"}
            obs_str = json.dumps(obs)[:4000]
            parts.append(
                f"Turn {i+1}: tool={call.get('tool')} args={json.dumps(call.get('args'))} "
                f"→ {obs_str}"
            )
        parts.append(
            "Respond now. Output exactly one JSON object: "
            '{"tool": "<name>", "args": {...}}.'
        )
        prompt = "\n".join(parts)
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.2,
                "response_mime_type": "application/json",
            },
        )
        raw = (getattr(response, "text", "") or "").strip()
        plan = _extract_json_object(raw)
        if not plan or "tool" not in plan:
            logger.warning("Gemini plan missing 'tool': %r", raw)
            return None
        plan.setdefault("args", {})
        if not isinstance(plan["args"], dict):
            plan["args"] = {}
        return plan
    except Exception as exc:
        logger.warning("Gemini plan failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


async def run_agent(scene_slug: str, user_message: str) -> dict[str, Any]:
    """Run the agent loop and return ``{response, tool_calls}``."""
    state = AgentState(scene_slug=scene_slug, user_message=user_message)

    for _ in range(MAX_ITERATIONS):
        state.iterations += 1
        plan = _gemini_plan(state) or _heuristic_plan(state)
        tool = str(plan.get("tool") or "answer")
        args = plan.get("args") or {}

        try:
            observation = await _dispatch(tool, scene_slug, args)
        except Exception as exc:
            logger.warning("tool %s failed: %s", tool, exc)
            observation = {"error": str(exc)}

        # Keep the PNG out of the stored transcript (agents echo back history).
        stored_obs = observation
        if isinstance(observation, dict) and "image_base64" in observation:
            stored_obs = {
                **{k: v for k, v in observation.items() if k != "image_base64"},
                "image_base64_len": len(observation.get("image_base64") or ""),
            }

        state.tool_calls.append(
            {"tool": tool, "args": args, "observation": stored_obs}
        )

        if tool == "answer":
            state.response = (
                observation.get("text") if isinstance(observation, dict) else None
            ) or ""
            break

    if state.response is None:
        # Loop exhausted without an answer → force-synthesize a closing line.
        state.response = (
            "I've looked into your question but couldn't reach a confident answer "
            "in the available steps."
        )

    return {"response": state.response, "tool_calls": state.tool_calls}
