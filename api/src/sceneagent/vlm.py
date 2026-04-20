"""Gemini Flash vision wrapper.

``describe_image`` accepts a base64-encoded PNG/JPEG and a question, and
returns a short description focused on answering that question. Falls back to
an informative stub when ``GEMINI_API_KEY`` is unset or the SDK is missing.
"""

from __future__ import annotations

import base64
import logging
import os

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = os.environ.get("VLM_MODEL", "gemini-2.0-flash")


def _fallback(question: str) -> str:
    return (
        "[VLM unavailable — no GEMINI_API_KEY or google-generativeai not "
        f"installed]. Skipping visual description for: {question!r}"
    )


def describe_image(
    image_base64: str,
    question: str,
    *,
    mime_type: str = "image/png",
    model_name: str | None = None,
) -> str:
    """Describe ``image_base64`` in the context of ``question``.

    Returns a short natural-language string. Never raises — errors fall back
    to a stub string so the agent can proceed.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set; describe_image returning stub")
        return _fallback(question)

    try:
        import google.generativeai as genai
    except Exception as exc:  # pragma: no cover
        logger.warning("google-generativeai import failed: %s", exc)
        return _fallback(question)

    try:
        # Decode the base64 to raw bytes — google-generativeai expects the
        # binary payload, not the base64 string.
        try:
            raw = base64.b64decode(image_base64, validate=True)
        except Exception:
            raw = base64.b64decode(image_base64)

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name or _DEFAULT_MODEL)
        prompt = (
            "You are assisting a 3D real-estate listing assistant. "
            "Describe only what you can see in the image that is relevant to "
            f"this question: {question}\n\nKeep it to 1–3 sentences."
        )
        response = model.generate_content(
            [
                {"mime_type": mime_type, "data": raw},
                prompt,
            ],
            generation_config={"temperature": 0.2},
        )
        text = (getattr(response, "text", "") or "").strip()
        return text or _fallback(question)
    except Exception as exc:
        logger.warning("Gemini describe_image failed: %s", exc)
        return _fallback(question)
