"""Note-category classifier.

Uses Gemini Flash to tag a note with one of seven categories. API key is read
from ``GEMINI_API_KEY`` in the environment — never hardcoded.

Falls back to a keyword heuristic when:
  - the google-generativeai package is missing,
  - GEMINI_API_KEY is unset,
  - the API call raises for any reason.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Tuple

logger = logging.getLogger(__name__)

ALLOWED_CATEGORIES = (
    "feature",
    "included",
    "issue",
    "info",
    "spec",
    "story",
    "other",
)

_DEFAULT_MODEL = os.environ.get("LLM_MODEL", "gemini-2.0-flash")

_SYSTEM_PROMPT = """You are classifying short notes that a real-estate seller \
pinned to a 3D listing.  Output a single JSON object (no markdown, no code \
fences) with these exact keys:
  category:   one of feature|included|issue|info|spec|story|other
  confidence: float between 0 and 1

Meanings:
  feature   — positive selling point (e.g. "heated floor")
  included  — item staying with the property (e.g. "desk is included")
  issue     — defect or warning (e.g. "window sticks")
  info      — neutral operational info (e.g. "router is behind the TV")
  spec      — numeric / measured fact (e.g. "ceilings are 3.2m")
  story     — historical / anecdotal (e.g. "building was a bakery")
  other     — anything that fits none of the above

Return ONLY the JSON object."""


_KEYWORDS = {
    "issue": ["sticks", "broken", "leak", "loud", "noisy", "crack", "warning", "careful", "problem"],
    "included": ["included", "comes with", "stays", "belongs", "stay with"],
    "spec": ["m ", "meter", "metre", "square", "sqft", "sq ft", "sqm", "cm", "mm", "kg"],
    "story": ["year", "century", "19", "was a", "used to be", "history"],
    "feature": ["heated", "new", "renovated", "luxury", "beautiful", "lovely", "premium"],
    "info": ["behind", "located", "router", "wifi", "fuse", "meter reader"],
}


def _heuristic(text: str) -> Tuple[str, float]:
    t = text.lower()
    best_cat = "other"
    best_hits = 0
    for cat, words in _KEYWORDS.items():
        hits = sum(1 for w in words if w in t)
        if hits > best_hits:
            best_hits = hits
            best_cat = cat
    confidence = 0.4 if best_hits == 0 else min(0.6 + 0.1 * best_hits, 0.9)
    return best_cat, float(confidence)


def _coerce_category(raw: str) -> str:
    s = (raw or "").strip().lower()
    if s in ALLOWED_CATEGORIES:
        return s
    # Loose match: pick the first allowed category that appears as a word.
    for cat in ALLOWED_CATEGORIES:
        if re.search(rf"\b{cat}\b", s):
            return cat
    return "other"


def classify_category(text: str, model_name: str | None = None) -> Tuple[str, float]:
    """Return ``(category, confidence)`` for the supplied note text."""
    text = (text or "").strip()
    if not text:
        return "other", 0.0

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set; using heuristic classifier")
        return _heuristic(text)

    try:
        import google.generativeai as genai
    except Exception as exc:  # pragma: no cover
        logger.warning("google-generativeai import failed: %s", exc)
        return _heuristic(text)

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name or _DEFAULT_MODEL, system_instruction=_SYSTEM_PROMPT
        )
        response = model.generate_content(
            f"Classify this note:\n\n{text}",
            generation_config={"temperature": 0.0, "response_mime_type": "application/json"},
        )
        raw = (getattr(response, "text", "") or "").strip()
        # Strip accidental markdown fences.
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?", "", raw).strip()
            raw = re.sub(r"```$", "", raw).strip()
        data = json.loads(raw)
        category = _coerce_category(str(data.get("category", "other")))
        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(confidence, 1.0))
        return category, confidence
    except Exception as exc:
        logger.warning("Gemini classify_category failed (%s); using heuristic", exc)
        return _heuristic(text)
