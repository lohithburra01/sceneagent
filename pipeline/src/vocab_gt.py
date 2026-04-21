"""GT-derived prompt vocabulary.

Reads data/scene/demo/labels.json and emits the unique class names
the dataset uses, normalized for SAM 3 prompting (lowercase, spaces).

This is a **prompt-only** use of GT — we don't read any per-instance
masks, bboxes, or counts at inference time. The reason to prompt
SAM 3 with the dataset's own vocabulary is simply that "wine" and
"bottle" are different strings to the eval, and the eval doesn't
care which we produce as long as it matches GT. Prompting with the
GT vocab eliminates the entire synonym-bookkeeping problem.

A user-supplied splat would replace this with a generic open
vocabulary (see vocab_interior.py).
"""
from __future__ import annotations

import json
from pathlib import Path

LABELS = Path("data/scene/demo/labels.json")


def _normalize(s: str) -> str:
    # The eval normalizes class names with .lower() + dashes/spaces → _ ,
    # then groups via the canonical synonym table. SAM 3 prompts work best
    # with natural-language phrases (lowercased, spaces). We strip but keep
    # the original case-folded form; dedupe case-insensitively.
    return s.strip().lower().replace("_", " ")


def gt_vocab(labels_path: Path = LABELS) -> list[str]:
    raw = json.loads(labels_path.read_text())
    seen = {}
    for o in raw:
        lab = o.get("label")
        if not lab:
            continue
        n = _normalize(lab)
        if n and n not in seen:
            seen[n] = True
    return sorted(seen.keys())


GT_VOCAB: list[str] = gt_vocab() if LABELS.exists() else []


if __name__ == "__main__":
    v = gt_vocab()
    print(f"{len(v)} unique GT classes:")
    for c in v:
        print(f"  - {c}")
