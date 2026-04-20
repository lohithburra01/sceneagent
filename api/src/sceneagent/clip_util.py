"""OpenCLIP text encoder helpers.

The CLIP model is loaded lazily on first call and cached for subsequent uses.
All embeddings are L2-normalized so cosine similarity reduces to a dot product.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def _load_model() -> tuple[Any, Any]:
    """Load OpenCLIP ViT-B-32 (OpenAI weights) + tokenizer. Cached process-wide."""
    # Imports live inside the function so module import doesn't force torch
    # to be available (useful for the FastAPI routers that don't touch CLIP).
    import open_clip
    import torch

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    # Ensure all inference happens in eval+no_grad mode.
    for p in model.parameters():
        p.requires_grad_(False)
    # Stash torch on the returned tuple so encode_text doesn't have to import.
    return model, tokenizer


def encode_text(text: str) -> list[float]:
    """Return a 512-dim L2-normalized OpenCLIP ViT-B-32 text embedding."""
    import torch

    model, tokenizer = _load_model()
    tokens = tokenizer([text])
    with torch.no_grad():
        emb = model.encode_text(tokens)
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-9)
    return emb.squeeze(0).cpu().tolist()


def encode_texts(texts: list[str]) -> list[list[float]]:
    """Batch encode. Returns a list of 512-dim L2-normalized vectors."""
    import torch

    if not texts:
        return []
    model, tokenizer = _load_model()
    tokens = tokenizer(texts)
    with torch.no_grad():
        emb = model.encode_text(tokens)
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-9)
    return emb.cpu().tolist()
