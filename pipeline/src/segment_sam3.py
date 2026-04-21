"""Automatic mask generation + open-vocab classification.

Prefers SAM 3 if installed (facebookresearch/sam3); transparently
falls back to MobileSAM — the open-vocab lift comes from the bigger
interior vocab + prompt-ensembled CLIP, not the mask backbone.

Does not read any GT file. Output:
  pipeline/output/masks/view_XX.json
  list[{mask_id, bbox[x,y,w,h], mask_rle, class_name, class_confidence,
        stability_score}]
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import open_clip

from pipeline.src.vocab_interior import INTERIOR_VOCAB

VIEWS = Path("pipeline/output/views")
OUT = Path("pipeline/output/masks")
OUT.mkdir(parents=True, exist_ok=True)

PROMPT_TEMPLATES = [
    "a photo of a {c}",
    "a {c} in a room",
    "a close-up photo of a {c}",
    "an interior view containing a {c}",
]


def _rle(mask: np.ndarray) -> dict:
    flat = mask.flatten().astype(np.uint8)
    changes = np.diff(np.concatenate([[0], flat, [0]]))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return {
        "shape": list(mask.shape),
        "runs": [[int(s), int(e - s)] for s, e in zip(starts, ends)],
    }


def _load_mask_generator(device: str):
    """Return (generator, backbone_name). Try SAM 3 → MobileSAM."""
    try:
        from sam3 import Sam3Model, Sam3AutoMaskGenerator  # type: ignore
        ckpt = Path.home() / "sam3" / "checkpoints" / "sam3_base.pt"
        sam = Sam3Model.from_checkpoint(str(ckpt)).to(device).eval()
        mg = Sam3AutoMaskGenerator(sam, points_per_side=32, min_mask_region_area=500)
        return mg, "sam3"
    except Exception as exc:  # noqa: BLE001
        print(f"[segment] SAM 3 unavailable ({exc.__class__.__name__}); using MobileSAM")
    from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
    ckpt_candidates = [
        Path("pipeline/weights/mobile_sam.pt"),
        Path("/mnt/c/Users/91910/Downloads/sceneagent/pipeline/weights/mobile_sam.pt"),
    ]
    ckpt = next((p for p in ckpt_candidates if p.exists()), None)
    if ckpt is None:
        raise FileNotFoundError(
            "mobile_sam.pt not found in pipeline/weights/. Download from "
            "https://github.com/ChaoningZhang/MobileSAM/tree/master/weights"
        )
    sam = sam_model_registry["vit_t"](checkpoint=str(ckpt))
    sam.to(device).eval()
    mg = SamAutomaticMaskGenerator(sam, points_per_side=32, min_mask_region_area=500)
    return mg, "mobile_sam"


def _load_clip(device: str):
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.to(device).eval()
    with torch.no_grad():
        embs = []
        for c in INTERIOR_VOCAB:
            prompts = [t.format(c=c) for t in PROMPT_TEMPLATES]
            toks = tokenizer(prompts).to(device)
            e = model.encode_text(toks)
            e = e / e.norm(dim=-1, keepdim=True)
            embs.append(e.mean(0, keepdim=True))
        text_emb = torch.cat(embs, dim=0)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    return model, preprocess, text_emb


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[segment] device={device}")
    mg, backbone = _load_mask_generator(device)
    print(f"[segment] mask backbone: {backbone}")
    clip_m, preprocess, text_emb = _load_clip(device)
    print(f"[segment] CLIP vocab size: {text_emb.shape[0]}")

    views = sorted(VIEWS.glob("view_*.png"))
    print(f"[segment] {len(views)} views to process")

    for v in tqdm(views, desc="segment"):
        img_bgr = cv2.imread(str(v))
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        masks = mg.generate(img)
        out = []
        for i, m in enumerate(masks):
            seg = m["segmentation"]
            x, y, w, h = [int(a) for a in m["bbox"]]
            if w < 20 or h < 20:
                continue
            mask_crop = seg[y:y + h, x:x + w]
            crop = img[y:y + h, x:x + w] * mask_crop[..., None]
            pil = Image.fromarray(crop.astype(np.uint8))
            t = preprocess(pil).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = clip_m.encode_image(t)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                sims = (emb @ text_emb.T).squeeze(0)
                conf, idx = sims.max(dim=0)
            out.append({
                "mask_id": i,
                "bbox": [x, y, w, h],
                "mask_rle": _rle(seg),
                "class_name": INTERIOR_VOCAB[int(idx)],
                "class_confidence": float(conf),
                "stability_score": float(m.get("stability_score", 0)),
            })
        (OUT / (v.stem + ".json")).write_text(json.dumps(out))

    print(f"[segment] wrote {len(views)} mask files to {OUT}")


if __name__ == "__main__":
    main()
