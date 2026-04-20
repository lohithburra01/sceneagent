"""Run MobileSAM + CLIP zero-shot classification on each rendered view.

Output: pipeline/output/masks/view_<idx>.json with list of
  {mask_id, bbox, class_name, class_confidence, mask_rle, stability_score}
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import open_clip
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator

VIEWS_DIR = Path("pipeline/output/views")
OUT_DIR = Path("pipeline/output/masks")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Vocabulary tuned for interior scenes: residential + commercial (wine bar / shop)
HOME_CLASSES = [
    "wall", "floor", "ceiling", "window", "door", "bed", "sofa", "chair",
    "table", "desk", "lamp", "television", "bookshelf", "cabinet", "sink",
    "toilet", "bathtub", "stove", "refrigerator", "radiator", "mirror",
    "plant", "painting", "rug", "curtain", "pillow", "nightstand", "stool",
    "ottoman", "shelf", "shower", "oven", "microwave", "washing_machine",
    "dryer", "shoe", "clock", "vase", "fireplace", "armchair",
    # commercial / wine bar
    "bottle", "wine_bottle", "wine_glass", "cup", "high_chair", "dining_table",
    "chandelier", "spotlight", "downlight", "fruit", "decorative_painting",
    "ornament", "jar", "bar_counter",
]


def load_sam():
    ckpt = Path("pipeline/weights/mobile_sam.pt")
    ckpt.parent.mkdir(exist_ok=True, parents=True)
    if not ckpt.exists():
        import urllib.request
        url = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
        urllib.request.urlretrieve(url, ckpt)
    sam = sam_model_registry["vit_t"](checkpoint=str(ckpt))
    sam.eval()
    return SamAutomaticMaskGenerator(sam, points_per_side=32, min_mask_region_area=500)


def load_clip():
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    prompts = [f"a photo of a {c.replace('_', ' ')}" for c in HOME_CLASSES]
    with torch.no_grad():
        text_emb = model.encode_text(tokenizer(prompts))
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    return model, preprocess, text_emb


def mask_to_rle(mask: np.ndarray) -> dict:
    flat = mask.flatten().astype(np.uint8)
    changes = np.diff(np.concatenate([[0], flat, [0]]))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return {"shape": list(mask.shape), "runs": [[int(s), int(e - s)] for s, e in zip(starts, ends)]}


def segment_view(img_path: Path, mask_gen, clip_model, clip_preprocess, clip_text_emb):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return []
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    masks = mask_gen.generate(img_rgb)

    results = []
    for m_idx, m in enumerate(masks):
        seg = m["segmentation"]
        x, y, w, h = [int(v) for v in m["bbox"]]
        if w < 20 or h < 20:
            continue
        crop = img_rgb[y:y + h, x:x + w]
        mask_crop = seg[y:y + h, x:x + w]
        masked_crop = crop * mask_crop[..., None]
        pil = Image.fromarray(masked_crop.astype(np.uint8))
        tensor = clip_preprocess(pil).unsqueeze(0)
        with torch.no_grad():
            img_emb = clip_model.encode_image(tensor)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            sims = (img_emb @ clip_text_emb.T).squeeze(0)
            conf, cls_idx = sims.max(dim=0)
        results.append({
            "mask_id": m_idx,
            "bbox": [x, y, w, h],
            "class_name": HOME_CLASSES[int(cls_idx)],
            "class_confidence": float(conf),
            "mask_rle": mask_to_rle(seg),
            "stability_score": float(m.get("stability_score", 0)),
        })
    return results


def main():
    mask_gen = load_sam()
    clip_model, clip_preprocess, clip_text_emb = load_clip()
    views = sorted(VIEWS_DIR.glob("view_*.png"))
    for v in tqdm(views, desc="segment"):
        res = segment_view(v, mask_gen, clip_model, clip_preprocess, clip_text_emb)
        out = OUT_DIR / (v.stem + ".json")
        out.write_text(json.dumps(res))
    print(f"segmented {len(views)} views")


if __name__ == "__main__":
    main()
