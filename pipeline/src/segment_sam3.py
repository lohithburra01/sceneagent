"""SAM 3 text-promptable open-vocab segmentation.

For each rendered view, prompts SAM 3 with each interior vocab class
("chair", "table", ...) and saves the returned instance masks.
This is more direct than the older auto-mask + CLIP-classify pattern:
the text prompt IS the class label, and SAM 3 was trained for exactly
this setup with a presence head that suppresses absent concepts.

Falls back to MobileSAM + CLIP only if SAM 3 is not installed.

Output:
  pipeline/output/masks/view_XX.json
  list[{mask_id, bbox[x,y,w,h], mask_rle, class_name, class_confidence,
        stability_score}]
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from pipeline.src.vocab_interior import INTERIOR_VOCAB

VIEWS = Path("pipeline/output/views")
OUT = Path("pipeline/output/masks")
OUT.mkdir(parents=True, exist_ok=True)


def _rle(mask: np.ndarray) -> dict:
    flat = mask.flatten().astype(np.uint8)
    changes = np.diff(np.concatenate([[0], flat, [0]]))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return {
        "shape": list(mask.shape),
        "runs": [[int(s), int(e - s)] for s, e in zip(starts, ends)],
    }


def _try_sam3():
    """Return (processor, backend_name) or (None, None) if SAM 3 unavailable."""
    try:
        import sam3
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except Exception as exc:  # noqa: BLE001
        print(f"[segment] SAM 3 import failed ({exc.__class__.__name__}: {exc})")
        return None, None
    bpe_path = os.path.join(
        os.path.dirname(sam3.__file__), "..", "assets",
        "bpe_simple_vocab_16e6.txt.gz",
    )
    print("[segment] loading SAM 3 image model (first run downloads ~1.5 GB) ...")
    t0 = time.time()
    model = build_sam3_image_model(bpe_path=bpe_path)
    proc = Sam3Processor(model, confidence_threshold=0.30)
    print(f"[segment] SAM 3 ready in {time.time() - t0:.1f}s")
    return proc, "sam3"


def _segment_with_sam3(processor, vocab: list[str]):
    views = sorted(VIEWS.glob("view_*.png"))
    print(f"[segment] {len(views)} views × {len(vocab)} classes = {len(views) * len(vocab)} prompts")

    for v in tqdm(views, desc="views"):
        img = Image.open(v).convert("RGB")
        H, W = img.height, img.width
        state = processor.set_image(img)

        out: list[dict] = []
        mask_id = 0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for cls in vocab:
                processor.reset_all_prompts(state)
                state = processor.set_text_prompt(prompt=cls, state=state)
                masks = state.get("masks")
                boxes = state.get("boxes")
                scores = state.get("scores")
                if masks is None or len(masks) == 0:
                    continue
                masks_np = masks.squeeze(1).cpu().numpy().astype(bool)
                boxes_np = boxes.cpu().numpy()
                scores_np = scores.cpu().numpy()
                for k in range(masks_np.shape[0]):
                    seg = masks_np[k]
                    if seg.sum() < 400:
                        continue
                    x0, y0, x1, y1 = [float(b) for b in boxes_np[k]]
                    bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
                    if bw < 20 or bh < 20:
                        continue
                    out.append({
                        "mask_id": mask_id,
                        "bbox": [int(x0), int(y0), int(bw), int(bh)],
                        "mask_rle": _rle(seg),
                        "class_name": cls,
                        "class_confidence": float(scores_np[k]),
                        "stability_score": float(scores_np[k]),
                    })
                    mask_id += 1
        (OUT / (v.stem + ".json")).write_text(json.dumps(out))
        # tqdm shows progress; one summary per view to stderr
        print(f"  {v.stem}: {len(out)} kept masks", flush=True)

    print(f"[segment] wrote {len(views)} mask files to {OUT}")


def _segment_with_mobile_sam(vocab: list[str]):
    """Fallback: MobileSAM auto-masks + CLIP open-vocab classification."""
    import cv2
    import open_clip
    from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = Path("pipeline/weights/mobile_sam.pt")
    if not ckpt.exists():
        ckpt = Path("/mnt/c/Users/91910/Downloads/sceneagent/pipeline/weights/mobile_sam.pt")
    sam = sam_model_registry["vit_t"](checkpoint=str(ckpt)).to(device).eval()
    mg = SamAutomaticMaskGenerator(sam, points_per_side=32, min_mask_region_area=500)

    clip_m, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    clip_m.to(device).eval()

    PROMPTS = ["a photo of a {c}", "a {c} in a room", "a close-up photo of a {c}"]
    with torch.no_grad():
        embs = []
        for c in vocab:
            toks = tokenizer([t.format(c=c) for t in PROMPTS]).to(device)
            e = clip_m.encode_text(toks)
            e = e / e.norm(dim=-1, keepdim=True)
            embs.append(e.mean(0, keepdim=True))
        text_emb = torch.cat(embs, dim=0)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    views = sorted(VIEWS.glob("view_*.png"))
    for v in tqdm(views, desc="views"):
        img = cv2.cvtColor(cv2.imread(str(v)), cv2.COLOR_BGR2RGB)
        masks = mg.generate(img)
        out = []
        for i, m in enumerate(masks):
            seg = m["segmentation"]
            x, y, w, h = [int(a) for a in m["bbox"]]
            if w < 20 or h < 20:
                continue
            mask_crop = seg[y:y + h, x:x + w]
            crop = img[y:y + h, x:x + w] * mask_crop[..., None]
            t = preprocess(Image.fromarray(crop.astype(np.uint8))).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = clip_m.encode_image(t)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                conf, idx = (emb @ text_emb.T).squeeze(0).max(dim=0)
            out.append({
                "mask_id": i, "bbox": [x, y, w, h], "mask_rle": _rle(seg),
                "class_name": vocab[int(idx)], "class_confidence": float(conf),
                "stability_score": float(m.get("stability_score", 0)),
            })
        (OUT / (v.stem + ".json")).write_text(json.dumps(out))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[segment] device={device}")
    proc, backend = _try_sam3()
    if proc is not None:
        _segment_with_sam3(proc, INTERIOR_VOCAB)
    else:
        print("[segment] falling back to MobileSAM + CLIP")
        _segment_with_mobile_sam(INTERIOR_VOCAB)


if __name__ == "__main__":
    main()
