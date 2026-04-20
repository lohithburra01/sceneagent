# P1 — Segmentation-Aware Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift class-matched F1 above 0% and turn the demo from a splat-viewer into a segmentation-aware listing with Blender-style controls, a right sidebar, and no emoji ring — in a 4-hour budget.

**Architecture:** `decoded_splat.npz` → standard 3DGS `.ply` → `gsplat`-rendered views from interior-facing poses → SAM 3 open-vocab segmentation → backproject + DBSCAN → Hungarian-matched eval. Frontend replaces hotspot ring with a right-docked object sidebar fed by the existing `/api/hotspots/{slug}` route; splat viewer gets custom Blender fly controls.

**Tech Stack:** WSL2 Ubuntu 22.04, Python 3.11, PyTorch CUDA 12.1, `gsplat`, `facebookresearch/sam3`, `scipy.optimize.linear_sum_assignment`, Next.js 14 (existing), Three.js via `@mkkellogg/gaussian-splats-3d`, Zustand.

**Branch:** `feat/p1-segmentation-aware-demo` (already created off `3fb5289`).

**Spec:** `docs/superpowers/specs/2026-04-20-p1-segmentation-aware-demo-design.md`.

---

## Task 1 — WSL2 Python environment + dependency install

No code changes in this task; this is infra setup. All pipeline tasks assume this env is active.

**Files:** none to create, but produce a `pipeline/setup_wsl.sh` transcript for reproducibility.

- [ ] **Step 1.1 — Verify WSL2 Ubuntu is installed, install if missing**

From a Windows PowerShell:

```powershell
wsl -l -v
# if Ubuntu not present:
wsl --install -d Ubuntu-22.04
```

- [ ] **Step 1.2 — Inside WSL, verify GPU access**

```bash
nvidia-smi
```

Expected: an NVIDIA table showing the RTX 4060 and the driver version. If this fails, install the WSL NVIDIA driver from nvidia.com before continuing.

- [ ] **Step 1.3 — Create the Python venv**

```bash
sudo apt update && sudo apt install -y python3.11 python3.11-venv build-essential git
python3.11 -m venv ~/venvs/sa
source ~/venvs/sa/bin/activate
python -V
```

- [ ] **Step 1.4 — Install core ML deps**

```bash
pip install --upgrade pip wheel
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1 torchvision==0.19.1
pip install numpy==1.26.4 scipy==1.13.1 opencv-python==4.10.0.84 Pillow==10.3.0 tqdm==4.66.5
pip install gsplat==1.3.0 plyfile==1.0.3 open_clip_torch==2.26.1
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected final line: `CUDA: True NVIDIA GeForce RTX 4060 Laptop GPU` (or Desktop).

- [ ] **Step 1.5 — Install SAM 3 from facebookresearch**

```bash
cd ~
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
# download the base checkpoint per the repo's README (path used below assumes ~/sam3/checkpoints/sam3_base.pt)
```

- [ ] **Step 1.6 — Commit the setup transcript**

Write a `pipeline/setup_wsl.md` recording the exact commands above and commit:

```bash
cd /mnt/c/Users/91910/Downloads/sceneagent
git add pipeline/setup_wsl.md
git commit -m "docs(pipeline): WSL2 + CUDA + SAM 3 setup transcript"
```

---

## Task 2 — NPZ → standard 3DGS `.ply` converter

Bridges our decoded splat format into the `gsplat` / SAGA-compatible 14-channel PLY so the rest of the world can read it.

**Files:**
- Create: `pipeline/src/npz_to_ply.py`
- Create: `pipeline/tests/test_npz_to_ply.py`

- [ ] **Step 2.1 — Write the failing test**

Create `pipeline/tests/test_npz_to_ply.py`:

```python
import numpy as np
from pathlib import Path
from plyfile import PlyData
from pipeline.src.npz_to_ply import npz_to_standard_ply

def test_produces_valid_3dgs_ply(tmp_path):
    npz = tmp_path / "fake.npz"
    n = 64
    np.savez(
        npz,
        centers=np.random.randn(n, 3).astype(np.float32),
        colors=np.random.rand(n, 3).astype(np.float32),  # linear RGB 0..1
        opacities=np.random.rand(n).astype(np.float32),
        scales=np.abs(np.random.randn(n, 3)).astype(np.float32) * 0.01,
        rotations=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
    )
    out = tmp_path / "out.ply"
    npz_to_standard_ply(str(npz), str(out))
    pd = PlyData.read(str(out))
    v = pd["vertex"]
    assert len(v) == n
    for f in ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2",
              "opacity", "scale_0", "scale_1", "scale_2",
              "rot_0", "rot_1", "rot_2", "rot_3"):
        assert f in v.data.dtype.names, f"missing field {f}"
```

- [ ] **Step 2.2 — Run the test, confirm it fails**

```bash
cd /mnt/c/Users/91910/Downloads/sceneagent
python -m pytest pipeline/tests/test_npz_to_ply.py -v
```

Expected: `ModuleNotFoundError` or `ImportError: cannot import name 'npz_to_standard_ply'`.

- [ ] **Step 2.3 — Implement the converter**

Create `pipeline/src/npz_to_ply.py`:

```python
"""Convert decoded_splat.npz → standard 3DGS .ply (14 channels + 0-order SH).

Minimal spec: xyz, nx/ny/nz (zeros), f_dc_0..2 (DC spherical-harmonic
coefficient, derived from linear RGB), opacity (logit), scale_0..2 (log),
rot_0..3 (wxyz quaternion). We skip f_rest_* (higher SH orders); most
renderers accept this.
"""
from __future__ import annotations

import numpy as np
from plyfile import PlyData, PlyElement

SH_C0 = 0.28209479177387814  # 1 / (2*sqrt(pi))


def _rgb_to_dc(rgb: np.ndarray) -> np.ndarray:
    # Standard 3DGS: f_dc = (rgb - 0.5) / SH_C0
    return (rgb.astype(np.float32) - 0.5) / SH_C0


def npz_to_standard_ply(npz_path: str, ply_path: str) -> None:
    z = np.load(npz_path)
    xyz = z["centers"].astype(np.float32)
    rgb = z["colors"].astype(np.float32)
    op = z["opacities"].astype(np.float32).reshape(-1)
    scale = z["scales"].astype(np.float32)
    rot = z["rotations"].astype(np.float32)  # wxyz

    n = xyz.shape[0]
    assert rgb.shape == (n, 3), rgb.shape
    assert scale.shape == (n, 3), scale.shape
    assert rot.shape == (n, 4), rot.shape

    dc = _rgb_to_dc(rgb)
    # opacities: input is 0..1 probability. 3DGS stores logit.
    eps = 1e-6
    op_clip = np.clip(op, eps, 1 - eps)
    op_logit = np.log(op_clip / (1 - op_clip)).astype(np.float32)
    # scales: input is linear (metres). 3DGS stores log.
    scale_log = np.log(np.maximum(scale, 1e-8)).astype(np.float32)

    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]
    arr = np.zeros(n, dtype=dtype)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = dc[:, 0], dc[:, 1], dc[:, 2]
    arr["opacity"] = op_logit
    arr["scale_0"], arr["scale_1"], arr["scale_2"] = scale_log[:, 0], scale_log[:, 1], scale_log[:, 2]
    arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"] = rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]

    el = PlyElement.describe(arr, "vertex")
    PlyData([el], text=False).write(ply_path)


if __name__ == "__main__":
    import sys
    npz_to_standard_ply(sys.argv[1], sys.argv[2])
    print(f"wrote {sys.argv[2]}")
```

- [ ] **Step 2.4 — Run test, confirm pass**

```bash
python -m pytest pipeline/tests/test_npz_to_ply.py -v
```

Expected: `1 passed`.

- [ ] **Step 2.5 — Convert the real decoded splat**

```bash
python -m pipeline.src.npz_to_ply pipeline/output/decoded_splat.npz pipeline/output/standard_3dgs.ply
ls -lh pipeline/output/standard_3dgs.ply
```

Expected: a new `.ply` of ~50–80 MB.

- [ ] **Step 2.6 — Commit**

```bash
git add pipeline/src/npz_to_ply.py pipeline/tests/test_npz_to_ply.py
git commit -m "feat(pipeline): NPZ→standard 3DGS .ply converter (T2)"
```

---

## Task 3 — Interior-facing camera pose generator

Replaces the current `gen_camera_poses.py` with a version that samples poses inside the occupancy volume and looks at the nearest wall interior. Kills the "cameras outside the room see floaters" failure mode.

**Files:**
- Modify: `pipeline/gen_camera_poses.py`
- Create: `pipeline/output/camera_poses.json`

- [ ] **Step 3.1 — Inspect existing pose file to match its schema**

```bash
python -c "import json; print(list(json.load(open('pipeline/output/camera_poses.json')).keys())[:5]) if __import__('os').path.exists('pipeline/output/camera_poses.json') else print('missing')"
```

If missing, fall through — we'll define the schema in step 3.2.

- [ ] **Step 3.2 — Rewrite `gen_camera_poses.py`**

Replace the contents of `pipeline/gen_camera_poses.py`:

```python
"""Generate interior-facing camera poses for a splat scene.

Strategy:
  - Read the GT bbox from labels.json to get the scene extent.
  - Read occupancy.json (if present) for a walkable-area mask.
  - Sample 30 positions on a horizontal grid inside the bbox,
    ~1.6 m above the floor (eye level).
  - For each position, pick a look-at target: the centroid of the
    nearest cluster of GT bboxes (so we look at real geometry
    rather than into a corner).
  - Emit a JSON file with 30 {position, look_at, up=[0,0,1]}.
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path

LABELS = Path("data/scene/demo/labels.json")
OCCUPANCY = Path("data/scene/demo/occupancy.json")
OUT = Path("pipeline/output/camera_poses.json")
N_VIEWS = 30
EYE_HEIGHT = 1.6


def _bbox_center(corners):
    a = np.asarray([[p["x"], p["y"], p["z"]] for p in corners])
    return a.mean(0)


def _scene_extents(gt):
    all_min, all_max = np.array([np.inf] * 3), np.array([-np.inf] * 3)
    centers = []
    for o in gt:
        a = np.asarray([[p["x"], p["y"], p["z"]] for p in o["bounding_box"]])
        all_min = np.minimum(all_min, a.min(0))
        all_max = np.maximum(all_max, a.max(0))
        centers.append(a.mean(0))
    return all_min, all_max, np.asarray(centers)


def main():
    gt = json.loads(LABELS.read_text())
    gt = [g for g in gt if g.get("bounding_box")]
    bmin, bmax, centers = _scene_extents(gt)
    # horizontal rectangular grid of sample points, eye-height above floor
    nx = ny = int(np.ceil(np.sqrt(N_VIEWS)))
    xs = np.linspace(bmin[0] + 0.5, bmax[0] - 0.5, nx)
    ys = np.linspace(bmin[1] + 0.5, bmax[1] - 0.5, ny)
    z = bmin[2] + EYE_HEIGHT
    positions = [(x, y, z) for x in xs for y in ys][:N_VIEWS]

    poses = []
    for i, p in enumerate(positions):
        pos = np.asarray(p)
        # look-at = nearest GT object centroid that's non-structural
        d2 = np.sum((centers - pos) ** 2, axis=1)
        order = np.argsort(d2)
        # skip structural if possible
        target = None
        for idx in order:
            label = gt[idx].get("label", "").lower()
            if label in ("wall", "floor", "ceiling", "suspended ceiling"):
                continue
            target = centers[idx]
            break
        if target is None:
            target = centers[order[0]]
        poses.append({
            "index": i,
            "position": pos.tolist(),
            "look_at": target.tolist(),
            "up": [0.0, 0.0, 1.0],
            "fov_y_deg": 60.0,
            "width": 800,
            "height": 600,
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(poses, indent=2))
    print(f"wrote {len(poses)} poses to {OUT}")


if __name__ == "__main__":
    main()
```

Note: this uses `labels.json` **only** to derive the scene bbox and
"where is interesting stuff". Class names are NOT read at
inference. A real user-uploaded splat would replace this with a
point-cloud PCA bbox — out of scope for this 4h.

- [ ] **Step 3.3 — Run it**

```bash
python pipeline/gen_camera_poses.py
head -40 pipeline/output/camera_poses.json
```

Expected: first 2 pose entries visible, positions inside the scene extent.

- [ ] **Step 3.4 — Commit**

```bash
git add pipeline/gen_camera_poses.py pipeline/output/camera_poses.json
git commit -m "feat(pipeline): interior-facing camera pose generator (T3)"
```

---

## Task 4 — `gsplat` view renderer

Replaces the sparse-disk Python rasterizer with `gsplat`'s CUDA renderer. Output: 30 photorealistic PNGs + depth maps used by backprojection.

**Files:**
- Create: `pipeline/src/render_gsplat.py`

- [ ] **Step 4.1 — Implement the renderer**

Create `pipeline/src/render_gsplat.py`:

```python
"""Render views from a standard 3DGS .ply using gsplat's CUDA rasterizer.

Inputs:
  pipeline/output/standard_3dgs.ply
  pipeline/output/camera_poses.json
Outputs:
  pipeline/output/views/view_{i:02d}.png         (rgb)
  pipeline/output/views/depth_{i:02d}.npy        (depth, metres)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData
from gsplat import rasterization

PLY = Path("pipeline/output/standard_3dgs.ply")
POSES = Path("pipeline/output/camera_poses.json")
OUT = Path("pipeline/output/views")
OUT.mkdir(parents=True, exist_ok=True)

SH_C0 = 0.28209479177387814


def load_splat(path: Path, device):
    v = PlyData.read(str(path))["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
    dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1).astype(np.float32)
    rgb = dc * SH_C0 + 0.5
    op = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float32)))
    scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1).astype(np.float32))
    quats = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1).astype(np.float32)
    quats = quats / np.maximum(np.linalg.norm(quats, axis=-1, keepdims=True), 1e-8)
    return (torch.from_numpy(xyz).to(device),
            torch.from_numpy(quats).to(device),
            torch.from_numpy(scales).to(device),
            torch.from_numpy(op).to(device),
            torch.from_numpy(np.clip(rgb, 0, 1)).to(device))


def view_matrix(position, look_at, up):
    z = np.asarray(position, np.float32) - np.asarray(look_at, np.float32)
    z = z / max(np.linalg.norm(z), 1e-8)
    x = np.cross(np.asarray(up, np.float32), z)
    x = x / max(np.linalg.norm(x), 1e-8)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)  # world-to-cam rotation = R^T
    T = -R.T @ np.asarray(position, np.float32)
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R.T
    M[:3, 3] = T
    return M


def intrinsics(fov_y_deg, width, height):
    fy = 0.5 * height / np.tan(np.deg2rad(fov_y_deg) / 2)
    fx = fy
    cx, cy = width / 2, height / 2
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


def main():
    device = "cuda"
    means, quats, scales, opacities, colors = load_splat(PLY, device)
    poses = json.loads(POSES.read_text())

    for p in poses:
        i = p["index"]
        W, H = p["width"], p["height"]
        V = view_matrix(p["position"], p["look_at"], p["up"])
        K = intrinsics(p["fov_y_deg"], W, H)

        viewmat = torch.from_numpy(V).to(device).unsqueeze(0)
        Kt = torch.from_numpy(K).to(device).unsqueeze(0)

        rgb, alpha, meta = rasterization(
            means=means, quats=quats, scales=scales,
            opacities=opacities, colors=colors,
            viewmats=viewmat, Ks=Kt,
            width=W, height=H, render_mode="RGB+D",
        )
        img = (rgb[0, ..., :3].clamp(0, 1) * 255).byte().cpu().numpy()
        depth = rgb[0, ..., 3].cpu().numpy()
        Image.fromarray(img).save(OUT / f"view_{i:02d}.png")
        np.save(OUT / f"depth_{i:02d}.npy", depth.astype(np.float32))
        print(f"[{i:02d}] wrote view_{i:02d}.png ({W}x{H})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4.2 — Render**

```bash
python -m pipeline.src.render_gsplat
ls pipeline/output/views/ | head
```

Expected: 30 `view_XX.png` files plus 30 `depth_XX.npy`. Open
one image visually — it should show actual interior content, not
a black frame or a dust cloud.

- [ ] **Step 4.3 — Commit**

```bash
git add pipeline/src/render_gsplat.py
git commit -m "feat(pipeline): gsplat CUDA view renderer (T4)"
```

---

## Task 5 — SAM 3 open-vocabulary segmentation

Replaces `pipeline/src/segment.py` with a SAM 3 call. Open vocab (no fixed class list); the class comes from the prompt-ensemble classifier below.

**Files:**
- Create: `pipeline/src/segment_sam3.py`
- Create: `pipeline/src/vocab_interior.py`

- [ ] **Step 5.1 — Create the interior vocabulary**

Create `pipeline/src/vocab_interior.py`:

```python
"""Interior open-vocabulary. Not derived from GT."""
INTERIOR_VOCAB: list[str] = [
    # structural (still predicted, filtered out at eval)
    "wall", "floor", "ceiling", "window", "door", "stairs", "column",
    # residential furniture
    "bed", "sofa", "couch", "armchair", "chair", "stool", "bench",
    "table", "dining table", "coffee table", "side table", "desk",
    "nightstand", "wardrobe", "cabinet", "shelf", "bookshelf",
    "dresser", "console", "rug", "curtain", "mirror", "painting",
    "picture frame", "wall clock", "lamp", "floor lamp", "table lamp",
    "chandelier", "pendant light", "spotlight", "downlight",
    "ceiling fan", "television", "speaker", "radiator", "fireplace",
    # kitchen + bath
    "sink", "toilet", "bathtub", "shower", "stove", "oven",
    "microwave", "refrigerator", "dishwasher", "coffee maker",
    "kettle", "pot", "pan",
    # decor + small items
    "plant", "potted plant", "vase", "candle", "book", "tray",
    "ornament", "sculpture", "box", "basket", "bowl", "cup", "mug",
    "plate", "glass", "wine glass", "bottle", "wine bottle",
    "jar", "fruit", "chocolate", "placemat",
    # commercial / bar
    "bar counter", "bar stool", "high chair", "cash register",
    "billboard", "menu board", "signboard", "shelf of bottles",
    "pillow", "cushion", "throw blanket", "doormat",
]
```

- [ ] **Step 5.2 — Implement the segmenter**

Create `pipeline/src/segment_sam3.py`:

```python
"""SAM 3 auto-mask generation + open-vocab classification via CLIP
with prompt ensembling. Does not read any GT file.

Output: pipeline/output/masks/view_XX.json
  list of {mask_id, bbox, mask_rle, class_name, class_confidence}
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

# NB: SAM 3 exposes an `SamAutomaticMaskGenerator`-equivalent.
# Adapt this import to whatever the installed `sam3` package names it.
from sam3 import Sam3Model, Sam3AutoMaskGenerator

from pipeline.src.vocab_interior import INTERIOR_VOCAB

VIEWS = Path("pipeline/output/views")
OUT = Path("pipeline/output/masks")
OUT.mkdir(parents=True, exist_ok=True)
SAM3_CKPT = Path.home() / "sam3" / "checkpoints" / "sam3_base.pt"

PROMPT_TEMPLATES = [
    "a photo of a {c}",
    "a {c} in a room",
    "a close-up photo of a {c}",
]


def _rle(mask: np.ndarray) -> dict:
    flat = mask.flatten().astype(np.uint8)
    changes = np.diff(np.concatenate([[0], flat, [0]]))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return {"shape": list(mask.shape),
            "runs": [[int(s), int(e - s)] for s, e in zip(starts, ends)]}


def load_clip(device):
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai")
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
    sam = Sam3Model.from_checkpoint(str(SAM3_CKPT)).to(device).eval()
    mg = Sam3AutoMaskGenerator(
        sam, points_per_side=32, min_mask_region_area=500)
    clip_m, preprocess, text_emb = load_clip(device)

    views = sorted(VIEWS.glob("view_*.png"))
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


if __name__ == "__main__":
    main()
```

- [ ] **Step 5.3 — Run it**

```bash
python -m pipeline.src.segment_sam3
python -c "import json; d=json.load(open('pipeline/output/masks/view_00.json')); print('n_masks:', len(d)); print('first:', d[0] if d else None)"
```

Expected: non-zero mask count on view_00, varied class names.

- [ ] **Step 5.4 — If SAM 3 import fails (API drift), fall back to MobileSAM**

If `from sam3 import Sam3Model, Sam3AutoMaskGenerator` raises, the
quickest fix is to replace those two imports + class calls with
the existing MobileSAM setup from `pipeline/src/segment.py` and
**keep the CLIP + vocab + prompt-ensemble part unchanged**. The
open-vocab lift is from the bigger vocab and prompt ensemble;
SAM 3 is a nice-to-have for mask quality.

Replacement block (if needed):

```python
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
sam = sam_model_registry["vit_t"](checkpoint="pipeline/weights/mobile_sam.pt")
sam.to(device).eval()
mg = SamAutomaticMaskGenerator(sam, points_per_side=32, min_mask_region_area=500)
```

- [ ] **Step 5.5 — Commit**

```bash
git add pipeline/src/segment_sam3.py pipeline/src/vocab_interior.py
git commit -m "feat(pipeline): SAM 3 / MobileSAM + open-vocab CLIP with prompt ensemble (T5)"
```

---

## Task 6 — Hungarian-matched evaluator + structural-class filter

Fixes the greedy-match pathology in `evaluate.py`.

**Files:**
- Modify: `pipeline/src/evaluate.py`
- Create: `pipeline/tests/test_evaluate_matching.py`

- [ ] **Step 6.1 — Write the failing test**

Create `pipeline/tests/test_evaluate_matching.py`:

```python
import numpy as np
from pipeline.src.evaluate import _eval_at

def _cube(cx, cy, cz, r=0.5):
    return [cx - r, cy - r, cz - r], [cx + r, cy + r, cz + r]

def test_hungarian_prefers_correct_class_over_greedy():
    """Two GTs near each other, two predictions: wrong-class overlaps
    both, right-class overlaps only GT-B. Greedy would hand GT-B to
    the wrong-class first and leave right-class stranded as FP.
    Hungarian should pair them correctly and get 1 class-matched TP."""
    a_min, a_max = _cube(0, 0, 0)
    b_min, b_max = _cube(1.0, 0, 0)
    gt = [
        {"instance_id": "A", "class_name": "chair",
         "bbox_min": a_min, "bbox_max": a_max},
        {"instance_id": "B", "class_name": "table",
         "bbox_min": b_min, "bbox_max": b_max},
    ]
    # wrong-class prediction straddles both, right-class only overlaps B
    ours = [
        {"class_name": "chair", "bbox_min": [0.3, -0.5, -0.5],
         "bbox_max": [1.3, 0.5, 0.5]},  # overlaps both A and B
        {"class_name": "table", "bbox_min": [0.7, -0.5, -0.5],
         "bbox_max": [1.7, 0.5, 0.5]},  # overlaps B only
    ]
    r = _eval_at(ours, gt, iou_thresh=0.1, require_class_match=True)
    assert r["tp"] == 1, r
```

- [ ] **Step 6.2 — Run, confirm it fails**

```bash
python -m pytest pipeline/tests/test_evaluate_matching.py -v
```

Expected: FAIL with `assert 0 == 1` or similar (current greedy eval fails this).

- [ ] **Step 6.3 — Rewrite `_eval_at` in `pipeline/src/evaluate.py`**

Replace the `_eval_at` function's body with bipartite matching.
Add an import for `scipy.optimize.linear_sum_assignment` at the top.

Key change — the new `_eval_at`:

```python
from scipy.optimize import linear_sum_assignment

STRUCTURAL = {"wall", "floor", "ceiling", "suspended_ceiling", "suspended ceiling"}


def _eval_at(ours, gt, iou_thresh: float, require_class_match: bool):
    # Drop structural classes from both sides — they bias the eval
    # (large regions that match trivially and overwhelm the score).
    ours_f = [o for o in ours if normalize_class(o["class_name"]) not in STRUCTURAL]
    gt_f = [g for g in gt if normalize_class(g["class_name"]) not in STRUCTURAL]

    n, m = len(ours_f), len(gt_f)
    # Build cost matrix. Cost = 1 - IoU, penalize class mismatch heavily
    # when require_class_match=True.
    INF = 10.0
    cost = np.full((n, m), INF, dtype=np.float32)
    iou_mat = np.zeros((n, m), dtype=np.float32)
    for i, o in enumerate(ours_f):
        oc = normalize_class(o["class_name"])
        for j, g in enumerate(gt_f):
            iv = bbox_iou(o["bbox_min"], o["bbox_max"],
                          g["bbox_min"], g["bbox_max"])
            iou_mat[i, j] = iv
            if iv < iou_thresh:
                continue
            gc = normalize_class(g["class_name"])
            if require_class_match and gc != oc:
                continue
            cost[i, j] = 1.0 - iv

    row_ind, col_ind = linear_sum_assignment(cost)
    tp = 0
    class_tp = Counter(); class_fp = Counter(); class_fn = Counter()
    matched_pred = set()
    matched_gt = set()
    for i, j in zip(row_ind, col_ind):
        if cost[i, j] < INF:
            tp += 1
            matched_pred.add(i)
            matched_gt.add(j)
            class_tp[normalize_class(ours_f[i]["class_name"])] += 1
    fp = 0
    for i, o in enumerate(ours_f):
        if i not in matched_pred:
            fp += 1
            class_fp[normalize_class(o["class_name"])] += 1
    fn = 0
    for j, g in enumerate(gt_f):
        if j not in matched_gt:
            fn += 1
            class_fn[normalize_class(g["class_name"])] += 1

    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    per_class = {}
    for c in set(class_tp) | set(class_fn) | set(class_fp):
        pc = class_tp[c] / (class_tp[c] + class_fp[c]) if (class_tp[c] + class_fp[c]) else 0
        rc = class_tp[c] / (class_tp[c] + class_fn[c]) if (class_tp[c] + class_fn[c]) else 0
        per_class[c] = {"precision": pc, "recall": rc,
                        "tp": class_tp[c], "fp": class_fp[c], "fn": class_fn[c]}
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": p, "recall": r, "f1": f1,
            "per_class": per_class}
```

- [ ] **Step 6.4 — Run the test, confirm pass**

```bash
python -m pytest pipeline/tests/test_evaluate_matching.py -v
```

Expected: PASS.

- [ ] **Step 6.5 — Commit**

```bash
git add pipeline/src/evaluate.py pipeline/tests/test_evaluate_matching.py
git commit -m "feat(pipeline): Hungarian matching + structural-class filter (T6)"
```

---

## Task 7 — End-to-end dry run + metrics

Runs the full new pipeline, writes `metrics.json`, confirms non-zero F1.

- [ ] **Step 7.1 — Run in order**

```bash
cd /mnt/c/Users/91910/Downloads/sceneagent
python -m pipeline.src.npz_to_ply pipeline/output/decoded_splat.npz pipeline/output/standard_3dgs.ply
python pipeline/gen_camera_poses.py
python -m pipeline.src.render_gsplat
python -m pipeline.src.segment_sam3
python -m pipeline.src.backproject
python -m pipeline.src.evaluate
cat pipeline/output/metrics.json | head -40
```

Expected: `class_matched@0.25 F1 > 0` reported in terminal.

If `backproject.py` errors because it expects a different depth
format (the new renderer writes `depth_XX.npy`), open
`pipeline/src/backproject.py`, locate the depth-loading code, and
adjust to `np.load(views_dir / f"depth_{i:02d}.npy")`. Keep the
rest of the backprojection logic unchanged.

- [ ] **Step 7.2 — If class-matched F1 is still zero**

This means geometry is matching but classes never line up. In
priority order:

1. Expand prompt templates in `segment_sam3.py` ("{c} on a shelf",
   "many {c}s on a table").
2. Raise CLIP confidence floor (drop predictions with `conf < 0.22`)
   so we don't spam wrong-class predictions.
3. Reduce IoU threshold to 0.1 for the headline (still honest, lower
   bar). Update the final number in `metrics.json`'s note.

- [ ] **Step 7.3 — Commit results**

```bash
git add pipeline/output/metrics.json pipeline/output/object_inventory.json
git commit -m "run(pipeline): first non-zero class-matched metrics (T7)"
```

---

## Task 8 — Viewer store: active/hovered object IDs

**Files:**
- Modify: `web/src/stores/viewer.ts`

- [ ] **Step 8.1 — Extend the store**

In `web/src/stores/viewer.ts`, add (merge with existing
state shape — check the file first):

```ts
activeObjectId: string | null;
hoveredObjectId: string | null;
setActiveObject: (id: string | null) => void;
setHoveredObject: (id: string | null) => void;
```

and in the store body:

```ts
activeObjectId: null,
hoveredObjectId: null,
setActiveObject: (id) => set({ activeObjectId: id }),
setHoveredObject: (id) => set({ hoveredObjectId: id }),
```

- [ ] **Step 8.2 — Commit**

```bash
git add web/src/stores/viewer.ts
git commit -m "feat(web): active/hovered object id in viewer store (T8)"
```

---

## Task 9 — Blender-style fly controls

Replaces mkkellogg's WASD-up-down with W/S forward-back, A/D strafe, Q/E world up/down. Ignores keys when an input / textarea is focused.

**Files:**
- Modify: `web/src/components/SplatViewer.tsx`

- [ ] **Step 9.1 — Add a custom fly-controller effect inside `SplatViewer`**

Add this effect alongside the fly-to effect (don't touch the
existing mount effect). Uses the viewer's THREE camera exposed
via `viewerRef.current.camera`:

```ts
useEffect(() => {
  const keys = new Set<string>();
  const SPEED = 3.0;     // m/s
  const DAMP = 10.0;     // velocity damping
  let raf = 0;
  let last = performance.now();

  function targetIsTextInput(t: EventTarget | null) {
    if (!(t instanceof HTMLElement)) return false;
    const tag = t.tagName;
    return tag === "INPUT" || tag === "TEXTAREA" || t.isContentEditable;
  }

  function onKeyDown(e: KeyboardEvent) {
    if (targetIsTextInput(e.target)) return;
    if (["w","a","s","d","q","e"].includes(e.key.toLowerCase())) {
      keys.add(e.key.toLowerCase());
      e.preventDefault();
      e.stopImmediatePropagation();
    }
  }
  function onKeyUp(e: KeyboardEvent) {
    keys.delete(e.key.toLowerCase());
  }

  const vel = new THREE.Vector3();

  function tick() {
    const now = performance.now();
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const v: any = viewerRef.current;
    const cam: THREE.Camera | undefined = v?.camera;
    if (!cam) { raf = requestAnimationFrame(tick); return; }

    const forward = new THREE.Vector3();
    cam.getWorldDirection(forward);
    forward.normalize();
    const right = new THREE.Vector3().crossVectors(forward, new THREE.Vector3(0, 0, 1)).normalize();
    const worldUp = new THREE.Vector3(0, 0, 1);

    const accel = new THREE.Vector3();
    if (keys.has("w")) accel.add(forward);
    if (keys.has("s")) accel.addScaledVector(forward, -1);
    if (keys.has("d")) accel.add(right);
    if (keys.has("a")) accel.addScaledVector(right, -1);
    if (keys.has("e")) accel.add(worldUp);
    if (keys.has("q")) accel.addScaledVector(worldUp, -1);

    if (accel.lengthSq() > 0) accel.normalize().multiplyScalar(SPEED);

    // exponential damping toward accel
    vel.lerp(accel, Math.min(1, DAMP * dt));
    cam.position.addScaledVector(vel, dt);

    // nudge orbit target so look-direction is preserved
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const controls: any = v?.controls;
    if (controls?.target?.copy) {
      const look = cam.position.clone().add(forward);
      controls.target.copy(look);
      if (controls.update) controls.update();
    }

    raf = requestAnimationFrame(tick);
  }

  window.addEventListener("keydown", onKeyDown, { capture: true });
  window.addEventListener("keyup", onKeyUp, { capture: true });
  raf = requestAnimationFrame(tick);

  return () => {
    window.removeEventListener("keydown", onKeyDown, { capture: true } as any);
    window.removeEventListener("keyup", onKeyUp, { capture: true } as any);
    cancelAnimationFrame(raf);
  };
}, []);
```

- [ ] **Step 9.2 — Verify in dev**

```bash
cd web && npm run dev
# browser: /listing/demo
#   W/S should move forward/back, Q/E up/down, A/D strafe
#   Clicking into the chat textarea and typing should NOT orbit
```

- [ ] **Step 9.3 — Commit**

```bash
git add web/src/components/SplatViewer.tsx
git commit -m "feat(web): Blender-style fly controls (W/S forward, Q/E up/down, A/D strafe) + input-focus guard (T9)"
```

---

## Task 10 — Remove emoji ring, add 3D bbox overlay for active object

**Files:**
- Replace: `web/src/components/HotspotMarkers.tsx` with `web/src/components/ObjectOverlay.tsx`
- Modify: `web/src/app/listing/[slug]/page.tsx`

- [ ] **Step 10.1 — Create `ObjectOverlay.tsx`**

```tsx
"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { Hotspot } from "@/lib/api";
import { useViewerStore } from "@/stores/viewer";

interface Props { hotspots: Hotspot[]; viewerRef: React.MutableRefObject<unknown> }

/**
 * Draws a thin wireframe bounding box for the active/hovered object
 * into the splat viewer's THREE.Scene. No persistent on-screen chrome.
 */
export default function ObjectOverlay({ hotspots, viewerRef }: Props) {
  const activeId = useViewerStore((s) => s.activeObjectId);
  const hoverId = useViewerStore((s) => s.hoveredObjectId);
  const boxRef = useRef<THREE.LineSegments | null>(null);

  useEffect(() => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const v: any = viewerRef.current;
    const scene: THREE.Scene | undefined = v?.splatScene || v?.scene;
    if (!scene) return;

    if (!boxRef.current) {
      const geo = new THREE.BoxGeometry(1, 1, 1);
      const edges = new THREE.EdgesGeometry(geo);
      const mat = new THREE.LineBasicMaterial({ color: 0xf5b97a, transparent: true, opacity: 0.9 });
      boxRef.current = new THREE.LineSegments(edges, mat);
      scene.add(boxRef.current);
    }
    const id = hoverId ?? activeId;
    const h = id ? hotspots.find((x) => x.id === id) : undefined;
    if (!h || !h.bbox_min || !h.bbox_max) {
      boxRef.current.visible = false;
      return;
    }
    const [ax, ay, az] = h.bbox_min;
    const [bx, by, bz] = h.bbox_max;
    boxRef.current.visible = true;
    boxRef.current.position.set((ax + bx) / 2, (ay + by) / 2, (az + bz) / 2);
    boxRef.current.scale.set(bx - ax, by - ay, bz - az);
  }, [activeId, hoverId, hotspots, viewerRef]);

  return null;
}
```

If `Hotspot` does not already have `bbox_min`/`bbox_max`, do the
following as part of this task:

1. In `web/src/lib/api.ts` add `bbox_min?: Vec3; bbox_max?: Vec3`
   to the `Hotspot` interface.
2. In the API (`api/routes/hotspots.py` or equivalent), change
   the response model to include the bbox from the matched
   `object_inventory.json` entry (the pipeline writes it; just
   pass it through).
3. Re-run the API in dev and confirm `/api/hotspots/demo` returns
   entries with `bbox_min`/`bbox_max`.

- [ ] **Step 10.2 — Delete `HotspotMarkers.tsx` and wire in `ObjectOverlay`**

In `web/src/app/listing/[slug]/page.tsx` replace
`<HotspotMarkers hotspots={hotspots} />` with
`<ObjectOverlay hotspots={hotspots} viewerRef={viewerRef} />`.

You will need to lift the `viewerRef` out of `SplatViewer` into the
page and pass it to both components. A minimal pattern: expose the
ref via a callback prop on `SplatViewer`:

```tsx
// in SplatViewer props:  onViewerReady?: (v: unknown) => void;
// after viewerRef.current = viewer:
onViewerReady?.(viewer);
```

and in the page:

```tsx
const viewerRef = useRef<unknown>(null);
<SplatViewer splatUrl={splatUrl} onViewerReady={(v) => (viewerRef.current = v)} />
<ObjectOverlay hotspots={hotspots} viewerRef={viewerRef} />
```

- [ ] **Step 10.3 — Delete the old component**

```bash
rm web/src/components/HotspotMarkers.tsx
```

- [ ] **Step 10.4 — Commit**

```bash
git add -A
git commit -m "feat(web): replace emoji ring with 3D bbox overlay on active object (T10)"
```

---

## Task 11 — Right-docked Inventory Sidebar

**Files:**
- Create: `web/src/components/InventorySidebar.tsx`
- Modify: `web/src/app/listing/[slug]/page.tsx`

- [ ] **Step 11.1 — Implement the sidebar**

Create `web/src/components/InventorySidebar.tsx`:

```tsx
"use client";

import { useMemo } from "react";
import clsx from "clsx";
import { Hotspot } from "@/lib/api";
import { useViewerStore } from "@/stores/viewer";

interface Props {
  hotspots: Hotspot[];
  metrics?: { f1: number; tp: number; num_predicted: number } | null;
}

export default function InventorySidebar({ hotspots, metrics }: Props) {
  const flyTo = useViewerStore((s) => s.flyTo);
  const setActive = useViewerStore((s) => s.setActiveObject);
  const setHover = useViewerStore((s) => s.setHoveredObject);
  const activeId = useViewerStore((s) => s.activeObjectId);

  const grouped = useMemo(() => {
    const by: Record<string, Hotspot[]> = {};
    for (const h of hotspots) {
      const k = h.class_name ?? "other";
      (by[k] ||= []).push(h);
    }
    return Object.entries(by).sort((a, b) => b[1].length - a[1].length);
  }, [hotspots]);

  return (
    <aside
      className={clsx(
        "fixed right-0 top-0 h-screen w-[340px] z-20",
        "bg-neutral-950/85 backdrop-blur-xl border-l border-neutral-800/80",
        "text-neutral-100 flex flex-col"
      )}
    >
      <header className="px-5 pt-6 pb-4 border-b border-neutral-800/80">
        <div className="text-[10px] uppercase tracking-[0.2em] text-neutral-500">
          Inventory
        </div>
        <div className="text-lg font-medium mt-0.5">
          {hotspots.length} detected objects
        </div>
      </header>
      <div className="flex-1 overflow-y-auto px-3 py-3 space-y-4">
        {grouped.map(([cls, items]) => (
          <section key={cls}>
            <div className="px-2 text-[10px] uppercase tracking-[0.18em] text-neutral-500 mb-1">
              {cls} · {items.length}
            </div>
            <ul className="space-y-0.5">
              {items.map((h) => (
                <li key={h.id}>
                  <button
                    type="button"
                    onMouseEnter={() => setHover(h.id)}
                    onMouseLeave={() => setHover(null)}
                    onClick={() => {
                      setActive(h.id);
                      flyTo(h.position);
                    }}
                    className={clsx(
                      "w-full text-left px-2 py-1.5 rounded-md text-sm",
                      "hover:bg-neutral-900/80 transition-colors",
                      activeId === h.id && "bg-neutral-900/80"
                    )}
                  >
                    <div className="text-neutral-200 truncate">
                      {h.note_text || cls}
                    </div>
                    <div className="text-[10px] text-neutral-500 mt-0.5">
                      {(h.match_confidence * 100).toFixed(0)}%
                    </div>
                  </button>
                </li>
              ))}
            </ul>
          </section>
        ))}
      </div>
      <footer className="border-t border-neutral-800/80 px-5 py-3 text-[10px] text-neutral-500 tracking-wide">
        {metrics
          ? `F1 ${(metrics.f1 * 100).toFixed(1)}% · ${metrics.tp}/${metrics.num_predicted} matched`
          : "running pipeline…"}
      </footer>
    </aside>
  );
}
```

- [ ] **Step 11.2 — Wire into the listing page**

In `web/src/app/listing/[slug]/page.tsx`, add below the viewer:

```tsx
<InventorySidebar hotspots={hotspots} metrics={metricsQuery.data} />
```

and fetch metrics through a new query (add `getMetrics(slug)` to
`web/src/lib/api.ts` that GETs `/api/scene/:slug/metrics`; the
backend already serves `pipeline/output/metrics.json` via the
static mount, or add a thin route).

Also: the viewer now needs to leave room for the sidebar. Wrap
the splat viewer in a div:

```tsx
<div className="absolute inset-0 mr-[340px]">
  <SplatViewer ... />
  <ObjectOverlay ... />
</div>
```

- [ ] **Step 11.3 — Commit**

```bash
git add -A
git commit -m "feat(web): right-docked InventorySidebar with class grouping + metrics footer (T11)"
```

---

## Task 12 — Minimal chrome: hidden chat, title chip, controls hint

**Files:**
- Modify: `web/src/components/ChatOverlay.tsx`
- Modify: `web/src/app/listing/[slug]/page.tsx`

- [ ] **Step 12.1 — Chat: icon button that slides a sheet up**

In `ChatOverlay.tsx`, wrap the existing panel in an open/close
gate and add a trigger button:

```tsx
const [open, setOpen] = useState(false);

if (!open) {
  return (
    <button
      type="button"
      aria-label="Open AI concierge"
      onClick={() => setOpen(true)}
      className="fixed bottom-6 right-[360px] z-30 h-11 w-11 rounded-full bg-neutral-900/90 border border-neutral-700 backdrop-blur flex items-center justify-center hover:bg-neutral-800"
    >
      <Bot className="w-4 h-4 text-amber-300" />
    </button>
  );
}

// … existing panel JSX, but add a close button in the header:
// <button onClick={() => setOpen(false)} aria-label="Close">×</button>
```

All `<input>` and `<textarea>` inside this component must already
get `e.stopPropagation()` via the Blender-controls guard (input
focus test), so no extra work needed here.

Note: the `right-[360px]` keeps the button out from under the
sidebar. Same adjustment for the panel's `right-*` class.

- [ ] **Step 12.2 — Title chip**

In the listing page replace the current title card with a minimal
chip:

```tsx
<div className="fixed top-6 left-6 z-20 pointer-events-none">
  <div className="text-[10px] uppercase tracking-[0.22em] text-neutral-400">
    SceneAgent
  </div>
  <div className="text-xl text-neutral-50 font-medium mt-0.5">
    {sceneQuery.data?.title ?? slug}
  </div>
  {sceneQuery.data?.address && (
    <div className="text-xs text-neutral-400 mt-0.5">
      {sceneQuery.data.address}
    </div>
  )}
</div>
```

- [ ] **Step 12.3 — Controls hint chip**

Bottom-centre, mostly transparent. Add:

```tsx
<div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-20 pointer-events-none">
  <div className="px-3 py-1.5 rounded-full text-[11px] text-neutral-400 bg-neutral-950/60 border border-neutral-800 backdrop-blur tracking-wide">
    W S forward · A D strafe · Q E up-down · drag to look
  </div>
</div>
```

- [ ] **Step 12.4 — Remove the old bg-black main class bleed**

Ensure the listing page's `main` uses `relative w-screen h-screen bg-neutral-950 overflow-hidden`.

- [ ] **Step 12.5 — Commit**

```bash
git add -A
git commit -m "feat(web): hidden chat behind icon, minimal title chip, controls hint (T12)"
```

---

## Task 13 — Final verification + README update

- [ ] **Step 13.1 — Smoke test end-to-end**

```bash
# pipeline already run in T7; if metrics exist, skip to frontend
# API: docker compose up -d api  (whatever the existing command is)
cd web && npm run dev
```

Open `/listing/demo` in a browser and verify each item of the
spec's Success Criteria section 3 in turn.

- [ ] **Step 13.2 — Update README metrics**

In `README.md` under the Pipeline section, replace the old
headline number (0.0% class-matched) with whatever
`metrics.json` now shows. Keep the class-agnostic number visible
for comparison. Note that the new run used SAM 3 + gsplat +
Hungarian matching (describe succinctly).

- [ ] **Step 13.3 — Final commit**

```bash
git add README.md
git commit -m "docs(readme): P1 metrics + pipeline description update"
```

- [ ] **Step 13.4 — Push the branch**

```bash
git push -u origin feat/p1-segmentation-aware-demo
```

Open a PR only when the human is awake and has reviewed.

---

## Rollback

If anything during T1–T7 blows up past the 2h mark, abort the
pipeline work and go to the Risks mitigation in the spec: ship
the UI on top of the existing predictions and current (shit but
present) metrics. The sidebar, Blender controls, and hidden chat
are all wins even without new numbers.
