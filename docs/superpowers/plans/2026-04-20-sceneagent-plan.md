# SceneAgent — 6-Hour Parallel Execution Plan

**Superseded:** The earlier sequential plan. This plan executes in 4 parallel tracks over 6 hours, with Track C adding a **real 2D→3D semantic segmentation pipeline** (SAM + CLIP + multi-view backprojection, evaluated against InteriorGS ground truth).

## How to execute this plan

You will run **4 tracks in parallel** using Claude Code:

- **Main session** = Track M (orchestration + Track C work + final integration)
- **Subagent 1** = Track A (frontend)
- **Subagent 2** = Track B (backend API + agent)
- **Subagent 3** = Track D (infra + docs + demo prep) — dispatched at T+3:00

**Kickoff in the main Claude Code session (inside `sceneagent/` folder):**

> *"Read `docs/superpowers/plans/2026-04-20-sceneagent-plan.md`. Execute Track M-Preflight (§Preflight). Then dispatch two subagents in parallel — one for Track A (frontend), one for Track B (backend). While they work, execute Track M (the CV pipeline) yourself. Check in every 30 min. Dispatch Track D at T+3:00. At T+5:00 begin integration. Use the Agent tool with subagent_type general-purpose."*

---

## Preflight (T+0:00 → T+0:15, Main only)

Dependencies verified, secrets wired, scene downloaded, all services compose-ready.

### Pre-0: Verify the environment

```bash
python --version    # expect 3.11.x
node --version      # expect v20+
docker --version    # expect 24+
hf whoami           # expect your username
minikube version    # expect 1.30+
```

If any fail, fix before continuing.

### Pre-1: Write .env

Create `.env` in repo root (never committed — see `.gitignore`):

```bash
GEMINI_API_KEY=<your_gemini_key_from_notepad>
LLM_MODEL=gemini-2.0-flash
VLM_MODEL=gemini-2.0-flash
DATABASE_URL=postgresql://sceneagent:sceneagent@postgres:5432/sceneagent
REDIS_URL=redis://redis:6379/0
SCENE_ID=demo
```

Also create `.env.example` without the real key (commit this one).

### Pre-2: Repo scaffold

```
sceneagent/
├── .gitignore                 # already exists
├── .env                       # NOT committed
├── .env.example
├── README.md
├── docker-compose.yml
├── db/init/001_schema.sql
├── scripts/
│   ├── download_scene.sh
│   └── README.md
├── data/scene/demo/           # downloaded scene goes here (gitignored)
├── api/                       # Track B builds here
├── web/                       # Track A builds here
├── pipeline/                  # Track M (CV work) builds here
├── k8s/                       # Track D builds here
└── .github/workflows/
```

Create all directories now with:

```bash
mkdir -p db/init scripts data/scene/demo api/src/sceneagent/routes api/src/sceneagent/agent api/tests web pipeline/src pipeline/tests k8s .github/workflows
```

### Pre-3: .gitignore + .env.example

`.gitignore` (replaces whatever is there):

```gitignore
__pycache__/
*.py[cod]
.venv/
*.egg-info/
node_modules/
.next/
dist/
data/scene/demo/
!data/scene/demo/demo_notes.json
.env
.env.local
.vscode/
.idea/
.DS_Store
Thumbs.db
*.pt
*.pth
_hf_cache/
pipeline/output/
```

`.env.example`:

```bash
GEMINI_API_KEY=
LLM_MODEL=gemini-2.0-flash
VLM_MODEL=gemini-2.0-flash
DATABASE_URL=postgresql://sceneagent:sceneagent@postgres:5432/sceneagent
REDIS_URL=redis://redis:6379/0
SCENE_ID=demo
```

### Pre-4: Download one InteriorGS scene (run in background)

`scripts/download_scene.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
# InteriorGS scene folders are named NNNN_XXXXXX at the repo root.
# Browse https://huggingface.co/datasets/spatialverse/InteriorGS/tree/main to pick one.
# Defaults to the first listed scene.
SCENE_SUBDIR="${1:-0038_839874}"
DEST="data/scene/demo"
mkdir -p "${DEST}"
hf download spatialverse/InteriorGS \
  --repo-type dataset \
  --include "${SCENE_SUBDIR}/*" \
  --local-dir "./_hf_cache"
for f in 3dgs_compressed.ply labels.json structure.json occupancy.png occupancy.json; do
  src="./_hf_cache/${SCENE_SUBDIR}/${f}"
  [[ -f "${src}" ]] || { echo "missing: ${src}"; exit 1; }
  cp "${src}" "${DEST}/${f}"
done
ls -lh "${DEST}"
```

Run:

```bash
chmod +x scripts/download_scene.sh
./scripts/download_scene.sh 0038_839874 &
```

(If the default scene fails, browse [InteriorGS on HF](https://huggingface.co/datasets/spatialverse/InteriorGS/tree/main) and pick another folder like `0002_839955`, `0003_839989`, etc.)

### Pre-5: docker-compose.yml + initial schema

`docker-compose.yml`:

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: sceneagent
      POSTGRES_PASSWORD: sceneagent
      POSTGRES_DB: sceneagent
    ports: ["5432:5432"]
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./db/init:/docker-entrypoint-initdb.d:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U sceneagent"]
      interval: 5s
      retries: 10
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
  api:
    build: ./api
    env_file: [.env]
    ports: ["8000:8000"]
    volumes: ["./data:/app/data:ro"]
    depends_on:
      postgres: { condition: service_healthy }
  web:
    build: ./web
    environment:
      NEXT_PUBLIC_API_URL: http://localhost:8000
    ports: ["3000:3000"]
    depends_on: [api]
volumes:
  pgdata:
```

`db/init/001_schema.sql`:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS scenes (
  id UUID PRIMARY KEY,
  slug TEXT UNIQUE NOT NULL,
  title TEXT NOT NULL,
  address TEXT,
  splat_url TEXT NOT NULL,
  camera_trajectory JSONB NOT NULL,
  processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS scene_objects (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  scene_id UUID REFERENCES scenes(id) ON DELETE CASCADE,
  instance_id INT NOT NULL,
  class_name TEXT NOT NULL,
  room_label TEXT,
  centroid DOUBLE PRECISION[] NOT NULL,
  bbox_min DOUBLE PRECISION[] NOT NULL,
  bbox_max DOUBLE PRECISION[] NOT NULL,
  clip_embedding VECTOR(512) NOT NULL,
  source TEXT NOT NULL DEFAULT 'ours',
  UNIQUE (scene_id, instance_id, source)
);
CREATE INDEX IF NOT EXISTS idx_scene_objects_embedding
  ON scene_objects USING hnsw (clip_embedding vector_cosine_ops);
CREATE TABLE IF NOT EXISTS notes (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  scene_id UUID REFERENCES scenes(id) ON DELETE CASCADE,
  text TEXT NOT NULL,
  video_timestamp DOUBLE PRECISION NOT NULL,
  category TEXT,
  category_confidence REAL,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS hotspots (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  note_id UUID UNIQUE REFERENCES notes(id) ON DELETE CASCADE,
  object_id UUID REFERENCES scene_objects(id) ON DELETE SET NULL,
  match_confidence REAL NOT NULL,
  position DOUBLE PRECISION[] NOT NULL,
  auto_accepted BOOLEAN NOT NULL DEFAULT TRUE
);
```

### Pre-6: Seed notes

`data/scene/demo/demo_notes.json`:

```json
[
  {"text": "The bedroom window sticks — pull firmly", "video_timestamp": 3.2},
  {"text": "This desk comes with the apartment", "video_timestamp": 8.5},
  {"text": "Ceilings are 3.2m in the living room", "video_timestamp": 12.1},
  {"text": "Heated bathroom floor, lovely in winter", "video_timestamp": 18.4},
  {"text": "WiFi router is behind the TV", "video_timestamp": 14.7},
  {"text": "Washing machine included, relatively new", "video_timestamp": 22.0},
  {"text": "Radiator near the couch is loud at 6am", "video_timestamp": 16.3},
  {"text": "Building was a bakery in the 1920s", "video_timestamp": 25.5}
]
```

### Pre-7: Commit and kick off subagents

```bash
git add -A
git commit -m "chore(preflight): scaffolding, docker-compose, db schema, seed notes"
```

Then: **dispatch Track A and Track B subagents now.** (Code blocks below.)

---

## Dispatching subagents

From the main Claude Code session, call the Agent tool with the following prompts.

### Dispatch Track A (frontend subagent)

```
subagent_type: general-purpose
description: Build SceneAgent frontend (Next.js + splat viewer)
prompt: (see §Track A below — include the ENTIRE Track A section as the prompt)
```

### Dispatch Track B (backend subagent)

```
subagent_type: general-purpose
description: Build SceneAgent API (FastAPI + agent + MCP)
prompt: (see §Track B below — include the ENTIRE Track B section as the prompt)
```

**Both subagents work inside the same repo (`C:\Users\91910\Downloads\sceneagent\`).** They don't need worktrees — they touch disjoint files (`web/` vs `api/`).

### Dispatch Track D (infra subagent) at T+3:00

Once Tracks A and B are ~60% done, dispatch Track D with the §Track D section.

---

## Track M — CV segmentation pipeline (main session runs this)

**Goal:** End-to-end 2D→3D semantic segmentation of the InteriorGS splat, evaluated against ground truth, output usable as product labels.

**Work location:** `pipeline/`

**Deliverable by T+4:00:** `data/scene/demo/object_inventory.json` (our labels) + `pipeline/output/metrics.json` (evaluation).

### M-1: Render 30 views from the splat via headless Puppeteer (T+0:15 → T+1:00)

**Files:**
- `pipeline/render_views/index.html` (loads the splat, exposes camera control)
- `pipeline/render_views/render.mjs` (Node + Puppeteer driver)
- `pipeline/render_views/camera_poses.json` (30 scripted poses)

**`pipeline/render_views/camera_poses.json`** — generate 30 poses evenly distributed around scene. Simple approach:
```json
[
  {"position": [3, 3, 1.7], "lookAt": [0, 0, 1], "yaw_deg": 0},
  {"position": [3, -3, 1.7], "lookAt": [0, 0, 1], "yaw_deg": 90},
  ... (30 total, ring of 10 at eye level + ring of 10 higher + ring of 10 inside rooms)
]
```

Write a small Python helper `pipeline/gen_camera_poses.py` that reads the bbox of all objects from `labels.json` and samples 30 poses around interesting regions.

**`pipeline/render_views/index.html`:**

```html
<!doctype html><html><body style="margin:0;background:#000">
<div id="viewer" style="position:fixed;inset:0"></div>
<script type="module">
import { Viewer } from 'https://unpkg.com/@mkkellogg/gaussian-splats-3d@0.4.7/dist/index.js';
const params = new URLSearchParams(location.search);
const splatUrl = params.get('splat');
const camX = parseFloat(params.get('x') || 0);
const camY = parseFloat(params.get('y') || 0);
const camZ = parseFloat(params.get('z') || 1.7);
const tgtX = parseFloat(params.get('tx') || 0);
const tgtY = parseFloat(params.get('ty') || 0);
const tgtZ = parseFloat(params.get('tz') || 1.0);
const viewer = new Viewer({
  rootElement: document.getElementById('viewer'),
  cameraUp: [0, 0, 1],
  initialCameraPosition: [camX, camY, camZ],
  initialCameraLookAt: [tgtX, tgtY, tgtZ],
});
viewer.addSplatScene(splatUrl).then(() => { viewer.start(); window._ready = true; });
</script></body></html>
```

**`pipeline/render_views/render.mjs`:**

```javascript
import puppeteer from 'puppeteer';
import fs from 'fs';
import path from 'path';
import http from 'http';

const POSES = JSON.parse(fs.readFileSync('pipeline/render_views/camera_poses.json'));
const SPLAT = 'http://localhost:8081/3dgs_compressed.ply';
const OUT_DIR = 'pipeline/output/views';
fs.mkdirSync(OUT_DIR, { recursive: true });

// Serve the scene files on :8081 so Puppeteer can load them
const staticServer = http.createServer((req, res) => {
  const filepath = 'data/scene/demo' + req.url;
  if (!fs.existsSync(filepath)) { res.statusCode = 404; res.end(); return; }
  res.end(fs.readFileSync(filepath));
}).listen(8081);

// Serve the viewer HTML on :8082
const viewerServer = http.createServer((req, res) => {
  res.setHeader('Content-Type', 'text/html');
  res.end(fs.readFileSync('pipeline/render_views/index.html'));
}).listen(8082);

const browser = await puppeteer.launch({ headless: 'new', args: ['--no-sandbox'] });
const page = await browser.newPage();
await page.setViewport({ width: 1024, height: 768 });

for (let i = 0; i < POSES.length; i++) {
  const p = POSES[i];
  const url = `http://localhost:8082/?splat=${encodeURIComponent(SPLAT)}&x=${p.position[0]}&y=${p.position[1]}&z=${p.position[2]}&tx=${p.lookAt[0]}&ty=${p.lookAt[1]}&tz=${p.lookAt[2]}`;
  await page.goto(url);
  await page.waitForFunction(() => window._ready === true, { timeout: 30000 });
  await new Promise(r => setTimeout(r, 1000));  // let splat settle
  const outPath = path.join(OUT_DIR, `view_${String(i).padStart(3, '0')}.png`);
  await page.screenshot({ path: outPath, fullPage: false });
  console.log(`rendered ${i+1}/${POSES.length}`);
}

await browser.close();
staticServer.close();
viewerServer.close();
// Also dump camera intrinsics alongside
fs.writeFileSync('pipeline/output/views/_intrinsics.json', JSON.stringify({
  width: 1024, height: 768, fov_vertical_deg: 60.0, poses: POSES,
}, null, 2));
```

**Install + run:**

```bash
cd pipeline/render_views && npm init -y && npm install puppeteer
cd ../..
node pipeline/render_views/render.mjs
```

**Verify:** `ls pipeline/output/views/` shows `view_000.png` through `view_029.png` plus `_intrinsics.json`.

**Fallback if Puppeteer fails:** run `pipeline/gen_camera_poses.py` in a mode that uses `matplotlib` to render simple bbox views from `labels.json` directly (2D mockups of the scene) — losing realism but preserving the rest of the pipeline structure. Document this fallback in README.

**Commit:**
```bash
git add pipeline/
git commit -m "feat(pipeline): headless Puppeteer renderer producing 30 splat views for segmentation"
```

### M-2: 2D segmentation with MobileSAM + CLIP zero-shot labeling (T+1:00 → T+2:00)

**Files:** `pipeline/src/segment.py`, `pipeline/pyproject.toml`

**`pipeline/pyproject.toml`:**

```toml
[project]
name = "sceneagent-pipeline"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "numpy>=1.26",
  "pillow>=10.0",
  "torch>=2.2",
  "torchvision>=0.17",
  "open-clip-torch>=2.24",
  "mobile-sam @ git+https://github.com/ChaoningZhang/MobileSAM.git",
  "scikit-learn>=1.4",
  "opencv-python>=4.9",
  "tqdm>=4.66",
]
```

**`pipeline/src/segment.py`:**

```python
"""Run MobileSAM + CLIP zero-shot classification on each rendered view.

Output: pipeline/output/masks/view_<idx>.json with list of
  {mask_id, bbox, class_name, class_confidence, mask_rle}
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

HOME_CLASSES = [
    "wall", "floor", "ceiling", "window", "door", "bed", "sofa", "chair",
    "table", "desk", "lamp", "television", "bookshelf", "cabinet", "sink",
    "toilet", "bathtub", "stove", "refrigerator", "radiator", "mirror",
    "plant", "painting", "rug", "curtain", "pillow", "nightstand", "stool",
    "ottoman", "shelf", "shower", "oven", "microwave", "washing_machine",
    "dryer", "shoe", "clock", "vase", "fireplace", "armchair",
]


def load_sam():
    # MobileSAM weights
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
    # Pre-encode class texts
    prompts = [f"a photo of a {c.replace('_', ' ')}" for c in HOME_CLASSES]
    with torch.no_grad():
        text_emb = model.encode_text(tokenizer(prompts))
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    return model, preprocess, text_emb


def mask_to_rle(mask: np.ndarray) -> dict:
    """Simple RLE: list of run lengths along flat binary mask."""
    flat = mask.flatten().astype(np.uint8)
    changes = np.diff(np.concatenate([[0], flat, [0]]))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return {"shape": list(mask.shape), "runs": [[int(s), int(e - s)] for s, e in zip(starts, ends)]}


def segment_view(
    img_path: Path, mask_gen, clip_model, clip_preprocess, clip_text_emb
) -> list[dict]:
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    masks = mask_gen.generate(img_rgb)

    results = []
    for m_idx, m in enumerate(masks):
        seg = m["segmentation"]
        bbox = m["bbox"]  # [x,y,w,h]
        # Crop masked region for CLIP
        x, y, w, h = [int(v) for v in bbox]
        if w < 20 or h < 20:
            continue
        crop = img_rgb[y:y+h, x:x+w]
        mask_crop = seg[y:y+h, x:x+w]
        # Zero out non-mask pixels
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
```

**Install + run:**

```bash
cd pipeline && pip install -e .
cd ..
python -m pipeline.src.segment
```

**Run time estimate:** ~25–40 seconds per view on CPU × 30 views = 12–20 min. Let it run in background while you do M-3 prep.

**Verify:** `ls pipeline/output/masks/` shows `view_000.json` through `view_029.json`. Spot-check one: should have ~10–30 masks with reasonable class names.

**Commit:**
```bash
git add pipeline/src/segment.py pipeline/pyproject.toml
git commit -m "feat(pipeline): 2D segmentation with MobileSAM + OpenCLIP zero-shot labeling"
```

### M-3: 2D→3D backprojection + DBSCAN instance clustering (T+2:00 → T+3:00)

**Files:** `pipeline/src/backproject.py`

**`pipeline/src/backproject.py`:**

```python
"""Backproject 2D masks onto 3D Gaussians; cluster into instances.

Input:
  pipeline/output/views/_intrinsics.json  (camera poses + intrinsics)
  pipeline/output/masks/view_*.json       (per-view SAM masks + CLIP labels)
  data/scene/demo/3dgs_compressed.ply     (Gaussian positions — we only need centers)

Output:
  pipeline/output/object_inventory.json   (our predicted object list)
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN
from plyfile import PlyData

VIEWS_DIR = Path("pipeline/output/views")
MASKS_DIR = Path("pipeline/output/masks")
PLY_PATH = Path("data/scene/demo/3dgs_compressed.ply")
OUT_PATH = Path("pipeline/output/object_inventory.json")


def load_gaussian_centers(ply_path: Path) -> np.ndarray:
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]
    return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)


def rle_to_mask(rle: dict) -> np.ndarray:
    h, w = rle["shape"]
    m = np.zeros(h * w, dtype=np.uint8)
    for start, length in rle["runs"]:
        m[start:start+length] = 1
    return m.reshape(h, w).astype(bool)


def pose_to_world_to_cam(pose: dict) -> np.ndarray:
    """Build 4x4 world→camera matrix from {position, lookAt, cameraUp assumed=[0,0,1]}."""
    pos = np.asarray(pose["position"])
    target = np.asarray(pose["lookAt"])
    up = np.array([0, 0, 1], dtype=float)
    fwd = target - pos
    fwd /= np.linalg.norm(fwd) + 1e-9
    right = np.cross(fwd, up); right /= np.linalg.norm(right) + 1e-9
    up2 = np.cross(right, fwd)
    R = np.stack([right, up2, -fwd], axis=0)
    t = -R @ pos
    M = np.eye(4)
    M[:3, :3] = R; M[:3, 3] = t
    return M


def project_points(
    points: np.ndarray, w2c: np.ndarray, width: int, height: int, fov_v_deg: float
) -> np.ndarray:
    """Return (N, 3): u, v, depth. u,v are pixel coords; depth < 0 means behind cam."""
    N = points.shape[0]
    homo = np.hstack([points, np.ones((N, 1))])
    cam = (w2c @ homo.T).T
    z = cam[:, 2]
    fov_rad = math.radians(fov_v_deg)
    f = (height / 2) / math.tan(fov_rad / 2)
    u = (cam[:, 0] / (-z + 1e-9)) * f + width / 2
    v = -(cam[:, 1] / (-z + 1e-9)) * f + height / 2
    return np.stack([u, v, -z], axis=1)  # depth positive = in front


def vote_class_per_gaussian(centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (class_idx per Gaussian, confidence per Gaussian)."""
    intrinsics = json.loads((VIEWS_DIR / "_intrinsics.json").read_text())
    W, H = intrinsics["width"], intrinsics["height"]
    FOV_V = intrinsics["fov_vertical_deg"]

    N = centers.shape[0]
    # Class vocabulary (same as segment.py)
    from pipeline.src.segment import HOME_CLASSES
    K = len(HOME_CLASSES)
    class_scores = np.zeros((N, K), dtype=np.float32)

    for i, pose in enumerate(intrinsics["poses"]):
        mask_path = MASKS_DIR / f"view_{i:03d}.json"
        if not mask_path.exists():
            continue
        masks = json.loads(mask_path.read_text())
        if not masks:
            continue

        w2c = pose_to_world_to_cam(pose)
        proj = project_points(centers, w2c, W, H, FOV_V)
        in_front = proj[:, 2] > 0.1
        in_img = (proj[:, 0] >= 0) & (proj[:, 0] < W) & (proj[:, 1] >= 0) & (proj[:, 1] < H)
        valid = in_front & in_img

        for m in masks:
            cls_idx = HOME_CLASSES.index(m["class_name"])
            conf = float(m["class_confidence"])
            mask_2d = rle_to_mask(m["mask_rle"])
            ui = proj[valid, 0].astype(int)
            vi = proj[valid, 1].astype(int)
            hits = mask_2d[vi, ui]
            valid_idx = np.where(valid)[0][hits]
            class_scores[valid_idx, cls_idx] += conf

    best_class = class_scores.argmax(axis=1)
    best_conf = class_scores.max(axis=1) / (class_scores.sum(axis=1) + 1e-6)
    return best_class, best_conf


def cluster_instances(
    centers: np.ndarray, class_idx: np.ndarray, min_vote_confidence: float = 0.1,
) -> np.ndarray:
    """Per-class DBSCAN → instance IDs. Unlabeled Gaussians get -1."""
    instance_ids = np.full(len(centers), -1, dtype=np.int32)
    next_iid = 0
    for cls in np.unique(class_idx):
        mask = class_idx == cls
        if mask.sum() < 20:
            continue
        pts = centers[mask]
        labels = DBSCAN(eps=0.3, min_samples=20).fit_predict(pts)
        global_labels = labels.copy()
        unique_labels = np.unique(labels[labels >= 0])
        for i, u in enumerate(unique_labels):
            global_labels[labels == u] = next_iid
            next_iid += 1
        instance_ids[mask] = global_labels
    return instance_ids


def instances_to_inventory(
    centers: np.ndarray, class_idx: np.ndarray, instance_ids: np.ndarray,
) -> list[dict]:
    from pipeline.src.segment import HOME_CLASSES
    out = []
    for iid in np.unique(instance_ids):
        if iid < 0:
            continue
        mask = instance_ids == iid
        pts = centers[mask]
        bbox_min = pts.min(0).tolist()
        bbox_max = pts.max(0).tolist()
        centroid = ((pts.min(0) + pts.max(0)) / 2).tolist()
        cls = int(class_idx[mask][0])
        out.append({
            "instance_id": int(iid),
            "class_name": HOME_CLASSES[cls],
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "centroid": centroid,
            "point_count": int(mask.sum()),
        })
    return out


def main():
    print("loading gaussians...")
    centers = load_gaussian_centers(PLY_PATH)
    print(f"{centers.shape[0]} gaussians")
    print("voting per gaussian...")
    cls_idx, cls_conf = vote_class_per_gaussian(centers)
    print("clustering instances...")
    iids = cluster_instances(centers, cls_idx)
    print(f"{iids[iids>=0].max()+1 if (iids>=0).any() else 0} instances")
    inv = instances_to_inventory(centers, cls_idx, iids)
    OUT_PATH.write_text(json.dumps(inv, indent=2))
    print(f"wrote {OUT_PATH} with {len(inv)} objects")


if __name__ == "__main__":
    main()
```

**Install extra dep + run:**

```bash
pip install plyfile
python -m pipeline.src.backproject
```

**Run time:** ~1–3 min.

**Verify:** `head pipeline/output/object_inventory.json` — should show a JSON array with reasonable `class_name` + `centroid` entries.

**Fallback if empty / broken:** Use InteriorGS's own `labels.json` as `object_inventory.json` input (just reshape it). Document in README: "Our segmentation pipeline struggled on this scene — here are our metrics; for the product demo we used ground truth as fallback."

**Commit:**
```bash
git add pipeline/src/backproject.py
git commit -m "feat(pipeline): 2D→3D multi-view voting + DBSCAN instance clustering"
```

### M-4: Evaluate vs InteriorGS ground truth (T+3:00 → T+3:30)

**Files:** `pipeline/src/evaluate.py`

**`pipeline/src/evaluate.py`:**

```python
"""Evaluate our predicted object inventory vs InteriorGS labels.json."""
from __future__ import annotations

import json
from pathlib import Path
from collections import Counter

import numpy as np

OURS_PATH = Path("pipeline/output/object_inventory.json")
GT_PATH = Path("data/scene/demo/labels.json")
OUT_PATH = Path("pipeline/output/metrics.json")


def bbox_from_corners(corners):
    a = np.asarray(corners)
    return a.min(0).tolist(), a.max(0).tolist()


def bbox_iou(a_min, a_max, b_min, b_max) -> float:
    a_min = np.asarray(a_min); a_max = np.asarray(a_max)
    b_min = np.asarray(b_min); b_max = np.asarray(b_max)
    inter_min = np.maximum(a_min, b_min)
    inter_max = np.minimum(a_max, b_max)
    inter = np.prod(np.maximum(inter_max - inter_min, 0))
    vol_a = np.prod(a_max - a_min)
    vol_b = np.prod(b_max - b_min)
    union = vol_a + vol_b - inter
    return float(inter / union) if union > 0 else 0.0


def normalize_class(s: str) -> str:
    return s.lower().replace("-", "_").replace(" ", "_")


def main():
    ours = json.loads(OURS_PATH.read_text())
    gt_raw = json.loads(GT_PATH.read_text())
    gt = []
    for obj in gt_raw.get("objects", []):
        bmin, bmax = bbox_from_corners(obj["bbox_corners"])
        gt.append({
            "instance_id": obj["instance_id"],
            "class_name": normalize_class(obj["class_name"]),
            "bbox_min": bmin, "bbox_max": bmax,
        })

    tp = fn = fp = 0
    class_tp = Counter(); class_fn = Counter(); class_fp = Counter()
    matched_gt = set()

    for o in ours:
        oc = normalize_class(o["class_name"])
        best_iou = 0.0; best_gt = None
        for g in gt:
            if g["instance_id"] in matched_gt:
                continue
            iou = bbox_iou(o["bbox_min"], o["bbox_max"], g["bbox_min"], g["bbox_max"])
            if iou > best_iou:
                best_iou, best_gt = iou, g
        if best_gt and best_iou >= 0.25:
            if normalize_class(best_gt["class_name"]) == oc:
                tp += 1
                class_tp[oc] += 1
                matched_gt.add(best_gt["instance_id"])
            else:
                fp += 1
                class_fp[oc] += 1
        else:
            fp += 1
            class_fp[oc] += 1

    for g in gt:
        if g["instance_id"] not in matched_gt:
            fn += 1
            class_fn[g["class_name"]] += 1

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    per_class = {}
    all_classes = set(class_tp) | set(class_fn) | set(class_fp)
    for c in all_classes:
        p = class_tp[c] / (class_tp[c] + class_fp[c]) if (class_tp[c] + class_fp[c]) else 0
        r = class_tp[c] / (class_tp[c] + class_fn[c]) if (class_tp[c] + class_fn[c]) else 0
        per_class[c] = {"precision": p, "recall": r, "tp": class_tp[c], "fp": class_fp[c], "fn": class_fn[c]}

    result = {
        "num_predicted": len(ours),
        "num_ground_truth": len(gt),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "iou_threshold": 0.25,
        "per_class": per_class,
    }
    OUT_PATH.write_text(json.dumps(result, indent=2))
    print(f"Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")
    print(f"TP: {tp}  FP: {fp}  FN: {fn}")


if __name__ == "__main__":
    main()
```

**Run:**

```bash
python -m pipeline.src.evaluate
```

**Verify:** `cat pipeline/output/metrics.json` shows non-zero numbers. Even 10% precision is a legitimate portfolio number as long as you report it honestly.

**Commit:**
```bash
git add pipeline/src/evaluate.py
git commit -m "feat(pipeline): evaluate predicted inventory vs InteriorGS GT (P/R/F1 + per-class breakdown)"
```

### M-5: Seed our labels into Postgres (T+3:30 → T+4:00)

**Files:** `pipeline/src/seed_db.py`

```python
"""Load our object_inventory.json into the SceneAgent Postgres.

Also computes a CLIP text embedding per object (one description string) for the note matcher.
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path

import asyncpg
import open_clip
import torch
from pgvector.asyncpg import register_vector

INV_PATH = Path("pipeline/output/object_inventory.json")
GT_FALLBACK_PATH = Path("data/scene/demo/labels.json")
SCENE_SLUG = os.environ.get("SCENE_ID", "demo")
DB_URL = os.environ.get("DATABASE_URL",
    "postgresql://sceneagent:sceneagent@localhost:5432/sceneagent")


def get_inventory() -> tuple[list[dict], str]:
    """Returns (inventory, source). Falls back to GT if our pipeline failed."""
    if INV_PATH.exists():
        inv = json.loads(INV_PATH.read_text())
        if len(inv) >= 10:
            return inv, "ours"
    # fallback
    raw = json.loads(GT_FALLBACK_PATH.read_text())
    inv = []
    import numpy as np
    for obj in raw["objects"]:
        a = np.asarray(obj["bbox_corners"])
        inv.append({
            "instance_id": obj["instance_id"],
            "class_name": obj["class_name"],
            "bbox_min": a.min(0).tolist(),
            "bbox_max": a.max(0).tolist(),
            "centroid": ((a.min(0) + a.max(0))/2).tolist(),
        })
    return inv, "ground_truth_fallback"


def compute_embeddings(inv: list[dict]) -> list[list[float]]:
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    prompts = [f"a {o['class_name'].replace('_', ' ')}" for o in inv]
    with torch.no_grad():
        emb = model.encode_text(tokenizer(prompts))
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.tolist()


async def seed():
    inv, source = get_inventory()
    embs = compute_embeddings(inv)
    print(f"seeding {len(inv)} objects from source={source}")

    pool = await asyncpg.create_pool(DB_URL, init=register_vector)
    async with pool.acquire() as con:
        row = await con.fetchrow("SELECT id FROM scenes WHERE slug=$1", SCENE_SLUG)
        if row:
            scene_id = row["id"]
        else:
            scene_id = uuid.uuid4()
            # build a simple linear camera trajectory for the scrubber UI
            trajectory = [
                {"timestamp": i * 1.0, "position": [i * 0.3, 0, 1.7], "yaw_deg": (i * 12) % 360}
                for i in range(30)
            ]
            await con.execute(
                """INSERT INTO scenes(id, slug, title, splat_url, camera_trajectory)
                   VALUES ($1, $2, $3, $4, $5)""",
                scene_id, SCENE_SLUG, f"Demo Listing ({SCENE_SLUG})",
                f"/static/scene/{SCENE_SLUG}/3dgs_compressed.ply",
                json.dumps(trajectory),
            )
        # clear prior 'ours' rows
        await con.execute(
            "DELETE FROM scene_objects WHERE scene_id=$1 AND source=$2", scene_id, source
        )
        for o, e in zip(inv, embs):
            await con.execute(
                """INSERT INTO scene_objects
                   (scene_id, instance_id, class_name, centroid, bbox_min, bbox_max,
                    clip_embedding, source)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8)""",
                scene_id, o["instance_id"], o["class_name"],
                list(o["centroid"]), list(o["bbox_min"]), list(o["bbox_max"]),
                e, source,
            )

        # seed demo notes
        notes = json.loads(Path("data/scene/demo/demo_notes.json").read_text())
        await con.execute("DELETE FROM notes WHERE scene_id=$1", scene_id)
        for n in notes:
            await con.execute(
                "INSERT INTO notes(scene_id, text, video_timestamp) VALUES ($1,$2,$3)",
                scene_id, n["text"], n["video_timestamp"],
            )
    print("seed complete")


if __name__ == "__main__":
    asyncio.run(seed())
```

**Run (after `docker compose up postgres` is healthy):**

```bash
docker compose up -d postgres
pip install asyncpg pgvector
python -m pipeline.src.seed_db
```

**Verify:**

```bash
docker compose exec postgres psql -U sceneagent -d sceneagent -c \
  "SELECT class_name, count(*) FROM scene_objects GROUP BY class_name ORDER BY 2 DESC LIMIT 10;"
```

**Commit:**
```bash
git add pipeline/src/seed_db.py
git commit -m "feat(pipeline): seed DB with our predicted objects (fallback to GT if pipeline weak)"
```

---

## Track A — Frontend (subagent 1)

**DISPATCH THIS PROMPT AS-IS TO A GENERAL-PURPOSE SUBAGENT.** The subagent works in `C:\Users\91910\Downloads\sceneagent\web\` only.

```
You are Track A of the SceneAgent project. Build the Next.js 14 frontend with Gaussian
splat viewer, hotspot overlay, AI chat, and realtor scrubber UI.

WORKING DIR: C:\Users\91910\Downloads\sceneagent\web

REFERENCE DOCS (read them first):
- docs/superpowers/specs/2026-04-20-sceneagent-design.md (sections 10, 12)
- docs/superpowers/plans/2026-04-20-sceneagent-plan.md (this plan, §Track A)

YOUR TASKS (execute in order, commit after each):

### A-1: Scaffold Next.js app (0:15-0:30)
cd web
npx create-next-app@latest . --ts --tailwind --app --src-dir --no-eslint --import-alias "@/*" --no-turbo
npm install three @mkkellogg/gaussian-splats-3d zustand @tanstack/react-query react-player lucide-react clsx
npm install -D @types/three
Add web/Dockerfile:
FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
CMD ["npm","run","dev","--","-p","3000"]
Commit: "feat(web): Next.js scaffold + deps"

### A-2: Zustand viewer store
Create web/src/stores/viewer.ts with:
- state: highlightedHotspotIds[], flyToPosition | null, tourStops | null
- actions: setHighlights, flyTo, setTour, clear
Commit: "feat(web): zustand viewer store"

### A-3: SplatViewer component
Create web/src/components/SplatViewer.tsx:
- Client-only component wrapping @mkkellogg/gaussian-splats-3d Viewer
- Props: splatUrl (string)
- Reacts to flyToPosition from store: lerp camera to that position over 1s
- cameraUp=[0,0,1], initialPosition=[4,4,2]
Commit: "feat(web): splat viewer with fly-to support"

### A-4: API client
Create web/src/lib/api.ts with functions:
- getScene(slug), getHotspots(slug, category?), createNote(slug, text, ts)
- chat(slug, message) returns {response, tool_calls[]}
- seedMatch(slug) POST /scenes/:slug/notes/seed-match
Base URL from NEXT_PUBLIC_API_URL env, default http://localhost:8000.
Commit: "feat(web): REST client"

### A-5: React Query provider + homepage
Create web/src/app/providers.tsx with QueryClientProvider.
Wrap children in web/src/app/layout.tsx.
Homepage at web/src/app/page.tsx with links to /listing/demo and /listing/demo/admin.
Commit: "feat(web): providers + homepage"

### A-6: HotspotMarkers component
Create web/src/components/HotspotMarkers.tsx:
- Props: hotspots[] with {id, position, note_text, category, class_name}
- Render each as a category-icon button positioned via CSS (for v1, just use screen-center stub + rely on flyTo to center the hotspot visually when clicked)
- On click: call flyTo(position), open a popup card with the note text
Category icons:
feature='✅', included='🎁', issue='⚠️', info='ℹ️', spec='📏', story='📖', other='📍'
Commit: "feat(web): clickable hotspot markers"

### A-7: ChatOverlay component
Create web/src/components/ChatOverlay.tsx:
- Fixed bottom-right 384px wide panel
- Shows messages (user / assistant / tool) with icons
- Input box + send button
- On send: POST /scenes/:slug/chat, append response
- When tool_calls include find_by_description or plan_tour: call flyTo(result[0].centroid)
Commit: "feat(web): chat overlay with camera side-effects"

### A-8: Listing page (buyer view)
Create web/src/app/listing/[slug]/page.tsx:
- Fetches scene + hotspots via useQuery
- On mount, calls seedMatch(slug) then refetches hotspots
- Renders full-screen <SplatViewer> + <HotspotMarkers> + <ChatOverlay>
- Top-left shows scene title
Commit: "feat(web): buyer listing page"

### A-9: Scrubber admin page
Create web/src/app/listing/[slug]/admin/page.tsx:
- Two columns
- Left: react-player video (url = ${NEXT_PUBLIC_API_URL}/static/scene/:slug/video.mp4), textarea, "Add note at current time" button
- On click: get current player time, POST note, invalidate hotspots query
- Right: list of hotspots (category, class, confidence, note text)
Commit: "feat(web): scrubber admin page"

VERIFICATION at end: `docker compose up web api` + browser to http://localhost:3000.

FALLBACK if @mkkellogg/gaussian-splats-3d fails to load compressed .ply:
try antimatter15/splat-based viewer OR pre-convert the .ply to .ksplat with
gaussian-splats-3d's CLI tool.

REPORT progress every 30 minutes. Final output: all commits pushed.
```

---

## Track B — Backend API (subagent 2)

**DISPATCH THIS PROMPT AS-IS TO A GENERAL-PURPOSE SUBAGENT.** Works in `api/` only.

```
You are Track B of the SceneAgent project. Build the FastAPI backend with
Postgres+pgvector access, note/hotspot endpoints, MCP server, and LangGraph agent.

WORKING DIR: C:\Users\91910\Downloads\sceneagent\api

REFERENCE DOCS:
- docs/superpowers/specs/2026-04-20-sceneagent-design.md (sections 4, 6, 9)
- docs/superpowers/plans/2026-04-20-sceneagent-plan.md (§Track B)

YOUR TASKS (execute in order, commit after each):

### B-1: FastAPI scaffold + DB connection
Create api/pyproject.toml with deps: fastapi, uvicorn[standard], pydantic, asyncpg,
pgvector, google-generativeai, langgraph, mcp, numpy, open-clip-torch, torch, pillow,
httpx, sse-starlette.
Create api/src/sceneagent/__init__.py
Create api/src/sceneagent/main.py:
- FastAPI app with lifespan that initializes asyncpg pool (read DATABASE_URL env)
- /health endpoint returning {"status":"ok"}
- Mount StaticFiles at /static serving /app/data (for the .ply + video.mp4)
Create api/src/sceneagent/db.py:
- init_db_pool(), close_db_pool(), pool() helpers
- Uses pgvector.asyncpg.register_vector as init callback
Create api/Dockerfile:
FROM python:3.11-slim
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml ./
RUN pip install --no-cache-dir .
COPY src/ ./src/
ENV PYTHONPATH=/app/src
CMD ["uvicorn","sceneagent.main:app","--host","0.0.0.0","--port","8000"]
Commit: "feat(api): FastAPI scaffold + DB pool + static file mount"

### B-2: Scene + objects read endpoints
Create api/src/sceneagent/routes/scenes.py:
- GET /scenes/:slug → scene row
- GET /scenes/:slug/objects?room=&class_name= → scene_objects filtered
Wire router in main.py.
Commit: "feat(api): scenes + objects endpoints"

### B-3: Frustum + CLIP utilities
Create api/src/sceneagent/geometry.py:
- is_point_visible_from_pose(point, pose, fov_deg=60, max_distance=15) → bool
- pose_from_yaw(position, yaw_deg) → CameraPose
Create api/src/sceneagent/clip_util.py:
- encode_text(text: str) -> list[float] using open_clip ViT-B-32
- @lru_cache the model loading
Commit: "feat(api): geometry + CLIP text encoder"

### B-4: Note matcher + categorizer
Create api/src/sceneagent/matcher.py:
- rank_objects_for_note(note_text, pose, objects, fov_deg=60, max_distance=15) -> list[dict]
- Frustum-filter objects, CLIP-similarity rank against note embedding, return sorted
Create api/src/sceneagent/categorizer.py:
- classify_category(text) -> (category: str, confidence: float)
- One Gemini Flash call, returns JSON {category, confidence}
- Categories: feature/included/issue/info/spec/story/other
Commit: "feat(api): matcher + categorizer"

### B-5: Notes + hotspots endpoints
Create api/src/sceneagent/routes/notes.py:
- POST /scenes/:slug/notes {text, video_timestamp}
  1. Insert note
  2. Classify category via LLM
  3. Get camera pose at timestamp from scene.camera_trajectory (nearest)
  4. Filter objects via frustum + rank by CLIP
  5. Create hotspot for top match
  Returns {note_id, category, hotspot}
- POST /scenes/:slug/notes/seed-match
  Matches all notes without hotspots (used after initial seed)
Create api/src/sceneagent/routes/hotspots.py:
- GET /scenes/:slug/hotspots?category= → hotspots joined with notes + objects
Wire both routers in main.py.
Commit: "feat(api): notes + hotspots endpoints"

### B-6: MCP server + agent tools
Create api/src/sceneagent/agent/tools.py with async functions (callable from agent OR MCP):
- list_objects(scene_slug, room=None, class_name=None, limit=50)
- list_hotspots(scene_slug, category=None)
- find_by_description(scene_slug, text, limit=5) -- uses pgvector <=> operator
- measure_distance(scene_slug, point_a, point_b) -> {meters}
- plan_tour(scene_slug, focus=None) -> list of stops [{position, dwell_seconds, narration_hint, highlight_hotspot_id}]
Create api/src/sceneagent/mcp_server.py with FastMCP wrapping each tool.
Commit: "feat(api): MCP server + tool implementations"

### B-7: VLM grounding + render_view proxy
Create api/src/sceneagent/vlm.py:
- describe_image(image_b64, question) -> str via Gemini Flash vision
Create api/src/sceneagent/render_proxy.py:
- render_view(scene_slug, position) -> {image_base64, width, height}
  For v1: picks the closest rendered view from pipeline/output/views/
  (pre-rendered by Track M) based on Euclidean distance to the given position.
  Falls back to a 640x360 gray PNG if no frames exist.
Commit: "feat(api): VLM wrapper + render_view proxy using pre-rendered splat views"

### B-8: LangGraph agent + SSE chat
Create api/src/sceneagent/agent/graph.py:
- State: {scene_slug, user_message, history, tool_calls, response}
- Single node: _execute that:
  1. Calls Gemini Flash with SYSTEM_PROMPT + tool list + history
  2. Gemini returns {tool: name, args: dict}
  3. Dispatches to the appropriate tool (list_hotspots, find_by_description,
     measure_distance, render_view, plan_tour, or answer)
  4. If tool=answer, set state.response and END
  5. Else append to history and loop (max 6 iterations)
Create api/src/sceneagent/routes/chat.py:
- POST /scenes/:slug/chat {message} -> {response, tool_calls}
- POST /scenes/:slug/chat/stream -> SSE (optional, stretch)
Wire chat router in main.py.
Commit: "feat(api): LangGraph agent + SSE chat endpoint"

VERIFICATION at end:
curl -X POST http://localhost:8000/scenes/demo/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Are there any issues?"}'
Should return JSON with response and tool_calls list including list_hotspots call.

REPORT progress every 30 min.
```

---

## Track D — Infra + README + CI (subagent 3, dispatched at T+3:00)

**DISPATCH THIS PROMPT AS-IS AT T+3:00.**

```
You are Track D of the SceneAgent project. Build Kubernetes manifests, GitHub Actions
CI, and flesh out the README.

WORKING DIR: C:\Users\91910\Downloads\sceneagent\

REFERENCE DOCS:
- docs/superpowers/specs/2026-04-20-sceneagent-design.md (sections 11, 12)

TASKS:

### D-1: K8s manifests (0:30)
Create minimal manifests in k8s/:
- 00-namespace.yaml: Namespace "sceneagent"
- 10-postgres.yaml: StatefulSet (pgvector/pgvector:pg16) + PVC + Service (5432)
- 20-redis.yaml: Deployment + Service (6379)
- 30-api.yaml: Deployment (image sceneagent-api:dev) + Service (8000), env from secret
- 40-web.yaml: Deployment (sceneagent-web:dev) + NodePort Service (30000)
- 99-secrets.example.yaml: Secret "agent-secrets" with GEMINI_API_KEY placeholder
Create k8s/apply.sh that:
- mounts ./data into minikube via `minikube mount` in background
- builds images inside minikube's docker (eval "$(minikube docker-env)")
- applies manifests in order
- waits for rollouts
- prints `minikube service -n sceneagent web --url`
Test apply.sh on the user's minikube. Note: skip if minikube isn't running; just ship manifests.
Commit: "feat(infra): K8s manifests + apply script for minikube"

### D-2: GitHub Actions CI (0:15)
Create .github/workflows/ci.yml:
jobs:
  api: ubuntu, python 3.11, ruff check api/src/, python -c "import sceneagent.main"
  web: ubuntu, node 20, npm ci, npx tsc --noEmit, npm run build
Triggers: push, pull_request.
Commit: "ci: lint + typecheck + build on push/PR"

### D-3: Flesh out README (0:30)
Edit README.md to include:
- Opening pitch
- Architecture diagram (ASCII or plain text; user can swap in image later)
- What this project demonstrates (CV + agentic AI + digital twin)
- Our segmentation pipeline section: quote metrics from pipeline/output/metrics.json
  if it exists (if it doesn't yet, leave a placeholder and note to fill in)
- Quick start (5 numbered steps)
- Kubernetes demo section
- v2 roadmap bullets
- Credits section: InteriorGS, Gaussian Grouping, @mkkellogg, open-clip, MobileSAM
- Demo video placeholder (to be filled at the end)
Commit: "docs: flesh out README with pipeline metrics + credits"

### D-4: Architecture diagram in README (0:15)
Convert the architecture diagram from the design spec (§4.1) into an ASCII block
inside README. Keep it simple but complete.
Commit: "docs: add architecture diagram to README"

REPORT when done.
```

---

## Final integration (T+5:00 → T+6:00, Main session)

### Int-1: End-to-end smoke test

```bash
# Ensure all components are ready
docker compose up -d --build
sleep 10
curl http://localhost:8000/health
curl http://localhost:8000/scenes/demo
curl http://localhost:8000/scenes/demo/objects | jq length
curl -X POST http://localhost:8000/scenes/demo/notes/seed-match
curl http://localhost:8000/scenes/demo/hotspots | jq length
curl -X POST http://localhost:8000/scenes/demo/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"List any issues I should know about"}'
```

Each should return expected shape. Fix integration issues on the fly.

Open browser: `http://localhost:3000` → verify homepage links work.
Open: `http://localhost:3000/listing/demo` → splat loads, hotspots visible, chat works.
Open: `http://localhost:3000/listing/demo/admin` → scrubber video plays, add note, hotspot appears.

### Int-2: Record demo video

Shot list (2–3 min):
1. 10s: Zillow screenshot, voiceover: "Zillow is 30 static photos. We fixed it."
2. 10s: Title "SceneAgent — AI-concierge 3D real estate listings"
3. 15s: Admin screen — scrub to different times, type 3 notes, hotspots appear on the right
4. 15s: Buyer listing opens, orbit scene, click 2 hotspots (one 🎁, one ⚠️)
5. 40s: Chat: "Any issues?" → agent lists them and highlights. "How tall are ceilings?" → agent measures. "Tour the features" → guided camera fly.
6. 15s: Terminal — `./k8s/apply.sh`, pods green, port-forward, browser reloads.
7. 15s: Show `cat pipeline/output/metrics.json` — "Real segmentation pipeline: P=X R=Y F1=Z against InteriorGS ground truth."
8. 10s: End card — GitHub URL, tech stack badges.

Record with OBS, rough-cut in DaVinci Resolve, voiceover, export 1080p30, upload to YouTube unlisted.

Update README with demo video link. Commit & push.

### Int-3: Push to GitHub

```bash
git push origin main
```

---

## Definition of Done

1. `docker compose up` runs clean end-to-end.
2. Homepage → listing → admin all functional.
3. At least one real `pipeline/output/metrics.json` file exists (even if numbers are modest).
4. README reports the pipeline metrics honestly.
5. `kubectl apply -f k8s/` brings up the stack on minikube.
6. GitHub Actions green.
7. Demo video uploaded & linked in README.
8. Repo pushed to GitHub.

**Ship it.**

---

## Risk register + fallbacks

| If this breaks | Do this |
|---|---|
| Puppeteer + @mkkellogg headless render fails | Use `gsplat` Python lib (`pip install gsplat`) to rasterize. Or convert .ply to standard 3DGS format via `@mkkellogg/gaussian-splats-3d`'s CLI first. |
| MobileSAM can't download weights | Use `facebook/sam-vit-base` via transformers library as fallback — slower but Python-native. |
| Whole segmentation pipeline takes > 3 hours | Cut to 15 views instead of 30; cut classes to 20 instead of 40. If still slow, use GT labels for product; keep whatever partial metrics we have for README. |
| Splat doesn't load in @mkkellogg viewer (compressed format issue) | Convert offline: download a non-compressed scene or decompress via `gaussian-splats-3d`'s included converter. |
| Gemini API rate limits mid-demo | Fall back to Groq free tier for text-only responses; VLM grounding degrades to text-only. |
| pgvector extension missing | Confirm we're using `pgvector/pgvector:pg16` image specifically. |

**Track M has the highest risk. If Track M falls behind, fall back to GT-based seed_db and keep moving — we still ship a product, and the README honestly documents the attempt.**
