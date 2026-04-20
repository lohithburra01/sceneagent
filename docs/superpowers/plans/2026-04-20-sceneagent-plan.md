# SceneAgent Implementation Plan

> **How to execute this plan:** Work through phases in order. Each task is sized so you can paste it into Claude Code and have it implemented in one shot. After each task: run the "Verify" command, inspect the result, then `git commit` with the suggested message. Don't skip verification — a task isn't done until the verify step passes.

**Goal:** Ship the v1 MVP described in `docs/superpowers/specs/2026-04-20-sceneagent-design.md` in 1–2 days.

**Architecture:** Next.js viewer + FastAPI backend + Postgres/pgvector + LangGraph agent + MCP server. Scene sourced from InteriorGS (no GPU training). All services Dockerized, K8s manifests for minikube demo.

**Tech Stack:** Python 3.11, Node.js 20, Next.js 14, Three.js, @mkkellogg/gaussian-splats-3d, FastAPI, LangGraph, Python MCP SDK, OpenCLIP, PostgreSQL 16 + pgvector, Redis 7, Docker Compose, Kubernetes (minikube), GitHub Actions.

**Execution order:**
- Phase 0: Repo + scene prep (scripts + data)
- Phase 1: Infrastructure (Docker Compose, Postgres)
- Phase 2: API skeleton (FastAPI + DB load)
- Phase 3: Note matcher + hotspot creation
- Phase 4: MCP server
- Phase 5: LangGraph agent + VLM grounding + SSE
- Phase 6: Frontend (viewer + chat + scrubber)
- Phase 7: Deployment polish (K8s, CI, README)
- Phase 8: Demo video

---

## Phase 0 — Repo & Scene Preparation

### Task 0.1: Repo scaffolding

**Files:**
- Create: `.gitignore`
- Create: `.env.example`
- Create: `README.md` (stub)

- [ ] **Step 1: Create `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
.venv/
*.egg-info/

# Node
node_modules/
.next/
dist/
*.log

# Data (the InteriorGS scene is large; fetched via script)
data/scene/demo/*
!data/scene/demo/demo_notes.json

# Env
.env
.env.local

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Model weights
*.pt
*.pth
*.onnx
```

- [ ] **Step 2: Create `.env.example`**

```bash
# LLM
GEMINI_API_KEY=your_gemini_api_key_here
LLM_MODEL=gemini-2.0-flash
VLM_MODEL=gemini-2.0-flash

# Database
DATABASE_URL=postgresql://sceneagent:sceneagent@postgres:5432/sceneagent
REDIS_URL=redis://redis:6379/0

# HuggingFace (for InteriorGS download)
HF_TOKEN=your_hf_token_here

# Scene
SCENE_ID=demo
```

- [ ] **Step 3: Create `README.md` stub**

```markdown
# SceneAgent

AI-concierge real-estate listings powered by 3D Gaussian Splatting + semantic segmentation + MCP-tool-using agents.

See `docs/superpowers/specs/2026-04-20-sceneagent-design.md` for the full design.

## Quick start

1. `cp .env.example .env` and fill in keys.
2. `./scripts/download_scene.sh` — one-off, downloads an InteriorGS scene.
3. `python scripts/prepare_scene.py` — preps data for the app.
4. `docker compose up --build` — starts everything.
5. Visit <http://localhost:3000/listing/demo>.
```

- [ ] **Step 4: Commit**

```bash
git add .gitignore .env.example README.md
git commit -m "chore: scaffold .gitignore, env template, readme stub"
```

---

### Task 0.2: HuggingFace scene download script

**Files:**
- Create: `scripts/download_scene.sh`

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# Download one InteriorGS scene from Hugging Face.
# Requires: huggingface-cli logged in (HF_TOKEN env var OR `huggingface-cli login`)
# Requires: user has accepted the InteriorGS dataset terms at
#   https://huggingface.co/datasets/spatialverse/InteriorGS

set -euo pipefail

SCENE_SUBDIR="${1:-}"
DEST="data/scene/demo"

if [[ -z "${SCENE_SUBDIR}" ]]; then
  echo "Usage: $0 <scene_subpath_within_interiorgs_repo>"
  echo "Example: $0 scenes/0001"
  echo ""
  echo "Browse available scenes at:"
  echo "  https://huggingface.co/datasets/spatialverse/InteriorGS/tree/main"
  exit 1
fi

mkdir -p "${DEST}"

echo "Downloading ${SCENE_SUBDIR} from spatialverse/InteriorGS..."
huggingface-cli download spatialverse/InteriorGS \
  --repo-type dataset \
  --include "${SCENE_SUBDIR}/*" \
  --local-dir "./_hf_cache"

# Move the 5 expected files into DEST (flatten the subpath)
for f in 3dgs_compressed.ply labels.json structure.json occupancy.png occupancy.json; do
  src="./_hf_cache/${SCENE_SUBDIR}/${f}"
  if [[ ! -f "${src}" ]]; then
    echo "ERROR: expected file not found: ${src}"
    exit 1
  fi
  cp "${src}" "${DEST}/${f}"
done

echo "Scene downloaded to ${DEST}/"
ls -lh "${DEST}"
```

- [ ] **Step 2: Make executable and commit**

```bash
chmod +x scripts/download_scene.sh
git add scripts/download_scene.sh
git commit -m "feat(scripts): InteriorGS scene downloader"
```

- [ ] **Step 3: Run it (manual one-off)**

```bash
# First time: huggingface-cli login (paste HF token)
# Accept InteriorGS license on the dataset page (browser)
./scripts/download_scene.sh scenes/0001
```

**Verify:** `ls data/scene/demo/` shows all 5 files (`3dgs_compressed.ply`, `labels.json`, `structure.json`, `occupancy.png`, `occupancy.json`).

> **If the scene subpath `scenes/0001` doesn't exist, browse the repo on HuggingFace and pick any valid scene ID. The script supports any subpath.**

---

### Task 0.3: Scene prep — parse labels.json, write object inventory

**Files:**
- Create: `scripts/prepare_scene.py`
- Create: `scripts/pyproject.toml`
- Create: `scripts/tests/test_parse_labels.py`

- [ ] **Step 1: Create `scripts/pyproject.toml`**

```toml
[project]
name = "sceneagent-scripts"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "numpy>=1.26",
  "pillow>=10.0",
  "shapely>=2.0",
  "open-clip-torch>=2.24",
  "torch>=2.2",
  "torchvision>=0.17",
  "playwright>=1.42",
  "ffmpeg-python>=0.2",
]

[tool.ruff]
line-length = 100
```

- [ ] **Step 2: Write the failing test `scripts/tests/test_parse_labels.py`**

```python
import json
from pathlib import Path
from scripts.prepare_scene import parse_labels_json, ObjectRecord


def test_parse_labels_extracts_instances(tmp_path):
    labels_data = {
        "objects": [
            {
                "instance_id": 0,
                "class_name": "chair",
                "bbox_corners": [
                    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
                ],
            },
            {
                "instance_id": 1,
                "class_name": "window",
                "bbox_corners": [
                    [5, 5, 0], [6, 5, 0], [6, 6, 0], [5, 6, 0],
                    [5, 5, 2], [6, 5, 2], [6, 6, 2], [5, 6, 2],
                ],
            },
        ]
    }
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(labels_data))

    result = parse_labels_json(labels_path)

    assert len(result) == 2
    assert all(isinstance(r, ObjectRecord) for r in result)
    chair = result[0]
    assert chair.instance_id == 0
    assert chair.class_name == "chair"
    assert chair.bbox_min == (0.0, 0.0, 0.0)
    assert chair.bbox_max == (1.0, 1.0, 1.0)
    assert chair.centroid == (0.5, 0.5, 0.5)
```

- [ ] **Step 3: Run to see it fail**

```bash
cd scripts && python -m pytest tests/test_parse_labels.py -v
```

Expected: `ModuleNotFoundError: No module named 'scripts.prepare_scene'`

- [ ] **Step 4: Write `scripts/prepare_scene.py` with just the parser**

```python
"""Scene preparation for SceneAgent.

Reads InteriorGS scene files, produces:
  data/scene/demo/object_inventory.json
  data/scene/demo/camera_trajectory.json
  data/scene/demo/views/<instance_id>.jpg
  data/scene/demo/video.mp4
  data/scene/demo/demo_notes.json (if not already present)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ObjectRecord:
    instance_id: int
    class_name: str
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    centroid: tuple[float, float, float]
    room_label: str | None = None


def _bbox_from_corners(corners: list[list[float]]) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
]:
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    zs = [c[2] for c in corners]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def parse_labels_json(labels_path: Path) -> list[ObjectRecord]:
    data = json.loads(labels_path.read_text())
    out: list[ObjectRecord] = []
    for obj in data["objects"]:
        bbox_min, bbox_max = _bbox_from_corners(obj["bbox_corners"])
        centroid = tuple((a + b) / 2.0 for a, b in zip(bbox_min, bbox_max))
        out.append(ObjectRecord(
            instance_id=obj["instance_id"],
            class_name=obj["class_name"],
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            centroid=centroid,  # type: ignore[arg-type]
        ))
    return out


def write_object_inventory(records: Iterable[ObjectRecord], out_path: Path) -> None:
    out_path.write_text(json.dumps([asdict(r) for r in records], indent=2))
```

- [ ] **Step 5: Run test, expect pass**

```bash
cd scripts && python -m pytest tests/test_parse_labels.py -v
```

- [ ] **Step 6: Commit**

```bash
git add scripts/prepare_scene.py scripts/pyproject.toml scripts/tests/
git commit -m "feat(scripts): parse InteriorGS labels.json to ObjectRecord list"
```

---

### Task 0.4: Scene prep — room assignment via floorplan

**Files:**
- Modify: `scripts/prepare_scene.py`
- Create: `scripts/tests/test_room_assignment.py`

- [ ] **Step 1: Write failing test**

```python
from scripts.prepare_scene import assign_rooms, ObjectRecord
from shapely.geometry import Polygon


def test_assign_rooms_matches_by_2d_containment():
    room_polys = {
        "living_room": Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]),
        "bedroom":     Polygon([(5, 0), (10, 0), (10, 5), (5, 5)]),
    }
    objs = [
        ObjectRecord(0, "sofa",  (1, 1, 0), (2, 2, 1), (1.5, 1.5, 0.5)),
        ObjectRecord(1, "bed",   (6, 1, 0), (8, 3, 0.5), (7.0, 2.0, 0.25)),
        ObjectRecord(2, "floor", (-1, -1, 0), (0, 0, 0), (-0.5, -0.5, 0)),  # outside any room
    ]

    result = assign_rooms(objs, room_polys)

    by_id = {r.instance_id: r for r in result}
    assert by_id[0].room_label == "living_room"
    assert by_id[1].room_label == "bedroom"
    assert by_id[2].room_label is None
```

- [ ] **Step 2: Run to fail**

```bash
cd scripts && python -m pytest tests/test_room_assignment.py -v
```

- [ ] **Step 3: Implement `assign_rooms` in `prepare_scene.py`**

Append:

```python
from shapely.geometry import Point, Polygon


def parse_structure_json(structure_path: Path) -> dict[str, Polygon]:
    """Extract room polygons from InteriorGS structure.json.

    Returns map of room_label → 2D Polygon (x,y plane).
    """
    data = json.loads(structure_path.read_text())
    rooms: dict[str, Polygon] = {}
    for room in data.get("rooms", []):
        label = room.get("type") or room.get("label") or f"room_{room.get('id', 'unknown')}"
        # polygon may be under "polygon" or "vertices"
        verts = room.get("polygon") or room.get("vertices")
        if not verts:
            continue
        rooms[label] = Polygon([(v[0], v[1]) for v in verts])
    return rooms


def assign_rooms(
    objects: list[ObjectRecord],
    room_polygons: dict[str, Polygon],
) -> list[ObjectRecord]:
    out = []
    for obj in objects:
        cx, cy, _ = obj.centroid
        pt = Point(cx, cy)
        label = None
        for room_label, poly in room_polygons.items():
            if poly.contains(pt):
                label = room_label
                break
        out.append(ObjectRecord(
            instance_id=obj.instance_id,
            class_name=obj.class_name,
            bbox_min=obj.bbox_min,
            bbox_max=obj.bbox_max,
            centroid=obj.centroid,
            room_label=label,
        ))
    return out
```

- [ ] **Step 4: Run test, expect pass**

```bash
cd scripts && python -m pytest tests/test_room_assignment.py -v
```

- [ ] **Step 5: Commit**

```bash
git add scripts/prepare_scene.py scripts/tests/test_room_assignment.py
git commit -m "feat(scripts): assign room labels by 2D point-in-polygon"
```

---

### Task 0.5: Scene prep — canonical object views via headless Three.js

**Files:**
- Create: `scripts/render_views.mjs`
- Modify: `scripts/prepare_scene.py` (invoke the Node renderer)

- [ ] **Step 1: Write `scripts/render_views.mjs`** (Node + Puppeteer/Playwright to render the splat headlessly is complex; use a simpler Python-only approach instead — render object bboxes as colored 3D boxes using matplotlib 3D, for CLIP classification. Good enough because InteriorGS already gives us class names.)

**Simpler alternative: skip real splat rendering. Since InteriorGS already provides `class_name`, the CLIP embedding per object only needs to represent the note-matching text space. Compute CLIP _text_ embeddings from the class names and a short description template.**

Replace Step 1 with:

- [ ] **Step 1 (revised): Add CLIP text-embedding helper to `prepare_scene.py`**

Append:

```python
def compute_object_embeddings(
    objects: list[ObjectRecord],
) -> dict[int, list[float]]:
    """Compute one CLIP text embedding per object.

    The embedding represents the object as a short descriptive string
    like 'a chair in the living_room'. This is what the note matcher
    ranks against.
    """
    import torch
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()

    def describe(o: ObjectRecord) -> str:
        room = o.room_label or "room"
        return f"a {o.class_name} in the {room}"

    texts = [describe(o) for o in objects]
    tokens = tokenizer(texts)
    with torch.no_grad():
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return {o.instance_id: emb[i].tolist() for i, o in enumerate(objects)}
```

- [ ] **Step 2: Quick integration test**

```python
# scripts/tests/test_embeddings.py
def test_compute_object_embeddings_shapes():
    from scripts.prepare_scene import ObjectRecord, compute_object_embeddings
    objs = [
        ObjectRecord(0, "chair", (0,0,0), (1,1,1), (0.5,0.5,0.5), "living_room"),
        ObjectRecord(1, "bed",   (5,5,0), (7,7,1), (6,6,0.5),     "bedroom"),
    ]
    emb = compute_object_embeddings(objs)
    assert set(emb.keys()) == {0, 1}
    assert all(len(v) == 512 for v in emb.values())
    # unit norm
    import math
    for v in emb.values():
        norm = math.sqrt(sum(x*x for x in v))
        assert abs(norm - 1.0) < 1e-4
```

- [ ] **Step 3: Run test**

```bash
cd scripts && python -m pytest tests/test_embeddings.py -v
```

(First run will download CLIP weights, ~600 MB. Takes 2–3 min.)

- [ ] **Step 4: Commit**

```bash
git add scripts/prepare_scene.py scripts/tests/test_embeddings.py
git commit -m "feat(scripts): CLIP text embeddings per object for note matching"
```

---

### Task 0.6: Scene prep — synthesize walkthrough video for scrubber UI

**Files:**
- Modify: `scripts/prepare_scene.py`
- Create: `scripts/synthesize_walkthrough.py`

- [ ] **Step 1: Create `scripts/synthesize_walkthrough.py`**

Pragmatic shortcut: generate a simple waypointed camera path by visiting each room's centroid in order, render basic textured boxes (walls + objects as colored cubes) with matplotlib 3D at 10fps for 30s total, encode to mp4. This produces a "walkthrough" that the scrubber UI can scrub against — it's clearly a preview, not photoreal, and the README acknowledges this.

```python
"""Generate a synthesized walkthrough video from the scene.

Renders a simple 3D walkthrough (boxes per object, colored by class)
at 10 fps, producing video.mp4 + camera_trajectory.json.
"""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch  # noqa: F401
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from prepare_scene import ObjectRecord


FPS = 10
DURATION_SEC = 30
TOTAL_FRAMES = FPS * DURATION_SEC


def _waypoints_for_rooms(objects: list[ObjectRecord]) -> list[tuple[float, float, float]]:
    """One waypoint per room centroid, ordered by x then y."""
    from collections import defaultdict
    by_room: dict[str, list[ObjectRecord]] = defaultdict(list)
    for o in objects:
        if o.room_label:
            by_room[o.room_label].append(o)
    waypoints = []
    for room_label in sorted(by_room.keys()):
        pts = [o.centroid for o in by_room[room_label]]
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        waypoints.append((cx, cy, 1.7))  # eye-level height
    return waypoints or [(0, 0, 1.7)]


def _camera_path(waypoints: list[tuple[float, float, float]], n_frames: int) -> list[dict]:
    """Linearly interpolate position between waypoints; each waypoint rotates 360° once."""
    n_segments = len(waypoints)
    frames_per_segment = n_frames // max(n_segments, 1)
    path = []
    t_abs = 0
    for i, wp in enumerate(waypoints):
        next_wp = waypoints[(i + 1) % len(waypoints)]
        for k in range(frames_per_segment):
            alpha = k / frames_per_segment
            pos = tuple(wp[j] * (1 - alpha) + next_wp[j] * alpha for j in range(3))
            yaw_deg = (k / frames_per_segment) * 360
            path.append({
                "timestamp": t_abs / FPS,
                "position": list(pos),
                "yaw_deg": yaw_deg,
            })
            t_abs += 1
    return path


def _render_frame(
    frame_dir: Path, idx: int, cam: dict, objects: list[ObjectRecord]
) -> None:
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    for o in objects:
        xs = [o.bbox_min[0], o.bbox_max[0]]
        ys = [o.bbox_min[1], o.bbox_max[1]]
        zs = [o.bbox_min[2], o.bbox_max[2]]
        ax.bar3d(
            xs[0], ys[0], zs[0],
            xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0],
            shade=True, alpha=0.5,
        )
    ax.view_init(elev=10, azim=cam["yaw_deg"])
    cx, cy, cz = cam["position"]
    ax.set_xlim(cx - 5, cx + 5)
    ax.set_ylim(cy - 5, cy + 5)
    ax.set_zlim(0, 3)
    ax.axis("off")
    fig.savefig(frame_dir / f"frame_{idx:05d}.png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def synthesize(
    objects: list[ObjectRecord], out_dir: Path
) -> None:
    frame_dir = out_dir / "_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    waypoints = _waypoints_for_rooms(objects)
    camera_path = _camera_path(waypoints, TOTAL_FRAMES)

    for i, cam in enumerate(camera_path):
        _render_frame(frame_dir, i, cam, objects)

    (out_dir / "camera_trajectory.json").write_text(json.dumps(camera_path, indent=2))

    # Encode with ffmpeg
    video_path = out_dir / "video.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(FPS),
        "-i", str(frame_dir / "frame_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(video_path),
    ], check=True)

    # Clean up
    for f in frame_dir.iterdir():
        f.unlink()
    frame_dir.rmdir()
```

- [ ] **Step 2: Wire it into `prepare_scene.py` as `main()`**

Append to `prepare_scene.py`:

```python
def main(scene_dir: Path = Path("data/scene/demo")) -> None:
    labels = parse_labels_json(scene_dir / "labels.json")
    rooms = parse_structure_json(scene_dir / "structure.json")
    labels_with_rooms = assign_rooms(labels, rooms)

    write_object_inventory(labels_with_rooms, scene_dir / "object_inventory.json")

    embeddings = compute_object_embeddings(labels_with_rooms)
    (scene_dir / "object_embeddings.json").write_text(json.dumps(embeddings))

    from synthesize_walkthrough import synthesize
    synthesize(labels_with_rooms, scene_dir)

    print(f"Scene prepared at {scene_dir}")


if __name__ == "__main__":
    import sys
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/scene/demo"))
```

- [ ] **Step 3: Run end-to-end (manual)**

```bash
cd scripts
pip install -e .
python prepare_scene.py ../data/scene/demo
```

**Verify:** `data/scene/demo/` now contains `object_inventory.json`, `object_embeddings.json`, `camera_trajectory.json`, `video.mp4`.

- [ ] **Step 4: Commit**

```bash
git add scripts/synthesize_walkthrough.py scripts/prepare_scene.py
git commit -m "feat(scripts): synthesize walkthrough video + camera trajectory from InteriorGS scene"
```

---

### Task 0.7: Seed demo notes

**Files:**
- Create: `data/scene/demo/demo_notes.json` (this one IS committed — tiny, reproducible)

- [ ] **Step 1: Hand-write 8 demo notes**

```json
[
  {"text": "The window in the bedroom sticks, you need to pull firmly", "video_timestamp": 3.2},
  {"text": "This desk comes with the apartment, it's included", "video_timestamp": 8.5},
  {"text": "Ceilings are 3.2 meters in the living room", "video_timestamp": 12.1},
  {"text": "Heated bathroom floor, very nice in winter", "video_timestamp": 18.4},
  {"text": "WiFi router is behind the TV in the living room", "video_timestamp": 14.7},
  {"text": "Washing machine is included and relatively new", "video_timestamp": 22.0},
  {"text": "The radiator near the couch is a bit loud in the morning", "video_timestamp": 16.3},
  {"text": "This building was originally a bakery in the 1920s", "video_timestamp": 25.5}
]
```

- [ ] **Step 2: Update `.gitignore` exception**

Already handled in Task 0.1 (`!data/scene/demo/demo_notes.json`).

- [ ] **Step 3: Commit**

```bash
git add data/scene/demo/demo_notes.json
git commit -m "data: seed demo notes with timestamps for scrubber walkthrough"
```

---

## Phase 1 — Infrastructure

### Task 1.1: Docker Compose skeleton

**Files:**
- Create: `docker-compose.yml`
- Create: `db/init/001_schema.sql`
- Create: `api/Dockerfile` (stub)
- Create: `web/Dockerfile` (stub)

- [ ] **Step 1: Write `docker-compose.yml`**

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: sceneagent
      POSTGRES_PASSWORD: sceneagent
      POSTGRES_DB: sceneagent
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./db/init:/docker-entrypoint-initdb.d:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U sceneagent"]
      interval: 5s
      timeout: 5s
      retries: 10

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  api:
    build: ./api
    environment:
      DATABASE_URL: postgresql://sceneagent:sceneagent@postgres:5432/sceneagent
      REDIS_URL: redis://redis:6379/0
      GEMINI_API_KEY: ${GEMINI_API_KEY}
      SCENE_ID: ${SCENE_ID:-demo}
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data:ro
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started

  web:
    build: ./web
    environment:
      NEXT_PUBLIC_API_URL: http://localhost:8000
    ports:
      - "3000:3000"
    depends_on:
      - api

volumes:
  pgdata:
```

- [ ] **Step 2: Write `db/init/001_schema.sql`**

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS scenes (
    id               UUID PRIMARY KEY,
    slug             TEXT UNIQUE NOT NULL,
    title            TEXT NOT NULL,
    address          TEXT,
    splat_url        TEXT NOT NULL,
    camera_trajectory JSONB NOT NULL,
    processed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS scene_objects (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    scene_id        UUID REFERENCES scenes(id) ON DELETE CASCADE,
    instance_id     INT NOT NULL,
    class_name      TEXT NOT NULL,
    room_label      TEXT,
    centroid        DOUBLE PRECISION[] NOT NULL,
    bbox_min        DOUBLE PRECISION[] NOT NULL,
    bbox_max        DOUBLE PRECISION[] NOT NULL,
    clip_embedding  VECTOR(512) NOT NULL,
    UNIQUE (scene_id, instance_id)
);
CREATE INDEX IF NOT EXISTS idx_scene_objects_embedding
    ON scene_objects USING hnsw (clip_embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_scene_objects_scene ON scene_objects (scene_id);

CREATE TABLE IF NOT EXISTS notes (
    id                   UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    scene_id             UUID REFERENCES scenes(id) ON DELETE CASCADE,
    text                 TEXT NOT NULL,
    video_timestamp      DOUBLE PRECISION NOT NULL,
    category             TEXT,
    category_confidence  REAL,
    created_at           TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS hotspots (
    id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    note_id           UUID UNIQUE REFERENCES notes(id) ON DELETE CASCADE,
    object_id         UUID REFERENCES scene_objects(id) ON DELETE SET NULL,
    match_confidence  REAL NOT NULL,
    position          DOUBLE PRECISION[] NOT NULL,
    auto_accepted     BOOLEAN NOT NULL DEFAULT TRUE
);
```

- [ ] **Step 3: Write stub Dockerfiles**

`api/Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn
COPY . .
CMD ["uvicorn", "src.sceneagent.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

`web/Dockerfile`:
```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install || true
COPY . .
CMD ["npm", "run", "dev"]
```

- [ ] **Step 4: Verify postgres starts and has schema**

```bash
docker compose up -d postgres
docker compose exec postgres psql -U sceneagent -d sceneagent -c '\dt'
# Should list scenes, scene_objects, notes, hotspots
docker compose down
```

- [ ] **Step 5: Commit**

```bash
git add docker-compose.yml db/init/001_schema.sql api/Dockerfile web/Dockerfile
git commit -m "feat(infra): docker-compose with pgvector postgres, redis, api+web stubs, initial schema"
```

---

## Phase 2 — API Skeleton

### Task 2.1: FastAPI project scaffold

**Files:**
- Create: `api/pyproject.toml`
- Create: `api/src/sceneagent/__init__.py`
- Create: `api/src/sceneagent/main.py`
- Create: `api/src/sceneagent/db.py`
- Create: `api/src/sceneagent/models.py`

- [ ] **Step 1: `api/pyproject.toml`**

```toml
[project]
name = "sceneagent-api"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "fastapi>=0.109",
  "uvicorn[standard]>=0.27",
  "pydantic>=2.5",
  "asyncpg>=0.29",
  "pgvector>=0.2",
  "google-generativeai>=0.4",
  "langgraph>=0.0.40",
  "mcp>=1.0",
  "numpy>=1.26",
  "open-clip-torch>=2.24",
  "torch>=2.2",
  "pillow>=10.0",
  "httpx>=0.26",
  "sse-starlette>=1.8",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-asyncio>=0.23", "ruff>=0.2", "mypy>=1.8"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 100

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

- [ ] **Step 2: `api/src/sceneagent/main.py`**

```python
from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI

from .db import init_db_pool, close_db_pool, seed_demo_scene


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db_pool()
    await seed_demo_scene()
    yield
    await close_db_pool()


app = FastAPI(title="SceneAgent API", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}
```

- [ ] **Step 3: `api/src/sceneagent/db.py`**

```python
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import asyncpg
from pgvector.asyncpg import register_vector

_pool: asyncpg.Pool | None = None

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://sceneagent:sceneagent@localhost:5432/sceneagent",
)
DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))
SCENE_ID_SLUG = os.environ.get("SCENE_ID", "demo")


async def init_db_pool() -> None:
    global _pool
    _pool = await asyncpg.create_pool(
        DATABASE_URL,
        min_size=1, max_size=10,
        init=register_vector,
    )


async def close_db_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()


def pool() -> asyncpg.Pool:
    assert _pool is not None, "db pool not initialized"
    return _pool


async def seed_demo_scene() -> None:
    """Idempotent: load object_inventory + camera_trajectory + demo_notes
    into the DB if the scene slug doesn't already exist."""
    scene_dir = DATA_DIR / "scene" / SCENE_ID_SLUG
    inv_path = scene_dir / "object_inventory.json"
    emb_path = scene_dir / "object_embeddings.json"
    traj_path = scene_dir / "camera_trajectory.json"
    notes_path = scene_dir / "demo_notes.json"

    if not inv_path.exists():
        print(f"[seed] inventory not found at {inv_path}, skipping seed")
        return

    async with pool().acquire() as con:
        row = await con.fetchrow(
            "SELECT id FROM scenes WHERE slug = $1", SCENE_ID_SLUG
        )
        if row is not None:
            print(f"[seed] scene '{SCENE_ID_SLUG}' already loaded, skipping")
            return

        scene_id = uuid.uuid4()
        trajectory = json.loads(traj_path.read_text())
        await con.execute(
            """INSERT INTO scenes (id, slug, title, splat_url, camera_trajectory)
               VALUES ($1, $2, $3, $4, $5)""",
            scene_id, SCENE_ID_SLUG,
            f"Demo Listing ({SCENE_ID_SLUG})",
            f"/static/scene/{SCENE_ID_SLUG}/3dgs_compressed.ply",
            json.dumps(trajectory),
        )

        inventory = json.loads(inv_path.read_text())
        embeddings = json.loads(emb_path.read_text())
        for obj in inventory:
            iid = obj["instance_id"]
            emb = embeddings[str(iid)]
            await con.execute(
                """INSERT INTO scene_objects
                   (scene_id, instance_id, class_name, room_label,
                    centroid, bbox_min, bbox_max, clip_embedding)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8)""",
                scene_id, iid, obj["class_name"], obj.get("room_label"),
                list(obj["centroid"]),
                list(obj["bbox_min"]),
                list(obj["bbox_max"]),
                emb,
            )

        # seed demo notes
        if notes_path.exists():
            notes = json.loads(notes_path.read_text())
            for n in notes:
                await con.execute(
                    """INSERT INTO notes (scene_id, text, video_timestamp)
                       VALUES ($1, $2, $3)""",
                    scene_id, n["text"], n["video_timestamp"],
                )

        print(f"[seed] loaded scene '{SCENE_ID_SLUG}' with "
              f"{len(inventory)} objects and {len(notes) if notes_path.exists() else 0} notes")
```

- [ ] **Step 4: Update `api/Dockerfile` for real**

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
RUN pip install --no-cache-dir .

COPY src/ ./src/
ENV PYTHONPATH=/app/src
CMD ["uvicorn", "sceneagent.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 5: Verify health endpoint**

```bash
docker compose up -d postgres redis
cd api && pip install -e . && DATABASE_URL=postgresql://sceneagent:sceneagent@localhost:5432/sceneagent DATA_DIR=../data python -m uvicorn sceneagent.main:app --reload
# In another terminal:
curl http://localhost:8000/health
# → {"status":"ok"}
```

- [ ] **Step 6: Commit**

```bash
git add api/
git commit -m "feat(api): FastAPI scaffold with lifespan seed of scenes+objects+notes from data/"
```

---

### Task 2.2: Read-only scene endpoints

**Files:**
- Create: `api/src/sceneagent/routes/__init__.py`
- Create: `api/src/sceneagent/routes/scenes.py`
- Modify: `api/src/sceneagent/main.py` to include the router
- Create: `api/tests/test_scenes.py`

- [ ] **Step 1: Failing test `api/tests/test_scenes.py`**

```python
import httpx
import pytest


@pytest.mark.asyncio
async def test_get_scene_by_slug():
    async with httpx.AsyncClient(base_url="http://api:8000") as client:
        r = await client.get("/scenes/demo")
        assert r.status_code == 200
        data = r.json()
        assert data["slug"] == "demo"
        assert "splat_url" in data
        assert "camera_trajectory" in data


@pytest.mark.asyncio
async def test_list_scene_objects():
    async with httpx.AsyncClient(base_url="http://api:8000") as client:
        r = await client.get("/scenes/demo/objects")
        assert r.status_code == 200
        objects = r.json()
        assert isinstance(objects, list)
        assert len(objects) > 0
        assert "class_name" in objects[0]
```

(Tests run against the live docker-compose stack; run from a container or host with services up.)

- [ ] **Step 2: Implement `routes/scenes.py`**

```python
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from ..db import pool

router = APIRouter(prefix="/scenes", tags=["scenes"])


@router.get("/{slug}")
async def get_scene(slug: str):
    async with pool().acquire() as con:
        row = await con.fetchrow(
            "SELECT id, slug, title, address, splat_url, camera_trajectory "
            "FROM scenes WHERE slug = $1", slug,
        )
    if not row:
        raise HTTPException(404, "scene not found")
    return dict(row)


@router.get("/{slug}/objects")
async def list_scene_objects(slug: str, room: str | None = None):
    async with pool().acquire() as con:
        scene = await con.fetchrow("SELECT id FROM scenes WHERE slug = $1", slug)
        if not scene:
            raise HTTPException(404, "scene not found")
        query = (
            "SELECT id, instance_id, class_name, room_label, "
            "centroid, bbox_min, bbox_max FROM scene_objects "
            "WHERE scene_id = $1"
        )
        params: list = [scene["id"]]
        if room:
            query += " AND room_label = $2"
            params.append(room)
        rows = await con.fetch(query, *params)
    return [dict(r) for r in rows]
```

- [ ] **Step 3: Wire router into `main.py`**

```python
from .routes.scenes import router as scenes_router
app.include_router(scenes_router)
```

- [ ] **Step 4: Verify**

```bash
curl http://localhost:8000/scenes/demo | jq .slug
# "demo"
curl "http://localhost:8000/scenes/demo/objects?room=bedroom" | jq 'length'
# some integer > 0
```

- [ ] **Step 5: Commit**

```bash
git add api/src/sceneagent/routes/ api/src/sceneagent/main.py api/tests/
git commit -m "feat(api): scenes read endpoints + tests"
```

---

## Phase 3 — Note Matcher + Hotspot Creation

### Task 3.1: Frustum visibility check utility

**Files:**
- Create: `api/src/sceneagent/geometry.py`
- Create: `api/tests/test_geometry.py`

- [ ] **Step 1: Failing test**

```python
from sceneagent.geometry import is_point_visible_from_pose
import numpy as np


def test_point_in_front_of_camera_facing_forward_is_visible():
    # Camera at origin, looking along +X, point 5 units forward
    pose = {
        "position": [0.0, 0.0, 0.0],
        "forward": [1.0, 0.0, 0.0],
        "up": [0.0, 0.0, 1.0],
    }
    visible = is_point_visible_from_pose(
        point=np.array([5.0, 0.0, 0.0]),
        pose=pose, fov_deg=60.0, max_distance=10.0,
    )
    assert visible is True


def test_point_behind_camera_is_not_visible():
    pose = {
        "position": [0.0, 0.0, 0.0],
        "forward": [1.0, 0.0, 0.0],
        "up": [0.0, 0.0, 1.0],
    }
    visible = is_point_visible_from_pose(
        point=np.array([-5.0, 0.0, 0.0]),
        pose=pose, fov_deg=60.0, max_distance=10.0,
    )
    assert visible is False


def test_point_too_far_is_not_visible():
    pose = {
        "position": [0.0, 0.0, 0.0],
        "forward": [1.0, 0.0, 0.0],
        "up": [0.0, 0.0, 1.0],
    }
    visible = is_point_visible_from_pose(
        point=np.array([15.0, 0.0, 0.0]),
        pose=pose, fov_deg=60.0, max_distance=10.0,
    )
    assert visible is False
```

- [ ] **Step 2: Implement**

```python
from __future__ import annotations

import math
from typing import TypedDict

import numpy as np


class CameraPose(TypedDict):
    position: list[float]
    forward: list[float]
    up: list[float]


def is_point_visible_from_pose(
    point: np.ndarray,
    pose: CameraPose,
    fov_deg: float = 60.0,
    max_distance: float = 15.0,
) -> bool:
    pos = np.asarray(pose["position"], dtype=float)
    fwd = np.asarray(pose["forward"], dtype=float)
    fwd /= (np.linalg.norm(fwd) + 1e-9)

    to_point = point - pos
    dist = np.linalg.norm(to_point)
    if dist < 1e-6 or dist > max_distance:
        return False

    dir_to_point = to_point / dist
    cos_angle = float(np.dot(fwd, dir_to_point))
    half_fov_cos = math.cos(math.radians(fov_deg) / 2.0)
    return cos_angle >= half_fov_cos


def pose_from_yaw(position: list[float], yaw_deg: float) -> CameraPose:
    """Convert (pos, yaw_deg) → CameraPose with forward computed from yaw."""
    yaw = math.radians(yaw_deg)
    forward = [math.cos(yaw), math.sin(yaw), 0.0]
    return CameraPose(position=position, forward=forward, up=[0.0, 0.0, 1.0])
```

- [ ] **Step 3: Run test, expect pass**

```bash
cd api && pytest tests/test_geometry.py -v
```

- [ ] **Step 4: Commit**

```bash
git add api/src/sceneagent/geometry.py api/tests/test_geometry.py
git commit -m "feat(api): frustum visibility check + yaw→pose helper"
```

---

### Task 3.2: Note → hotspot matcher

**Files:**
- Create: `api/src/sceneagent/matcher.py`
- Create: `api/src/sceneagent/clip_util.py`
- Create: `api/tests/test_matcher.py`

- [ ] **Step 1: CLIP wrapper `clip_util.py`**

```python
from __future__ import annotations

import functools
import torch
import open_clip


@functools.lru_cache(maxsize=1)
def _load_model():
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    return model, tokenizer


def encode_text(text: str) -> list[float]:
    model, tokenizer = _load_model()
    tokens = tokenizer([text])
    with torch.no_grad():
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb[0].tolist()
```

- [ ] **Step 2: Matcher logic test `test_matcher.py`**

```python
import pytest
from sceneagent.matcher import rank_objects_for_note


@pytest.mark.asyncio
async def test_matcher_ranks_visible_objects_by_clip(monkeypatch):
    # Fake DB row objects
    objects = [
        {"id": "obj-1", "centroid": [1.0, 0.0, 0.5], "clip_embedding": [1.0, 0.0] + [0.0]*510},
        {"id": "obj-2", "centroid": [0.5, 0.0, 0.5], "clip_embedding": [0.0, 1.0] + [0.0]*510},
    ]
    pose = {"position": [0.0, 0.0, 0.5], "forward": [1.0, 0.0, 0.0], "up": [0.0, 0.0, 1.0]}
    # Fake CLIP: note text encodes as [1, 0, ...] — matches obj-1 perfectly
    monkeypatch.setattr(
        "sceneagent.matcher.encode_text",
        lambda text: [1.0, 0.0] + [0.0]*510,
    )

    ranked = rank_objects_for_note(
        note_text="the thing on the wall",
        pose=pose,
        objects=objects,
    )

    assert ranked[0]["id"] == "obj-1"
    assert ranked[0]["score"] > ranked[1]["score"]
```

- [ ] **Step 3: Implement `matcher.py`**

```python
from __future__ import annotations

import numpy as np

from .clip_util import encode_text
from .geometry import is_point_visible_from_pose


def rank_objects_for_note(
    note_text: str,
    pose: dict,
    objects: list[dict],
    fov_deg: float = 60.0,
    max_distance: float = 15.0,
) -> list[dict]:
    note_emb = np.array(encode_text(note_text))

    visible: list[dict] = []
    for o in objects:
        c = np.array(o["centroid"])
        if is_point_visible_from_pose(c, pose, fov_deg, max_distance):
            obj_emb = np.array(o["clip_embedding"])
            score = float(np.dot(note_emb, obj_emb))
            visible.append({**o, "score": score})

    visible.sort(key=lambda x: x["score"], reverse=True)
    return visible
```

- [ ] **Step 4: Run tests**

```bash
cd api && pytest tests/test_matcher.py -v
```

- [ ] **Step 5: Commit**

```bash
git add api/src/sceneagent/matcher.py api/src/sceneagent/clip_util.py api/tests/test_matcher.py
git commit -m "feat(api): CLIP-based note→object matcher with frustum prefilter"
```

---

### Task 3.3: LLM note categorizer

**Files:**
- Create: `api/src/sceneagent/categorizer.py`
- Create: `api/tests/test_categorizer.py`

- [ ] **Step 1: Failing test**

```python
import pytest
from sceneagent.categorizer import classify_category, CATEGORIES


@pytest.mark.asyncio
async def test_classify_category_handles_issue(monkeypatch):
    async def fake_call(text, candidates):
        return "issue", 0.9
    monkeypatch.setattr("sceneagent.categorizer._call_llm", fake_call)

    cat, conf = await classify_category("window sticks, pull firmly")
    assert cat == "issue"
    assert conf >= 0.5
```

- [ ] **Step 2: Implement**

```python
from __future__ import annotations

import json
import os
from typing import Literal

import google.generativeai as genai

Category = Literal["feature", "included", "issue", "info", "spec", "story", "other"]
CATEGORIES: list[Category] = ["feature", "included", "issue", "info", "spec", "story", "other"]

_MODEL = os.environ.get("LLM_MODEL", "gemini-2.0-flash")
_API_KEY = os.environ.get("GEMINI_API_KEY")
if _API_KEY:
    genai.configure(api_key=_API_KEY)


async def _call_llm(text: str, candidates: list[str]) -> tuple[str, float]:
    prompt = (
        "Classify this real-estate listing note into exactly one category from: "
        f"{', '.join(candidates)}. "
        "Reply with a JSON object: {\"category\": \"...\", \"confidence\": 0.0-1.0}. "
        f"Note: {text!r}"
    )
    model = genai.GenerativeModel(_MODEL)
    resp = await model.generate_content_async(
        prompt, generation_config={"response_mime_type": "application/json"}
    )
    try:
        parsed = json.loads(resp.text)
        cat = parsed.get("category", "other").lower()
        if cat not in candidates:
            cat = "other"
        conf = float(parsed.get("confidence", 0.5))
    except Exception:
        cat, conf = "other", 0.3
    return cat, conf


async def classify_category(text: str) -> tuple[str, float]:
    return await _call_llm(text, CATEGORIES)
```

- [ ] **Step 3: Run test**

```bash
cd api && pytest tests/test_categorizer.py -v
```

- [ ] **Step 4: Commit**

```bash
git add api/src/sceneagent/categorizer.py api/tests/test_categorizer.py
git commit -m "feat(api): Gemini-based note category classifier (issue/feature/included/info/spec/story)"
```

---

### Task 3.4: Notes + hotspots endpoints

**Files:**
- Create: `api/src/sceneagent/routes/notes.py`
- Create: `api/src/sceneagent/routes/hotspots.py`
- Modify: `api/src/sceneagent/main.py`

- [ ] **Step 1: `routes/notes.py`**

```python
from __future__ import annotations

import json
import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..db import pool
from ..matcher import rank_objects_for_note
from ..categorizer import classify_category
from ..geometry import pose_from_yaw

router = APIRouter(prefix="/scenes/{slug}/notes", tags=["notes"])


class NoteIn(BaseModel):
    text: str
    video_timestamp: float


async def _camera_pose_at(slug: str, t: float) -> dict:
    async with pool().acquire() as con:
        row = await con.fetchrow(
            "SELECT camera_trajectory FROM scenes WHERE slug = $1", slug,
        )
    if not row:
        raise HTTPException(404, "scene not found")
    trajectory = row["camera_trajectory"]
    if isinstance(trajectory, str):
        trajectory = json.loads(trajectory)
    # pick closest timestamp
    closest = min(trajectory, key=lambda p: abs(p["timestamp"] - t))
    return pose_from_yaw(closest["position"], closest["yaw_deg"])


@router.post("")
async def create_note(slug: str, payload: NoteIn):
    # 1. Insert the note
    async with pool().acquire() as con:
        scene = await con.fetchrow(
            "SELECT id FROM scenes WHERE slug = $1", slug,
        )
        if not scene:
            raise HTTPException(404, "scene not found")
        note_id = uuid.uuid4()
        await con.execute(
            "INSERT INTO notes (id, scene_id, text, video_timestamp) "
            "VALUES ($1,$2,$3,$4)",
            note_id, scene["id"], payload.text, payload.video_timestamp,
        )

        # 2. Classify category (async LLM call)
        category, cat_conf = await classify_category(payload.text)
        await con.execute(
            "UPDATE notes SET category=$1, category_confidence=$2 WHERE id=$3",
            category, cat_conf, note_id,
        )

        # 3. Fetch objects for matcher
        objs = await con.fetch(
            "SELECT id, centroid, clip_embedding FROM scene_objects "
            "WHERE scene_id = $1",
            scene["id"],
        )
        objs_dicts = [dict(r) for r in objs]

    # 4. Get camera pose and rank
    pose = await _camera_pose_at(slug, payload.video_timestamp)
    ranked = rank_objects_for_note(payload.text, pose, objs_dicts)

    hotspot_info = None
    if ranked:
        best = ranked[0]
        # Create hotspot
        async with pool().acquire() as con:
            obj_centroid = await con.fetchval(
                "SELECT centroid FROM scene_objects WHERE id = $1", best["id"],
            )
            hs_id = uuid.uuid4()
            await con.execute(
                "INSERT INTO hotspots (id, note_id, object_id, match_confidence, "
                "position, auto_accepted) VALUES ($1,$2,$3,$4,$5,$6)",
                hs_id, note_id, best["id"], best["score"],
                list(obj_centroid), True,
            )
            hotspot_info = {
                "id": str(hs_id), "position": list(obj_centroid),
                "confidence": best["score"],
            }

    return {
        "note_id": str(note_id),
        "category": category,
        "category_confidence": cat_conf,
        "hotspot": hotspot_info,
    }


@router.post("/seed-match")
async def match_all_unmatched(slug: str):
    """Matches all notes that don't yet have hotspots. Used after initial seed."""
    async with pool().acquire() as con:
        scene = await con.fetchrow("SELECT id FROM scenes WHERE slug=$1", slug)
        if not scene:
            raise HTTPException(404, "scene not found")
        note_rows = await con.fetch(
            """SELECT n.id, n.text, n.video_timestamp FROM notes n
               LEFT JOIN hotspots h ON h.note_id = n.id
               WHERE n.scene_id = $1 AND h.id IS NULL""",
            scene["id"],
        )
        objs = await con.fetch(
            "SELECT id, centroid, clip_embedding FROM scene_objects WHERE scene_id=$1",
            scene["id"],
        )
        objs_dicts = [dict(r) for r in objs]

    matched = 0
    for n in note_rows:
        pose = await _camera_pose_at(slug, n["video_timestamp"])
        ranked = rank_objects_for_note(n["text"], pose, objs_dicts)
        if not ranked:
            continue
        best = ranked[0]
        # categorize
        cat, conf = await classify_category(n["text"])
        async with pool().acquire() as con:
            obj_centroid = await con.fetchval(
                "SELECT centroid FROM scene_objects WHERE id=$1", best["id"],
            )
            await con.execute(
                "UPDATE notes SET category=$1, category_confidence=$2 WHERE id=$3",
                cat, conf, n["id"],
            )
            await con.execute(
                "INSERT INTO hotspots (note_id, object_id, match_confidence, "
                "position, auto_accepted) VALUES ($1,$2,$3,$4,$5)",
                n["id"], best["id"], best["score"], list(obj_centroid), True,
            )
        matched += 1
    return {"matched": matched}
```

- [ ] **Step 2: `routes/hotspots.py`**

```python
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from ..db import pool

router = APIRouter(prefix="/scenes/{slug}/hotspots", tags=["hotspots"])


@router.get("")
async def list_hotspots(slug: str, category: str | None = None):
    async with pool().acquire() as con:
        scene = await con.fetchrow("SELECT id FROM scenes WHERE slug = $1", slug)
        if not scene:
            raise HTTPException(404, "scene not found")
        query = """
            SELECT h.id, h.position, h.match_confidence,
                   n.text AS note_text, n.category, n.video_timestamp,
                   o.class_name, o.room_label
            FROM hotspots h
            JOIN notes n ON n.id = h.note_id
            LEFT JOIN scene_objects o ON o.id = h.object_id
            WHERE n.scene_id = $1
        """
        params: list = [scene["id"]]
        if category:
            query += " AND n.category = $2"
            params.append(category)
        rows = await con.fetch(query, *params)
    return [dict(r) for r in rows]
```

- [ ] **Step 3: Wire routers in `main.py`**

```python
from .routes.notes import router as notes_router
from .routes.hotspots import router as hotspots_router
app.include_router(notes_router)
app.include_router(hotspots_router)
```

- [ ] **Step 4: End-to-end manual verify**

```bash
docker compose up -d --build
curl -X POST http://localhost:8000/scenes/demo/notes/seed-match
# → {"matched": 8}
curl http://localhost:8000/scenes/demo/hotspots | jq length
# → 8
```

- [ ] **Step 5: Commit**

```bash
git add api/src/sceneagent/routes/notes.py api/src/sceneagent/routes/hotspots.py api/src/sceneagent/main.py
git commit -m "feat(api): notes POST + hotspots GET endpoints with matcher + categorizer wiring"
```

---

## Phase 4 — MCP Server

### Task 4.1: MCP server with scene-manipulation tools

**Files:**
- Create: `api/src/sceneagent/mcp_server.py`
- Create: `api/src/sceneagent/agent/tools.py` (same tool impls, callable directly)

- [ ] **Step 1: Define tools in `agent/tools.py`** (so LangGraph can import them directly, and MCP re-exposes the same functions)

```python
from __future__ import annotations

import json
import numpy as np
from typing import Any

from ..db import pool
from ..clip_util import encode_text


async def list_objects(
    scene_slug: str,
    room: str | None = None,
    class_name: str | None = None,
    limit: int = 50,
) -> list[dict]:
    async with pool().acquire() as con:
        scene = await con.fetchrow("SELECT id FROM scenes WHERE slug=$1", scene_slug)
        if not scene:
            return []
        q = (
            "SELECT instance_id, class_name, room_label, centroid, bbox_min, bbox_max "
            "FROM scene_objects WHERE scene_id = $1"
        )
        params: list = [scene["id"]]
        if room:
            q += f" AND room_label = ${len(params)+1}"
            params.append(room)
        if class_name:
            q += f" AND class_name = ${len(params)+1}"
            params.append(class_name)
        q += f" LIMIT {int(limit)}"
        rows = await con.fetch(q, *params)
    return [dict(r) for r in rows]


async def list_hotspots(
    scene_slug: str, category: str | None = None,
) -> list[dict]:
    async with pool().acquire() as con:
        scene = await con.fetchrow("SELECT id FROM scenes WHERE slug=$1", scene_slug)
        if not scene:
            return []
        q = """
            SELECT h.id::text AS id, h.position, h.match_confidence,
                   n.text AS note_text, n.category,
                   o.class_name, o.room_label
            FROM hotspots h JOIN notes n ON n.id=h.note_id
            LEFT JOIN scene_objects o ON o.id=h.object_id
            WHERE n.scene_id = $1
        """
        params: list = [scene["id"]]
        if category:
            q += " AND n.category = $2"
            params.append(category)
        rows = await con.fetch(q, *params)
    return [dict(r) for r in rows]


async def find_by_description(
    scene_slug: str, text: str, limit: int = 5,
) -> list[dict]:
    emb = encode_text(text)
    async with pool().acquire() as con:
        scene = await con.fetchrow("SELECT id FROM scenes WHERE slug=$1", scene_slug)
        if not scene:
            return []
        rows = await con.fetch(
            """SELECT id::text, class_name, room_label, centroid,
                      1.0 - (clip_embedding <=> $2) AS score
               FROM scene_objects
               WHERE scene_id = $1
               ORDER BY clip_embedding <=> $2
               LIMIT $3""",
            scene["id"], emb, limit,
        )
    return [dict(r) for r in rows]


async def measure_distance(
    scene_slug: str, point_a: list[float], point_b: list[float],
) -> dict:
    d = float(np.linalg.norm(np.asarray(point_a) - np.asarray(point_b)))
    return {"meters": round(d, 3)}


async def plan_tour(
    scene_slug: str, focus: str | None = None,
) -> list[dict]:
    """Return ordered list of camera stops with narration hints."""
    hs = await list_hotspots(scene_slug, category=focus)
    stops = []
    for h in hs[:6]:  # cap at 6 stops for a short tour
        stops.append({
            "position": list(h["position"]),
            "dwell_seconds": 4.0,
            "narration_hint": h["note_text"],
            "highlight_hotspot_id": h["id"],
        })
    return stops
```

- [ ] **Step 2: MCP server `mcp_server.py`**

```python
from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from .agent import tools

mcp = FastMCP("SceneAgent")


@mcp.tool()
async def list_objects_tool(
    scene_slug: str,
    room: str | None = None,
    class_name: str | None = None,
) -> list[dict]:
    """List objects in a scene, optionally filtered by room or class."""
    return await tools.list_objects(scene_slug, room=room, class_name=class_name)


@mcp.tool()
async def list_hotspots_tool(
    scene_slug: str, category: str | None = None,
) -> list[dict]:
    """List hotspots in a scene, optionally filtered by category
    (feature/included/issue/info/spec/story)."""
    return await tools.list_hotspots(scene_slug, category=category)


@mcp.tool()
async def find_by_description_tool(
    scene_slug: str, text: str,
) -> list[dict]:
    """Find objects best matching a free-text description via CLIP similarity."""
    return await tools.find_by_description(scene_slug, text)


@mcp.tool()
async def measure_distance_tool(
    scene_slug: str, point_a: list[float], point_b: list[float],
) -> dict:
    """Compute distance in meters between two 3D points."""
    return await tools.measure_distance(scene_slug, point_a, point_b)


@mcp.tool()
async def plan_tour_tool(
    scene_slug: str, focus: str | None = None,
) -> list[dict]:
    """Plan a guided tour visiting hotspots (optionally focused on a category)."""
    return await tools.plan_tour(scene_slug, focus=focus)
```

- [ ] **Step 3: Verify MCP server can start**

```bash
cd api && python -m sceneagent.mcp_server --help
# Should print FastMCP usage
```

- [ ] **Step 4: Commit**

```bash
git add api/src/sceneagent/agent/ api/src/sceneagent/mcp_server.py
git commit -m "feat(api): MCP server exposing list_objects / list_hotspots / find_by_description / measure_distance / plan_tour"
```

---

## Phase 5 — LangGraph Agent + VLM Grounding

### Task 5.1: Gemini VLM wrapper

**Files:**
- Create: `api/src/sceneagent/vlm.py`

- [ ] **Step 1: Implement**

```python
from __future__ import annotations

import base64
import os

import google.generativeai as genai
from PIL import Image
from io import BytesIO

_MODEL = os.environ.get("VLM_MODEL", "gemini-2.0-flash")


async def describe_image(image_b64: str, question: str) -> str:
    """Feed a base64 PNG + a question to Gemini Flash vision, return the answer."""
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    img_bytes = base64.b64decode(image_b64)
    img = Image.open(BytesIO(img_bytes))
    model = genai.GenerativeModel(_MODEL)
    resp = await model.generate_content_async([
        question, img,
    ])
    return resp.text
```

- [ ] **Step 2: Commit**

```bash
git add api/src/sceneagent/vlm.py
git commit -m "feat(api): Gemini Flash VLM wrapper for agent visual grounding"
```

---

### Task 5.2: Render-view tool (for VLM grounding)

**Files:**
- Create: `api/src/sceneagent/render_proxy.py`

For v1 simplicity: **`render_view` returns a synthesized image from our matplotlib walkthrough render** (or picks the frame in the synthesized video closest to the requested pose). The agent doesn't get a real splat render — it gets the closest synthesized frame. Good enough for the VLM to reason about spatial layout.

- [ ] **Step 1: Implement**

```python
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))


def _find_closest_frame(pose: dict, trajectory: list[dict]) -> int:
    target = np.asarray(pose["position"])
    best_idx, best_dist = 0, float("inf")
    for i, p in enumerate(trajectory):
        d = float(np.linalg.norm(target - np.asarray(p["position"])))
        if d < best_dist:
            best_idx, best_dist = i, d
    return best_idx


async def render_view(scene_slug: str, position: list[float]) -> dict:
    """Return the walkthrough frame closest to `position` as base64 PNG."""
    scene_dir = DATA_DIR / "scene" / scene_slug
    traj = json.loads((scene_dir / "camera_trajectory.json").read_text())
    idx = _find_closest_frame({"position": position}, traj)

    # The walkthrough was encoded into video.mp4; for a single frame we can
    # pre-extract frames on startup OR re-use ffmpeg on demand.
    # Simpler path: pre-extracted frames are stored in scene_dir/frames/.
    frame_path = scene_dir / "frames" / f"frame_{idx:05d}.png"
    if not frame_path.exists():
        # Fallback: blank image so agent doesn't crash
        img = Image.new("RGB", (640, 360), (200, 200, 200))
    else:
        img = Image.open(frame_path)

    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format="PNG")
    return {
        "image_base64": base64.b64encode(buf.getvalue()).decode(),
        "width": img.width,
        "height": img.height,
    }
```

> **Note:** This requires frames/ to exist. Update `scripts/synthesize_walkthrough.py` so frames aren't deleted at the end (remove the cleanup loop) — or add a separate pre-extraction step.

- [ ] **Step 2: Quick change to `synthesize_walkthrough.py`**

Remove the block at the end:
```python
# Clean up
for f in frame_dir.iterdir():
    f.unlink()
frame_dir.rmdir()
```

Rename `_frames` → `frames` so they persist for `render_view`.

- [ ] **Step 3: Commit**

```bash
git add api/src/sceneagent/render_proxy.py scripts/synthesize_walkthrough.py
git commit -m "feat(api): render_view tool returns closest walkthrough frame as base64 png"
```

---

### Task 5.3: LangGraph agent

**Files:**
- Create: `api/src/sceneagent/agent/graph.py`
- Create: `api/src/sceneagent/routes/chat.py`
- Modify: `api/src/sceneagent/main.py`

- [ ] **Step 1: LangGraph state graph `agent/graph.py`**

```python
from __future__ import annotations

import json
import os
from typing import Annotated, TypedDict

import google.generativeai as genai
from langgraph.graph import StateGraph, END

from . import tools
from ..vlm import describe_image
from ..render_proxy import render_view


class AgentState(TypedDict):
    scene_slug: str
    user_message: str
    history: list[dict]
    tool_calls: list[dict]
    response: str


_TOOL_DEFS = [
    {
        "name": "list_hotspots",
        "description": "List hotspots in the scene, optionally by category (feature/included/issue/info/spec/story).",
        "parameters": {"category": "string | null"},
    },
    {
        "name": "find_by_description",
        "description": "Find the best-matching object in the scene for a given text description.",
        "parameters": {"text": "string"},
    },
    {
        "name": "measure_distance",
        "description": "Distance in meters between two [x,y,z] points.",
        "parameters": {"point_a": "list[float]", "point_b": "list[float]"},
    },
    {
        "name": "render_view",
        "description": "Render the scene from a given [x,y,z] position and get a text description of what is visible. Use when you need visual context to answer.",
        "parameters": {"position": "list[float]", "question": "string"},
    },
    {
        "name": "plan_tour",
        "description": "Plan a guided tour through hotspots, optionally focused on a category.",
        "parameters": {"focus": "string | null"},
    },
    {
        "name": "answer",
        "description": "Provide the final answer to the user. Call this when you have enough information.",
        "parameters": {"text": "string", "highlight_hotspot_ids": "list[string] | null"},
    },
]

SYSTEM_PROMPT = """You are the AI concierge for a 3D real-estate listing.
You have access to these tools to investigate the scene. Use tools as needed,
then call `answer` with your final response.

Tools:
{tool_defs}

Always respond with a JSON object: {{"tool": "<tool_name>", "args": {{...}}}}.
"""


async def _llm_pick_tool(state: AgentState) -> dict:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(os.environ.get("LLM_MODEL", "gemini-2.0-flash"))
    prompt = SYSTEM_PROMPT.format(tool_defs=json.dumps(_TOOL_DEFS, indent=2))
    history = "\n".join(
        f"User: {h['user']}" if h.get("user") else f"Tool({h['tool']}) -> {h['result']}"
        for h in state["history"]
    )
    full = f"{prompt}\n\nHistory:\n{history}\nUser: {state['user_message']}\n"
    resp = await model.generate_content_async(
        full, generation_config={"response_mime_type": "application/json"}
    )
    return json.loads(resp.text)


async def _execute_tool(state: AgentState) -> AgentState:
    decision = await _llm_pick_tool(state)
    name = decision["tool"]
    args = decision.get("args", {})
    slug = state["scene_slug"]

    if name == "answer":
        state["response"] = args["text"]
        state["tool_calls"].append({"tool": "answer", "args": args})
        return state

    if name == "list_hotspots":
        result = await tools.list_hotspots(slug, category=args.get("category"))
    elif name == "find_by_description":
        result = await tools.find_by_description(slug, text=args["text"])
    elif name == "measure_distance":
        result = await tools.measure_distance(slug, args["point_a"], args["point_b"])
    elif name == "render_view":
        img = await render_view(slug, args["position"])
        result = {"description": await describe_image(img["image_base64"], args.get("question", "What do you see?"))}
    elif name == "plan_tour":
        result = await tools.plan_tour(slug, focus=args.get("focus"))
    else:
        result = {"error": f"unknown tool {name}"}

    state["tool_calls"].append({"tool": name, "args": args, "result": result})
    state["history"].append({"tool": name, "result": result})
    return state


def _should_continue(state: AgentState) -> str:
    if state.get("response"):
        return END
    if len(state["tool_calls"]) >= 6:
        state["response"] = "I couldn't fully resolve that — here's what I found so far."
        return END
    return "execute"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("execute", _execute_tool)
    g.set_entry_point("execute")
    g.add_conditional_edges("execute", _should_continue, {"execute": "execute", END: END})
    return g.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
```

- [ ] **Step 2: SSE chat endpoint `routes/chat.py`**

```python
from __future__ import annotations

import asyncio
import json
from fastapi import APIRouter
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from ..agent.graph import get_graph, AgentState

router = APIRouter(prefix="/scenes/{slug}/chat", tags=["chat"])


class ChatIn(BaseModel):
    message: str


@router.post("")
async def chat(slug: str, payload: ChatIn):
    state: AgentState = {
        "scene_slug": slug,
        "user_message": payload.message,
        "history": [],
        "tool_calls": [],
        "response": "",
    }
    result = await get_graph().ainvoke(state)
    return {
        "response": result["response"],
        "tool_calls": result["tool_calls"],
    }


@router.post("/stream")
async def chat_stream(slug: str, payload: ChatIn):
    async def event_gen():
        state: AgentState = {
            "scene_slug": slug, "user_message": payload.message,
            "history": [], "tool_calls": [], "response": "",
        }
        graph = get_graph()
        async for event in graph.astream(state):
            # langgraph emits node-level updates
            yield {"event": "update", "data": json.dumps(event, default=str)}
        yield {"event": "done", "data": json.dumps({})}
    return EventSourceResponse(event_gen())
```

- [ ] **Step 3: Register chat router in main.py**

```python
from .routes.chat import router as chat_router
app.include_router(chat_router)
```

- [ ] **Step 4: Manual verify**

```bash
curl -X POST http://localhost:8000/scenes/demo/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Are there any issues I should know about?"}'
# → JSON with response + tool_calls array
```

- [ ] **Step 5: Commit**

```bash
git add api/src/sceneagent/agent/graph.py api/src/sceneagent/routes/chat.py api/src/sceneagent/main.py
git commit -m "feat(api): LangGraph agent with tool-call loop, VLM grounding, SSE chat endpoint"
```

---

## Phase 6 — Frontend

### Task 6.1: Next.js scaffold

**Files:** `web/*`

- [ ] **Step 1: Bootstrap Next.js**

```bash
cd web
npx create-next-app@latest . --typescript --tailwind --app --src-dir --no-eslint --import-alias "@/*"
# answer all prompts; keep defaults
```

- [ ] **Step 2: Add dependencies**

```bash
npm install three @mkkellogg/gaussian-splats-3d zustand @tanstack/react-query react-player @radix-ui/react-dialog @radix-ui/react-popover lucide-react clsx
npm install -D @types/three
```

- [ ] **Step 3: Update `web/Dockerfile`**

```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
CMD ["npm", "run", "dev", "--", "-p", "3000"]
```

- [ ] **Step 4: Commit**

```bash
git add web/
git commit -m "feat(web): Next.js 14 scaffold with Three.js, gaussian-splats-3d, zustand, tanstack-query"
```

---

### Task 6.2: Zustand viewer store

**Files:**
- Create: `web/src/stores/viewer.ts`

- [ ] **Step 1: Write**

```typescript
import { create } from 'zustand';

export type Vec3 = [number, number, number];

export interface Hotspot {
  id: string;
  position: Vec3;
  note_text: string;
  category: 'feature' | 'included' | 'issue' | 'info' | 'spec' | 'story' | 'other';
  class_name: string | null;
}

interface ViewerState {
  highlightedHotspotIds: string[];
  flyToPosition: Vec3 | null;
  tourStops: { position: Vec3; dwell_seconds: number; narration: string }[] | null;
  setHighlights: (ids: string[]) => void;
  flyTo: (pos: Vec3) => void;
  setTour: (stops: ViewerState['tourStops']) => void;
  clear: () => void;
}

export const useViewerStore = create<ViewerState>((set) => ({
  highlightedHotspotIds: [],
  flyToPosition: null,
  tourStops: null,
  setHighlights: (ids) => set({ highlightedHotspotIds: ids }),
  flyTo: (pos) => set({ flyToPosition: pos }),
  setTour: (stops) => set({ tourStops: stops }),
  clear: () => set({ highlightedHotspotIds: [], flyToPosition: null, tourStops: null }),
}));
```

- [ ] **Step 2: Commit**

```bash
git add web/src/stores/viewer.ts
git commit -m "feat(web): zustand viewer store for cross-component scene state"
```

---

### Task 6.3: SplatViewer component

**Files:**
- Create: `web/src/components/SplatViewer.tsx`

- [ ] **Step 1: Write (client-only, dynamic import)**

```tsx
'use client';

import { useEffect, useRef } from 'react';
import * as THREE from 'three';
// @ts-expect-error no types
import { Viewer as GaussianViewer } from '@mkkellogg/gaussian-splats-3d';
import { useViewerStore } from '@/stores/viewer';

interface Props { splatUrl: string; }

export function SplatViewer({ splatUrl }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<any>(null);
  const flyToPosition = useViewerStore((s) => s.flyToPosition);

  useEffect(() => {
    if (!containerRef.current) return;
    const viewer = new GaussianViewer({
      rootElement: containerRef.current,
      cameraUp: [0, 0, 1],
      initialCameraPosition: [4, 4, 2],
      initialCameraLookAt: [0, 0, 1],
      sphericalHarmonicsDegree: 2,
    });
    viewerRef.current = viewer;
    viewer.addSplatScene(splatUrl).then(() => viewer.start());
    return () => {
      viewer.dispose?.();
    };
  }, [splatUrl]);

  useEffect(() => {
    if (!flyToPosition || !viewerRef.current) return;
    const v = viewerRef.current;
    const cam: THREE.Camera = v.camera;
    if (!cam) return;
    const target = new THREE.Vector3(...flyToPosition);
    // simple linear interpolation over 1s
    const start = cam.position.clone();
    const t0 = performance.now();
    const tick = () => {
      const t = Math.min(1, (performance.now() - t0) / 1000);
      cam.position.lerpVectors(start, target, t);
      cam.lookAt(target);
      if (t < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }, [flyToPosition]);

  return <div ref={containerRef} className="absolute inset-0 w-full h-full" />;
}
```

- [ ] **Step 2: Commit**

```bash
git add web/src/components/SplatViewer.tsx
git commit -m "feat(web): SplatViewer component with fly-to support from zustand"
```

---

### Task 6.4: Hotspot markers overlay

**Files:**
- Create: `web/src/components/HotspotMarkers.tsx`

- [ ] **Step 1: Write**

```tsx
'use client';

import { useState } from 'react';
import { useViewerStore, type Hotspot } from '@/stores/viewer';

const CATEGORY_ICON: Record<Hotspot['category'], string> = {
  feature:  '✅', included: '🎁', issue: '⚠️',
  info:     'ℹ️', spec:     '📏', story: '📖', other: '📍',
};

function project(pos: [number, number, number]): { left: number; top: number } {
  // TODO: real projection from Three.js camera. For v1, stub at screen center until
  // we wire a frame-by-frame projection loop. Acceptable for the demo since agent
  // also calls flyTo which re-centers on the hotspot.
  return { left: 50, top: 50 };
}

export function HotspotMarkers({ hotspots }: { hotspots: Hotspot[] }) {
  const [openId, setOpenId] = useState<string | null>(null);
  const flyTo = useViewerStore((s) => s.flyTo);

  return (
    <div className="absolute inset-0 pointer-events-none">
      {hotspots.map((h) => {
        const p = project(h.position);
        return (
          <button
            key={h.id}
            className="absolute -translate-x-1/2 -translate-y-1/2 text-3xl pointer-events-auto hover:scale-110 transition"
            style={{ left: `${p.left}%`, top: `${p.top}%` }}
            onClick={() => {
              flyTo(h.position);
              setOpenId(h.id === openId ? null : h.id);
            }}
          >
            {CATEGORY_ICON[h.category]}
            {openId === h.id && (
              <div className="absolute left-8 top-0 bg-white text-black text-sm p-3 rounded shadow-lg w-64 z-10">
                <div className="font-semibold mb-1 capitalize">{h.category}</div>
                <div>{h.note_text}</div>
              </div>
            )}
          </button>
        );
      })}
    </div>
  );
}
```

> **Note:** Accurate 3D→2D projection needs the viewer's camera matrix. The stub above is acceptable for v1 because the agent triggers `flyTo` before highlighting — the hotspot becomes the visual center. A proper projection upgrade is in v2.

- [ ] **Step 2: Commit**

```bash
git add web/src/components/HotspotMarkers.tsx
git commit -m "feat(web): HotspotMarkers overlay with category icons and click-to-fly-to"
```

---

### Task 6.5: API client

**Files:**
- Create: `web/src/lib/api.ts`

```typescript
const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function getScene(slug: string) {
  const r = await fetch(`${API}/scenes/${slug}`);
  if (!r.ok) throw new Error('scene fetch failed');
  return r.json();
}

export async function getHotspots(slug: string, category?: string) {
  const url = new URL(`${API}/scenes/${slug}/hotspots`);
  if (category) url.searchParams.set('category', category);
  const r = await fetch(url);
  if (!r.ok) throw new Error('hotspots fetch failed');
  return r.json();
}

export async function createNote(slug: string, text: string, video_timestamp: number) {
  const r = await fetch(`${API}/scenes/${slug}/notes`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, video_timestamp }),
  });
  if (!r.ok) throw new Error('note create failed');
  return r.json();
}

export async function seedMatch(slug: string) {
  const r = await fetch(`${API}/scenes/${slug}/notes/seed-match`, { method: 'POST' });
  return r.json();
}

export async function chat(slug: string, message: string) {
  const r = await fetch(`${API}/scenes/${slug}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
  });
  return r.json();
}
```

Commit:

```bash
git add web/src/lib/api.ts
git commit -m "feat(web): REST client for scenes/hotspots/notes/chat"
```

---

### Task 6.6: Listing page (buyer view)

**Files:**
- Create: `web/src/app/listing/[slug]/page.tsx`

```tsx
'use client';

import { useQuery } from '@tanstack/react-query';
import { SplatViewer } from '@/components/SplatViewer';
import { HotspotMarkers } from '@/components/HotspotMarkers';
import { ChatOverlay } from '@/components/ChatOverlay';
import { getScene, getHotspots, seedMatch } from '@/lib/api';
import { useEffect } from 'react';

export default function ListingPage({ params }: { params: { slug: string } }) {
  const { data: scene } = useQuery({
    queryKey: ['scene', params.slug],
    queryFn: () => getScene(params.slug),
  });
  const { data: hotspots = [], refetch } = useQuery({
    queryKey: ['hotspots', params.slug],
    queryFn: () => getHotspots(params.slug),
  });

  useEffect(() => {
    // one-time: match any unmatched seed notes
    seedMatch(params.slug).then(() => refetch());
  }, []);

  if (!scene) return <div className="p-8">Loading…</div>;

  const splatSrc = `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}${scene.splat_url}`;

  return (
    <div className="relative w-screen h-screen bg-black">
      <SplatViewer splatUrl={splatSrc} />
      <HotspotMarkers hotspots={hotspots} />
      <ChatOverlay slug={params.slug} />
      <div className="absolute top-4 left-4 bg-black/60 text-white p-3 rounded">
        <h1 className="text-lg font-semibold">{scene.title}</h1>
        <p className="text-sm opacity-80">{scene.address || '—'}</p>
      </div>
    </div>
  );
}
```

> **Static splat file serving:** The API needs to serve `/static/scene/demo/3dgs_compressed.ply`. Add to `api/src/sceneagent/main.py`:
> ```python
> from fastapi.staticfiles import StaticFiles
> app.mount("/static", StaticFiles(directory="/app/data"), name="static")
> ```

Commit:

```bash
git add web/src/app/listing/[slug]/page.tsx api/src/sceneagent/main.py
git commit -m "feat(web): listing buyer page with splat viewer, hotspots, chat overlay"
```

---

### Task 6.7: Chat overlay

**Files:**
- Create: `web/src/components/ChatOverlay.tsx`

```tsx
'use client';

import { useState } from 'react';
import { chat } from '@/lib/api';
import { useViewerStore } from '@/stores/viewer';

type Msg = { role: 'user' | 'assistant' | 'tool'; text: string };

export function ChatOverlay({ slug }: { slug: string }) {
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const flyTo = useViewerStore((s) => s.flyTo);

  const send = async () => {
    if (!input.trim() || busy) return;
    const userMsg = input.trim();
    setMsgs((m) => [...m, { role: 'user', text: userMsg }]);
    setInput('');
    setBusy(true);
    try {
      const res = await chat(slug, userMsg);
      for (const tc of res.tool_calls || []) {
        setMsgs((m) => [...m, { role: 'tool', text: `${tc.tool} → ${JSON.stringify(tc.result ?? tc.args).slice(0, 200)}` }]);
        // side effect: move camera if tool suggests
        if (tc.tool === 'find_by_description' && tc.result?.[0]?.centroid) {
          flyTo(tc.result[0].centroid as [number, number, number]);
        }
        if (tc.tool === 'plan_tour' && Array.isArray(tc.result) && tc.result.length > 0) {
          flyTo(tc.result[0].position as [number, number, number]);
        }
      }
      setMsgs((m) => [...m, { role: 'assistant', text: res.response }]);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="absolute bottom-4 right-4 w-96 max-h-[60vh] bg-white rounded-lg shadow-xl flex flex-col">
      <div className="p-3 border-b font-semibold">AI Concierge</div>
      <div className="flex-1 overflow-y-auto p-3 space-y-2 text-sm">
        {msgs.map((m, i) => (
          <div key={i} className={
            m.role === 'user' ? 'text-blue-700' :
            m.role === 'tool' ? 'text-gray-500 font-mono text-xs' :
            'text-black'
          }>
            {m.role === 'user' ? '🧑 ' : m.role === 'tool' ? '🔧 ' : '🤖 '}
            {m.text}
          </div>
        ))}
      </div>
      <div className="p-2 border-t flex gap-2">
        <input
          className="flex-1 border rounded px-2 py-1 text-sm"
          value={input}
          placeholder="Ask about this listing…"
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && send()}
          disabled={busy}
        />
        <button
          className="bg-black text-white px-3 py-1 rounded disabled:opacity-50"
          onClick={send} disabled={busy}
        >Send</button>
      </div>
    </div>
  );
}
```

Commit:

```bash
git add web/src/components/ChatOverlay.tsx
git commit -m "feat(web): ChatOverlay with agent round-trips + camera side-effects from tool calls"
```

---

### Task 6.8: Scrubber admin page

**Files:**
- Create: `web/src/app/listing/[slug]/admin/page.tsx`

```tsx
'use client';

import { useRef, useState } from 'react';
import ReactPlayer from 'react-player';
import { createNote, getHotspots } from '@/lib/api';
import { useQuery, useQueryClient } from '@tanstack/react-query';

export default function AdminPage({ params }: { params: { slug: string } }) {
  const [text, setText] = useState('');
  const playerRef = useRef<any>(null);
  const qc = useQueryClient();
  const { data: hotspots = [] } = useQuery({
    queryKey: ['hotspots', params.slug],
    queryFn: () => getHotspots(params.slug),
  });

  const videoUrl = `${process.env.NEXT_PUBLIC_API_URL}/static/scene/${params.slug}/video.mp4`;

  const addNote = async () => {
    if (!text.trim()) return;
    const t = playerRef.current?.getCurrentTime?.() ?? 0;
    await createNote(params.slug, text.trim(), t);
    setText('');
    qc.invalidateQueries({ queryKey: ['hotspots', params.slug] });
  };

  return (
    <div className="grid grid-cols-2 gap-4 p-4 min-h-screen">
      <div>
        <h2 className="font-semibold mb-2">Walkthrough</h2>
        <ReactPlayer ref={playerRef} url={videoUrl} controls width="100%" />
        <div className="mt-3">
          <textarea
            className="w-full border rounded p-2"
            placeholder="At the current moment, note something about this spot…"
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={3}
          />
          <button
            className="mt-2 bg-black text-white px-4 py-2 rounded"
            onClick={addNote}
          >Add note at current time</button>
        </div>
      </div>
      <div>
        <h2 className="font-semibold mb-2">Hotspots</h2>
        <ul className="space-y-2 text-sm">
          {hotspots.map((h: any) => (
            <li key={h.id} className="border rounded p-2">
              <div className="text-xs opacity-60">
                {h.category} · {h.class_name} · conf {h.match_confidence?.toFixed?.(2)}
              </div>
              <div>{h.note_text}</div>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
```

Commit:

```bash
git add web/src/app/listing/[slug]/admin/page.tsx
git commit -m "feat(web): admin scrubber page — video + note input + live hotspot list"
```

---

### Task 6.9: Wire React Query provider + homepage

**Files:**
- Modify: `web/src/app/layout.tsx`
- Create: `web/src/app/providers.tsx`
- Modify: `web/src/app/page.tsx`

- [ ] **Step 1: Providers**

```tsx
// web/src/app/providers.tsx
'use client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useState } from 'react';

export default function Providers({ children }: { children: React.ReactNode }) {
  const [client] = useState(() => new QueryClient());
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}
```

- [ ] **Step 2: Wrap layout**

In `web/src/app/layout.tsx`, wrap `{children}` with `<Providers>`.

- [ ] **Step 3: Homepage**

```tsx
// web/src/app/page.tsx
import Link from 'next/link';

export default function Home() {
  return (
    <main className="p-10 min-h-screen">
      <h1 className="text-3xl font-bold mb-6">SceneAgent Demo</h1>
      <div className="space-x-4">
        <Link className="underline" href="/listing/demo">Buyer view</Link>
        <Link className="underline" href="/listing/demo/admin">Realtor admin</Link>
      </div>
    </main>
  );
}
```

- [ ] **Step 4: Commit**

```bash
git add web/src/app/providers.tsx web/src/app/layout.tsx web/src/app/page.tsx
git commit -m "feat(web): React Query provider + simple homepage with buyer/admin links"
```

---

## Phase 7 — Deployment polish

### Task 7.1: K8s manifests

**Files:** `k8s/*.yaml`

- [ ] **Step 1: Namespace**

`k8s/00-namespace.yaml`:
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: sceneagent
```

- [ ] **Step 2: Postgres StatefulSet + service**

`k8s/10-postgres.yaml`:
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata: { name: postgres, namespace: sceneagent }
spec:
  serviceName: postgres
  replicas: 1
  selector: { matchLabels: { app: postgres } }
  template:
    metadata: { labels: { app: postgres } }
    spec:
      containers:
        - name: postgres
          image: pgvector/pgvector:pg16
          env:
            - { name: POSTGRES_USER, value: sceneagent }
            - { name: POSTGRES_PASSWORD, value: sceneagent }
            - { name: POSTGRES_DB, value: sceneagent }
          ports: [{ containerPort: 5432 }]
          volumeMounts:
            - { name: data, mountPath: /var/lib/postgresql/data }
  volumeClaimTemplates:
    - metadata: { name: data }
      spec:
        accessModes: ["ReadWriteOnce"]
        resources: { requests: { storage: 1Gi } }
---
apiVersion: v1
kind: Service
metadata: { name: postgres, namespace: sceneagent }
spec:
  selector: { app: postgres }
  ports: [{ port: 5432, targetPort: 5432 }]
```

- [ ] **Step 3: Redis**

`k8s/20-redis.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata: { name: redis, namespace: sceneagent }
spec:
  replicas: 1
  selector: { matchLabels: { app: redis } }
  template:
    metadata: { labels: { app: redis } }
    spec:
      containers:
        - name: redis
          image: redis:7-alpine
          ports: [{ containerPort: 6379 }]
---
apiVersion: v1
kind: Service
metadata: { name: redis, namespace: sceneagent }
spec:
  selector: { app: redis }
  ports: [{ port: 6379, targetPort: 6379 }]
```

- [ ] **Step 4: API + web deployments + services + ingress**

`k8s/30-api.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata: { name: api, namespace: sceneagent }
spec:
  replicas: 1
  selector: { matchLabels: { app: api } }
  template:
    metadata: { labels: { app: api } }
    spec:
      containers:
        - name: api
          image: sceneagent-api:dev
          imagePullPolicy: IfNotPresent
          env:
            - { name: DATABASE_URL, value: "postgresql://sceneagent:sceneagent@postgres:5432/sceneagent" }
            - { name: REDIS_URL, value: "redis://redis:6379/0" }
            - { name: SCENE_ID, value: "demo" }
          envFrom:
            - secretRef: { name: agent-secrets }
          ports: [{ containerPort: 8000 }]
          volumeMounts:
            - { name: scene-data, mountPath: /app/data, readOnly: true }
      volumes:
        - name: scene-data
          hostPath: { path: /mnt/sceneagent-data, type: DirectoryOrCreate }
---
apiVersion: v1
kind: Service
metadata: { name: api, namespace: sceneagent }
spec:
  selector: { app: api }
  ports: [{ port: 8000, targetPort: 8000 }]
```

`k8s/40-web.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata: { name: web, namespace: sceneagent }
spec:
  replicas: 1
  selector: { matchLabels: { app: web } }
  template:
    metadata: { labels: { app: web } }
    spec:
      containers:
        - name: web
          image: sceneagent-web:dev
          imagePullPolicy: IfNotPresent
          env:
            - { name: NEXT_PUBLIC_API_URL, value: "http://localhost:8000" }
          ports: [{ containerPort: 3000 }]
---
apiVersion: v1
kind: Service
metadata: { name: web, namespace: sceneagent }
spec:
  type: NodePort
  selector: { app: web }
  ports: [{ port: 3000, targetPort: 3000, nodePort: 30000 }]
```

`k8s/99-secrets.example.yaml`:
```yaml
apiVersion: v1
kind: Secret
metadata: { name: agent-secrets, namespace: sceneagent }
type: Opaque
stringData:
  GEMINI_API_KEY: "replace_me"
```

- [ ] **Step 5: Apply script**

`k8s/apply.sh`:
```bash
#!/usr/bin/env bash
set -e
# Mount scene data into minikube
minikube mount "$(pwd)/data:/mnt/sceneagent-data" &
MOUNT_PID=$!
trap "kill $MOUNT_PID" EXIT
# Build images in minikube's docker
eval "$(minikube docker-env)"
docker build -t sceneagent-api:dev api/
docker build -t sceneagent-web:dev web/
# Apply manifests
kubectl apply -f k8s/00-namespace.yaml
kubectl apply -f k8s/10-postgres.yaml
kubectl apply -f k8s/20-redis.yaml
kubectl apply -f k8s/99-secrets.example.yaml  # user edits before running
kubectl apply -f k8s/30-api.yaml
kubectl apply -f k8s/40-web.yaml
kubectl -n sceneagent rollout status deploy/api
kubectl -n sceneagent rollout status deploy/web
echo "Web: minikube service -n sceneagent web --url"
```

- [ ] **Step 6: Commit**

```bash
git add k8s/
git commit -m "feat(infra): Kubernetes manifests + apply script for minikube"
```

---

### Task 7.2: GitHub Actions CI

**Files:**
- Create: `.github/workflows/ci.yml`

```yaml
name: CI
on:
  push: { branches: [main] }
  pull_request:

jobs:
  api:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install ruff mypy
      - run: cd api && pip install -e .
      - run: ruff check api/src/
      - run: cd api && python -c "import sceneagent.main"

  web:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: '20' }
      - run: cd web && npm ci
      - run: cd web && npx tsc --noEmit
      - run: cd web && npm run build
```

Commit:

```bash
git add .github/workflows/ci.yml
git commit -m "ci: lint + typecheck + build on push/PR"
```

---

### Task 7.3: Flesh out README

**Files:**
- Modify: `README.md`

```markdown
# SceneAgent

AI-concierge real-estate listings. Upload a walkthrough video → get an interactive 3D digital twin with an AI agent that can see, tour buyers through, and answer questions pinned to exact 3D locations.

![architecture diagram placeholder — insert image]

## What this is

A portfolio project demonstrating a 3D computer vision + agentic AI + digital twin pipeline:

- **3D scene representation** — Gaussian Splats (from [InteriorGS](https://huggingface.co/datasets/spatialverse/InteriorGS))
- **Semantic layer** — per-object class labels, instance IDs, 3D bounding boxes
- **AI agent** — LangGraph + MCP tool server + Gemini Flash vision grounding
- **Scrubber note UI** — realtor types timestamped notes; matcher turns them into 3D hotspots
- **Infra** — FastAPI + Postgres+pgvector + Docker Compose + Kubernetes manifests

## Architecture

Three containers: `web` (Next.js 14), `api` (FastAPI + LangGraph + MCP), `postgres` (pgvector). `redis` is scaffolded for the v2 async splat-training pipeline.

**Agent pattern:** The agent grounds itself in the scene by calling `render_view` as an MCP tool and piping the result to Gemini Flash vision. It doesn't "know about" the scene abstractly — it *looks at it* on demand.

## Quick start

```bash
cp .env.example .env && $EDITOR .env   # set GEMINI_API_KEY, HF_TOKEN

# Download one InteriorGS scene (requires HF login + license acceptance)
./scripts/download_scene.sh scenes/0001

# Prep scene: parse labels → embeddings → synthesize walkthrough video (~15 min, CPU only)
pip install -e scripts/
python scripts/prepare_scene.py data/scene/demo

# Launch
docker compose up --build
open http://localhost:3000/listing/demo
```

## Kubernetes demo

```bash
./k8s/apply.sh
minikube service -n sceneagent web --url
```

## v1 scope

- [x] InteriorGS scene loader (no GPU, ~15 min CPU)
- [x] Scrubber UI for timestamped notes
- [x] Note→hotspot matcher (CLIP + frustum)
- [x] Gemini-based note categorization
- [x] LangGraph agent + MCP server
- [x] VLM grounding via `render_view`
- [x] Guided tour
- [x] Docker Compose + K8s manifests

## v2 roadmap (not built)

- Voice narration during capture (Whisper STT)
- Realtor review UI (swipe-approve low-confidence matches)
- Adaptive re-rendering for borderline matches
- Real scene editing (move/remove objects — Gaussian Grouping supports it)
- User-uploaded video → splat training pipeline (Celery + GPU worker)
- Multi-scene comparison
- Move-in furniture placement

## Credits

- Scene data: [InteriorGS](https://huggingface.co/datasets/spatialverse/InteriorGS) (manycore-research)
- 3D segmentation reference: [Gaussian Grouping (ECCV 2024)](https://github.com/lkeab/gaussian-grouping)
- Splat viewer: [@mkkellogg/gaussian-splats-3d](https://github.com/mkkellogg/GaussianSplats3D)

## Demo video

[YouTube link — insert after recording]
```

Commit:

```bash
git add README.md
git commit -m "docs: flesh out README with architecture, quick start, v2 roadmap, credits"
```

---

## Phase 8 — Demo Video

### Task 8.1: Record the 2–3 min demo

**Files (out of repo):** OBS recording, DaVinci Resolve project, final `demo.mp4`.

Shot list (as in spec §12):

1. **0:00–0:10** — Static Zillow screenshot, voiceover: *"Zillow listings are 30 static photos. Buyers email sellers 'does the desk come with it?' We fixed that."*
2. **0:10–0:20** — Title card, fade to homepage → `/listing/demo/admin`.
3. **0:20–0:30** — Video uploader mock screen (record "processing..." cut, then scene appears).
4. **0:30–0:55** — Scrubber: drag to 0:08, type "the desk comes with the apartment," ENTER. Drag to 0:03, type "window sticks, pull firmly," ENTER. Drag to 0:18, type "heated bathroom floor," ENTER. Cut to hotspots pane — they appear.
5. **0:55–1:10** — Switch to `/listing/demo` (buyer view). Orbit around. Click 🎁 hotspot → see "desk included". Click ⚠️ → "window sticks".
6. **1:10–1:50** — Open chat:
   - "Are there any issues I should know about?" → agent calls `list_hotspots(issue)`, lists them, highlights in 3D
   - "How tall are the ceilings in the living room?" → agent calls `find_by_description`, then `measure_distance`, returns 3.2m
   - "Give me a tour of the best features." → agent calls `plan_tour(feature)`, camera flies through stops
7. **1:50–2:05** — Cut to terminal: `./k8s/apply.sh` → pods green → `minikube service web --url` → browser reloads.
8. **2:05–2:15** — End card: GitHub URL + tech stack badges.

- [ ] **Step 1: Recording**

```
- OBS Studio: record screen at 1920x1080 @ 30fps
- Record each segment separately (no interactive mistakes mid-shot)
- Record voiceover separately in Audacity (headset mic, noise-reduce)
```

- [ ] **Step 2: Edit in DaVinci Resolve (free)**

Rough-cut each segment, sync voiceover, add simple title cards (Resolve's built-in), export 1080p30 mp4.

- [ ] **Step 3: Upload & link**

```
- Upload to YouTube as UNLISTED
- Update README `Demo video` section with the link
- git add README.md && git commit -m "docs: add demo video link"
```

---

## Definition of Done

1. `docker compose up --build` starts cleanly.
2. `http://localhost:3000/listing/demo` loads the splat with hotspots.
3. `http://localhost:3000/listing/demo/admin` accepts notes via scrubber.
4. Chat works for: "List any issues", "What's included?", "How tall are the ceilings?", "Tour the features."
5. Agent calls `render_view` + VLM at least once.
6. `./k8s/apply.sh` on minikube brings stack up, web reachable via port-forward.
7. README complete: architecture, quick start, v2 roadmap, credits, demo video link.
8. GitHub Actions CI green.
9. Demo video recorded & linked.

Ship it.
