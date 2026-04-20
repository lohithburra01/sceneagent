# SceneAgent — Design Spec

**Date:** 2026-04-20
**Working title:** SceneAgent (rebrand before publishing — candidates: ListingTwin, TwinListing, OpenHouse AI)
**Author:** Lohith Burra
**Purpose:** Portfolio project targeting 3D Computer Vision + Agentic AI + Digital Twin roles (2026 JD landscape).
**Build window:** 6 hours, parallel execution across 4 tracks. Scene is downloaded from **InteriorGS** (pre-trained Gaussian splat — saves 90 min of GPU training) but **we run our own 2D→3D semantic segmentation pipeline** using SAM + CLIP + multi-view backprojection, and evaluate it against InteriorGS's ground-truth labels. The splat is a dataset crutch; the segmentation is ours.

---

## 1. Product Vision

> A real-estate listing platform where every apartment is an interactive 3D digital twin, and every listing has an AI concierge that can actually see, walk buyers through the space, and answer questions using seller-authored notes pinned to exact 3D locations.

**One-line pitch for the demo video:** *"What if every Zillow listing was 3D and had an AI realtor that actually knew the place?"*

**The novel technical angle:** The AI agent grounds itself in the 3D scene through MCP tools that render views on demand and feed them to a vision-language model. The agent doesn't "know about" the scene abstractly — it *looks at it* when it needs to.

---

## 2. User Flows

### 2.1 Realtor (seller side)
1. Record a walkthrough video of the property (phone, handheld, any orientation).
2. Upload video to the web app.
3. Backend processes the video offline (simulated in v1 — pre-computed).
4. While waiting, type timestamped notes via a video-scrubber UI: scrub to a moment, type a note, hit enter.
5. When processing completes, realtor sees the 3D scene with hotspots auto-placed from the notes.
6. Realtor reviews low-confidence hotspots (swipe to confirm/reject).
7. Publish the listing.

### 2.2 Buyer (viewer side)
1. Open a listing URL — 3D viewer loads the Gaussian splat.
2. Orbit/walk through the scene with mouse + WASD.
3. Click any hotspot to read the seller's note (categorized: ⚠️ issue, 🎁 included, ✅ feature, ℹ️ info, 📏 spec, 📖 story).
4. Open the chat overlay — "Ask the AI about this place."
5. The AI answers using:
   - Hotspot notes (RAG-style lookup via pgvector)
   - Live VLM inspection of rendered views (for spatial/visual questions)
   - Scene tools (measure distance, find by description, etc.)
6. Ask for a guided tour — the agent flies the camera through hotspots and narrates.

---

## 3. v1 Scope (What Ships in 1–2 Days)

### 3.1 Real and working

| Component | Status |
|---|---|
| Next.js 14 frontend with Gaussian splat viewer (@mkkellogg/gaussian-splats-3d) | ✅ |
| Realtor-side: video scrubber UI for timestamped text notes | ✅ |
| Viewer-side: 3D hotspot markers (billboarded sprites, clickable popups) | ✅ |
| Chat overlay (buyer-side) | ✅ |
| FastAPI backend + PostgreSQL + pgvector | ✅ |
| MCP server exposing scene tools | ✅ |
| LangGraph agent that consumes the MCP server | ✅ |
| VLM grounding via `render_view` + Gemini Flash (free tier) | ✅ |
| Guided tour (agent-planned camera path through hotspots with narration) | ✅ |
| Hotspot taxonomy with LLM-inferred categories (one call per note) | ✅ |
| Notes → hotspot matcher (CLIP + timestamp → pose → visibility filter) | ✅ |
| Docker Compose for local dev (all services up with one command) | ✅ |
| Kubernetes manifests (plain YAML, runnable on minikube for demo) | ✅ |
| GitHub Actions CI (lint + typecheck + build) | ✅ |
| README with architecture diagram | ✅ |
| 2–3 min demo video | ✅ |

### 3.2 Pre-computed / faked (with honest documentation)

| Component | v1 behavior | README explanation |
|---|---|---|
| Video → splat training pipeline | **Skipped entirely.** Scene is downloaded from [InteriorGS](https://huggingface.co/datasets/spatialverse/InteriorGS). Upload endpoint exists in code path but the demo uses the downloaded scene. | "Splat training takes ~45 min on a T4 GPU. Production would queue this via Celery + GPU worker; the demo uses an InteriorGS scene (pre-trained, pre-labeled) for reproducibility." |
| 3D semantic segmentation | **Real pipeline runs.** Puppeteer + @mkkellogg/gaussian-splats-3d renders 30 views from the splat; MobileSAM produces 2D instance masks; OpenCLIP zero-shot assigns class names; our backprojection code projects masks onto Gaussians + majority-votes; DBSCAN clusters into instances. Output compared to InteriorGS `labels.json` ground truth — per-class precision/recall/mIoU reported in README. | If segmentation quality is too low by hour 4, product falls back to InteriorGS GT labels but README documents our pipeline with whatever metrics we got (honest portfolio artifact). |
| Notes storage for demo | **JSON file** (`demo_notes.json`) with hardcoded timestamped notes. Also addable live via scrubber UI. | "Demo ships with pre-written notes for reproducibility; the scrubber UI lets you add more live." |
| Splat training service | Celery worker skeleton present but not wired to GPU. Redis container runs. | "Celery infrastructure is in place; production would attach GPU nodes to the worker pool." |

### 3.3 Out of scope for v1 (documented in README as v2)

- Voice narration + Whisper transcription
- Realtor review UI (swipe-approve low-confidence matches) — confidence threshold is loose in v1
- Adaptive re-rendering for borderline matches
- Real scene editing (moving/removing objects in the splat)
- Multi-scene comparison
- Buyer-added questions → seller queue
- Move-in furniture placement simulator
- Multi-user auth / real listings database

---

## 4. Architecture

### 4.1 Services (all Dockerized)

```
┌─────────────────────────────────────────────────────────────┐
│                   web (Next.js 14)                          │
│   - Viewer (gaussian-splats-3d + Three.js)                  │
│   - Scrubber UI / Chat overlay                              │
│   - Hotspot markers + highlight layer                       │
└──────────────────────┬──────────────────────────────────────┘
                       │ REST + SSE
┌──────────────────────▼──────────────────────────────────────┐
│                 api (FastAPI + Python 3.11)                 │
│   - /notes, /hotspots, /agent/chat (SSE stream)             │
│   - /scene/:id metadata                                     │
│   - MCP server mounted at /mcp                              │
└──┬───────────────────┬─────────────────────────┬────────────┘
   │                   │                         │
   ▼                   ▼                         ▼
┌────────┐    ┌──────────────────┐    ┌────────────────────┐
│ postgres│    │   langgraph     │    │  gemini flash API   │
│+pgvector│    │     agent       │───▶│ (free tier, vision) │
└────────┘    └──────────────────┘    └────────────────────┘
                       │
                       │ calls MCP tools
                       ▼
              [ scene tools: render_view,
                list_objects, find_by_description,
                highlight_region, measure_distance,
                move_camera_to, plan_tour ]

┌────────────────────────────────────────────────────────────┐
│  (not wired in v1, scaffolded only) redis + celery worker  │
│  → reserved for async splat training pipeline              │
└────────────────────────────────────────────────────────────┘
```

### 4.2 Component responsibilities

- **web** — frontend, user-facing everything. Owns viewer, scrubber, chat UI, hotspot rendering.
- **api** — HTTP surface for the frontend, hosts the MCP server, orchestrates the LangGraph agent. Reads from Postgres.
- **postgres** — scenes, objects, hotspots, notes, listings. pgvector for CLIP embeddings.
- **langgraph agent** — in-process inside the api container. Plans tool calls against the MCP server.
- **MCP server** — thin Python wrapper exposing scene-manipulation tools. Consumed by LangGraph.
- **redis + celery worker** — scaffolded, idle in v1. Ready for v2 splat training pipeline.

---

## 5. Tech Stack (Locked)

### Frontend
- **Next.js 14** (App Router) + **TypeScript**
- **Three.js**
- **@mkkellogg/gaussian-splats-3d** (mature web splat viewer)
- **Tailwind CSS** + **shadcn/ui** for UI components
- **Zustand** for client state (viewer ↔ chat ↔ hotspots)
- **TanStack Query** for server state
- **react-player** for the scrubber video

### Backend
- **Python 3.11**
- **FastAPI** + **Pydantic v2**
- **asyncpg** (direct Postgres, no ORM — keep it lean)
- **LangGraph** (the agent)
- **mcp** (Python SDK for Model Context Protocol server)
- **openclip-torch** (CLIP embeddings, runs on CPU)
- **numpy**, **scipy** (post-processing)

### 3D scene source (no splat training required)
- **InteriorGS dataset** ([HuggingFace](https://huggingface.co/datasets/spatialverse/InteriorGS)) — one scene downloaded. Ships the pre-trained compressed Gaussian splat. Their `labels.json` is used **only as evaluation ground truth**, not as the product's label source.
- **SuperSplat-compressed .ply** format (the splat) — loaded directly by @mkkellogg/gaussian-splats-3d

### Our segmentation pipeline (runs in Track C, ~2 hours wall time)
- **Puppeteer + @mkkellogg/gaussian-splats-3d** — headless Chrome renders 30 views from the splat along a scripted camera path covering all rooms
- **MobileSAM** ([ChaoningZhang/MobileSAM](https://github.com/ChaoningZhang/MobileSAM)) — 2D instance mask generator, 5× faster than vanilla SAM on CPU
- **OpenCLIP ViT-B/32** — zero-shot classification of each mask against a curated 40-class home-object vocabulary
- **Our backprojection code (~80 lines)** — for each Gaussian center, project into each view using known camera pose; vote on class label by summing contributions across all views; assign winning class
- **DBSCAN (scikit-learn)** — cluster per-class Gaussians spatially → instance IDs
- **Evaluation (~40 lines)** — compare predicted per-instance labels to InteriorGS `labels.json` ground truth; compute per-class precision/recall/mIoU; dump `metrics.json` for the README

### Note matcher (unchanged)
- **OpenCLIP text embeddings** of object descriptions for note matcher (runs against *our* labels, with GT-label fallback)

### Data
- **PostgreSQL 16** + **pgvector** extension (CLIP embeddings indexed for cosine similarity)
- **Redis 7** (scaffolded, unused in v1 request path)

### LLM / VLM
- **Gemini 2.0 Flash** API (free tier — generous limits, has vision endpoint) as primary
- Fallback option in code: **Groq** free tier with Llama 3.3 (text-only) if Gemini quota exceeded

### Infra
- **Docker** + **Docker Compose**
- **Kubernetes** manifests (plain YAML, no Helm — keep it simple to demo on minikube)
- **GitHub Actions** — one workflow: lint (ruff + eslint) + typecheck (mypy + tsc) + build (`docker compose build`)

---

## 6. Data Model (PostgreSQL)

```sql
-- A scene is one processed listing
CREATE TABLE scenes (
    id               UUID PRIMARY KEY,
    title            TEXT NOT NULL,
    address          TEXT,
    splat_url        TEXT NOT NULL,         -- path to .ply file
    camera_trajectory JSONB NOT NULL,       -- list of {timestamp, pose_4x4}
    processed_at     TIMESTAMPTZ NOT NULL,
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

-- An object detected in the scene
CREATE TABLE scene_objects (
    id              UUID PRIMARY KEY,
    scene_id        UUID REFERENCES scenes(id) ON DELETE CASCADE,
    instance_id     INT NOT NULL,            -- from Gaussian Grouping
    class_name      TEXT NOT NULL,           -- from CLIP zero-shot
    room_label      TEXT,                    -- from spatial clustering
    centroid        DOUBLE PRECISION[3] NOT NULL,
    bbox_min        DOUBLE PRECISION[3] NOT NULL,
    bbox_max        DOUBLE PRECISION[3] NOT NULL,
    clip_embedding  VECTOR(512) NOT NULL     -- OpenCLIP ViT-B/32
);
CREATE INDEX ON scene_objects USING hnsw (clip_embedding vector_cosine_ops);

-- A note written by the realtor, timestamped to a moment in the video
CREATE TABLE notes (
    id            UUID PRIMARY KEY,
    scene_id      UUID REFERENCES scenes(id) ON DELETE CASCADE,
    text          TEXT NOT NULL,
    video_timestamp DOUBLE PRECISION NOT NULL,   -- seconds
    category      TEXT,                          -- feature|included|issue|info|spec|story
    category_confidence REAL,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- A hotspot is a note resolved to a specific 3D object
CREATE TABLE hotspots (
    id             UUID PRIMARY KEY,
    note_id        UUID UNIQUE REFERENCES notes(id) ON DELETE CASCADE,
    object_id      UUID REFERENCES scene_objects(id) ON DELETE SET NULL,
    match_confidence REAL NOT NULL,
    position       DOUBLE PRECISION[3] NOT NULL,  -- denormalized centroid for speed
    auto_accepted  BOOLEAN NOT NULL
);
```

---

## 7. Scene Setup (runs once, no GPU required)

Executed by a Python script `scripts/prepare_scene.py` in the repo.

```
Input:  one InteriorGS scene directory (downloaded via Hugging Face)
          ├── 3dgs_compressed.ply
          ├── labels.json           # per-object class + instance_id + 3D OBB
          ├── structure.json        # rooms, walls, doors, windows
          ├── occupancy.png         # grayscale navigability map
          └── occupancy.json        # occupancy metadata

Output: data/scene/demo/
          ├── 3dgs_compressed.ply     (unchanged, served to web viewer)
          ├── object_inventory.json   (our normalized form)
          ├── camera_trajectory.json  (synthesized — see below)
          ├── video.mp4               (synthesized walkthrough, see below)
          └── views/                  (canonical per-object renders for CLIP)
```

### Steps
1. **Download the scene** — `scripts/download_scene.sh` uses `huggingface-cli` to pull one InteriorGS scene (user accepts license on HF first, one-time). License acceptance documented in README.
2. **Parse `labels.json`** → for each object: extract `class_name`, `instance_id`, 8-corner OBB → compute `centroid`, `bbox_min`, `bbox_max` (axis-aligned).
3. **Assign room labels** by projecting each object centroid onto `structure.json` room polygons (2D point-in-polygon test). Output: every object has a `room_label` ("living_room", "master_bedroom", etc.).
4. **Render one canonical view per object** using a headless Three.js renderer (Node.js + Puppeteer) — position camera 2m in front of object bbox, facing centroid. Save `views/<instance_id>.jpg`.
5. **Compute CLIP embeddings** for each canonical view (OpenCLIP ViT-B/32, CPU, ~2 min total for a 100-object scene). Store as `object_inventory.json`.
6. **Synthesize walkthrough video** for the scrubber UI: generate a scripted camera trajectory (waypoints through rooms using `occupancy.png` for navigability), render frames via headless Three.js, encode to mp4 with ffmpeg. Save `video.mp4` + `camera_trajectory.json` (timestamp → pose lookup).

**Total time to run on a modest laptop: ~15–20 minutes.** One-off.

### Loaded into Postgres at startup
On `api` container startup: read `object_inventory.json` + `camera_trajectory.json` → insert into `scenes`, `scene_objects` (if not already present). Also seed `notes` from `data/scene/demo/demo_notes.json` (hardcoded demo notes with timestamps).

---

## 8. Hotspot Generation

Performed at note-creation time.

```python
def create_hotspot_for_note(note):
    # 1. Get camera pose at note's video timestamp
    pose = scene.camera_trajectory.pose_at(note.video_timestamp)

    # 2. Filter objects visible from that pose (frustum test)
    visible_objects = [o for o in scene_objects
                       if in_frustum(o.centroid, pose, fov_deg=60)]

    # 3. CLIP-rank visible objects by similarity to note text
    note_embedding = clip_model.encode_text(note.text)
    ranked = sorted(visible_objects,
                    key=lambda o: cosine_similarity(note_embedding, o.clip_embedding),
                    reverse=True)

    if not ranked:
        return None  # nothing visible — flag for realtor review (v2)

    best = ranked[0]
    confidence = float(cosine_similarity(note_embedding, best.clip_embedding))

    # 4. LLM classifies note category (one API call)
    category, cat_conf = classify_note_category(note.text)
    note.category = category
    note.category_confidence = cat_conf

    # 5. Create hotspot
    return Hotspot(
        note_id=note.id,
        object_id=best.id,
        match_confidence=confidence,
        position=best.centroid,
        auto_accepted=(confidence >= 0.5),  # v1: permissive
    )
```

**v1 confidence threshold: 0.5** (permissive — we want all hotspots to auto-accept for the demo). A review UI comes in v2.

---

## 9. Agent + MCP Server

### 9.1 MCP tools (Python, exposed by `api`)

| Tool | Signature | Returns |
|---|---|---|
| `list_objects(scene_id, room=None, class_name=None)` | filters | `[{id, class_name, room, centroid, bbox}]` |
| `list_hotspots(scene_id, category=None)` | filters | `[{id, note_text, category, position, object_class}]` |
| `find_by_description(scene_id, text)` | free-text query | `[{object_id, score}]` (top-5 via pgvector) |
| `render_view(scene_id, camera_pose)` | 4x4 matrix or `{position, target}` | `{image_base64, width, height}` |
| `highlight_region(object_ids \| bbox)` | — | UI event — viewer draws highlight |
| `measure_distance(scene_id, point_a, point_b)` | two 3D points | `{meters}` |
| `move_camera_to(scene_id, pose_or_object_id)` | target | UI event — viewer animates camera |
| `plan_tour(scene_id, focus=None)` | optional focus ("features", "issues") | `[{pose, dwell_seconds, narration}]` |
| `execute_tour(tour_plan)` | plan from above | streams camera moves + narration |

### 9.2 LangGraph flow

```
   START
     │
     ▼
[classify intent]  ← "question" | "tour request" | "find" | "other"
     │
     ▼
[plan tool calls]  ← LLM decides which tools; may use render_view for visual grounding
     │
     ▼
[execute tool]  ───► [tool result]
     │                    │
     └─── loop ◄──────────┘  (up to 5 iterations, then force-answer)
     │
     ▼
[compose answer]  ← LLM synthesizes final message
     │
     ▼
    END
```

**Free-tier LLM:** Gemini 2.0 Flash (1M context, vision support, generous free tier).

### 9.3 VLM grounding pattern (the novel bit)

When the agent cannot answer from metadata alone (e.g., "is the kitchen cozy?"), it calls `render_view` with a chosen pose, gets back a base64 PNG, and passes that image to a follow-up Gemini Flash vision call ("describe what you see relevant to: [original question]"). The text result feeds back into the agent's reasoning.

**Example conversation trace:**
```
User: "Is there space for a 2m couch against the back wall of the living room?"
Agent:
  → find_by_description("back wall living room")  → wall object at pose P
  → measure_distance(corner1, corner2)            → 3.4m
  → render_view(facing that wall)                 → image
  → [VLM: "clear floor space, no blocking furniture"]
  → compose: "Yes — 3.4m of clear floor. Highlighting the region now."
  → highlight_region(bbox)                        → UI shows outline
```

---

## 10. Frontend Layout

### Pages
- `/` — homepage: list of demo scenes (v1: one scene).
- `/listing/[id]` — the 3D viewer with chat overlay (buyer side).
- `/listing/[id]/admin` — scrubber UI for notes + hotspot status (realtor side).

### Key components
- `<SplatViewer/>` — wraps @mkkellogg/gaussian-splats-3d, exposes imperative API for camera moves + highlights.
- `<HotspotMarkers/>` — renders one `<Sprite/>` per hotspot, category icon, click → popup.
- `<ChatOverlay/>` — streaming chat via SSE, renders agent messages + tool-call indicators.
- `<ScrubberNotesUI/>` — video player + text input + list of created notes.
- `<HighlightLayer/>` — Three.js overlay for agent-triggered highlights (wireframe bboxes, arrows).

### Viewer ↔ Chat communication
Zustand store with:
```ts
interface ViewerStore {
  currentPose: Mat4;
  highlightedObjectIds: string[];
  cameraAnimation: CameraPath | null;
  setPose: (pose: Mat4) => void;
  setHighlights: (ids: string[]) => void;
  setCameraAnimation: (path: CameraPath) => void;
}
```
Agent tool calls post to this store; the viewer reacts.

---

## 11. Infra

### Docker Compose (`docker-compose.yml`)
Services: `web`, `api`, `postgres`, `redis`.
- `postgres` initialized with pgvector extension.
- `api` mounts `data/scene/` read-only for splat + inventory.
- `web` served in dev mode for hot reload.

### Kubernetes (`k8s/*.yaml`)
Separate YAMLs for: `namespace`, `postgres-statefulset`, `postgres-pvc`, `postgres-service`, `redis-deployment`, `redis-service`, `api-deployment`, `api-service`, `web-deployment`, `web-service`, `ingress` (nginx).
- Built images pushed to GitHub Container Registry (ghcr.io) via CI.
- Runs on minikube with `kubectl apply -f k8s/`.
- Demo video includes a 20-second minikube segment (`kubectl get pods`, `kubectl port-forward`, browser loads listing).

### GitHub Actions (`.github/workflows/ci.yml`)
- On push: lint (ruff, eslint), typecheck (mypy, tsc), build (docker compose build).
- On tag: build + push images to ghcr.io.

---

## 12. Demo Video Plan (2–3 minutes)

**Shot list:**

| # | Duration | Content |
|---|---|---|
| 1 | 10s | Problem hook: Zillow listing with 30 static photos, buyer emailing seller "does the desk come with it?" |
| 2 | 10s | Title card: "SceneAgent — every listing, a 3D digital twin, with an AI concierge" |
| 3 | 15s | Realtor uploads a video. Fake processing screen ("splat training…segmenting objects…"). Cuts to finished scene. |
| 4 | 25s | Realtor scrubs video, types notes: "window sticks" at 2:30, "desk is included" at 1:15, "heated floor" at 3:45. Hotspots appear in 3D on the right. |
| 5 | 15s | Switch to buyer view. 3D scene. Orbit, click a 🎁 hotspot → "Desk is included." Click a ⚠️ hotspot → "Window sticks, pull firmly." |
| 6 | 40s | Open chat. "Are there any issues I should know about?" → agent lists ⚠️ hotspots, highlights them in 3D. "How tall are the ceilings?" → agent measures live, shows 3.2m. "Give me a tour of the best features." → guided tour, camera flies, narration. |
| 7 | 15s | Infra cut: terminal with `kubectl apply -f k8s/`, pods green, `kubectl port-forward`, browser loads. |
| 8 | 10s | Close: GitHub URL, tech stack badges (Python, Next.js, Postgres, pgvector, Gaussian Splatting, LangGraph, MCP, Kubernetes). |

**Recording tooling:** OBS Studio (free) for screen capture, DaVinci Resolve (free) for edit, your own voiceover, royalty-free background music (YouTube Audio Library).

---

## 13. Success Criteria (v1 is "done" when)

1. `docker compose up` starts all services with no manual intervention.
2. Visiting `http://localhost:3000/listing/demo` loads the pre-computed Gaussian splat in the browser and shows pre-authored hotspots.
3. The scrubber admin page (`/listing/demo/admin`) accepts new notes and they appear as hotspots in the viewer within 3 seconds.
4. The chat works for these specific prompts: "List any issues", "What's included?", "How tall are the ceilings?", "Give me a tour."
5. The agent demonstrates VLM grounding at least once in the demo conversation (i.e., actually calls `render_view` and uses the vision output).
6. `kubectl apply -f k8s/` on minikube brings the full stack up and the app is reachable via port-forward.
7. README has: architecture diagram, setup instructions, v2 roadmap, demo video link, link to Gaussian Grouping and other third-party credits.
8. GitHub Actions CI is green.
9. Demo video is uploaded (YouTube unlisted) and linked in README.

---

## 14. Risks + Mitigations (decided upfront)

| Risk | Mitigation |
|---|---|
| InteriorGS license acceptance flow blocks immediate access | License is free, one-click acceptance on Hugging Face. README includes screenshots of the acceptance flow. Fallback: SceneSplat-7K ([GaussianWorld/scene_splat_7k](https://huggingface.co/datasets/GaussianWorld/scene_splat_7k)) is a similar ScanNet-derived dataset with labels. |
| SuperSplat-compressed .ply format not loadable by @mkkellogg/gaussian-splats-3d | @mkkellogg supports SuperSplat format natively. Fallback: decompress with the SuperSplat CLI to standard 3DGS .ply. |
| Synthesized walkthrough video feels fake in the demo | The synthesized video uses the occupancy map for realistic paths; in the demo video we show the scrubber working on it. README notes: "Production accepts user-uploaded video; for the demo the walkthrough is synthesized from the InteriorGS occupancy map so the scrubber UI has something to scrub against." |
| Gemini Flash free tier quota exceeded mid-demo | Code path has a Groq fallback for text-only; demo is short enough to stay under quota. |
| pgvector not available in our chosen Postgres image | Use `pgvector/pgvector:pg16` image (official, Docker Hub). |
| Minikube demo fails live | Record the minikube segment in advance; use pre-recorded clip in demo video. |

---

## 15. Post-v1 Roadmap (v2 items, README only)

1. Voice narration during capture (Whisper → timestamped notes).
2. Realtor review UI (swipe-approve low-confidence matches).
3. Adaptive re-rendering for borderline matches.
4. Real scene editing (move/remove objects — Gaussian Grouping supports this natively, we just haven't wired UI).
5. Multi-scene comparison ("which apartment has better light?").
6. Buyer-added questions → seller queue.
7. Move-in furniture placement simulator.
8. Multi-tenant listings database + auth.
9. Live splat training pipeline (wire the Celery worker to a GPU node pool).

---

## 16. File/Folder Layout

```
sceneagent/
├── README.md
├── docker-compose.yml
├── .env.example
├── .github/workflows/ci.yml
├── k8s/
│   ├── namespace.yaml
│   ├── postgres-*.yaml
│   ├── redis-*.yaml
│   ├── api-*.yaml
│   ├── web-*.yaml
│   └── ingress.yaml
├── data/
│   └── scene/demo/           # prepared InteriorGS scene (gitignored, fetched via script)
│       ├── 3dgs_compressed.ply
│       ├── object_inventory.json
│       ├── camera_trajectory.json
│       ├── views/            # canonical per-object renders
│       ├── video.mp4         # synthesized walkthrough for scrubber UI
│       └── demo_notes.json   # seed notes with timestamps
├── scripts/
│   ├── download_scene.sh     # fetch one InteriorGS scene from HF
│   └── prepare_scene.py      # parse labels → object_inventory, render views, compute CLIP, synthesize walkthrough
├── api/                        # Python FastAPI service
│   ├── pyproject.toml
│   ├── Dockerfile
│   ├── src/sceneagent/
│   │   ├── main.py
│   │   ├── routes/{scenes.py, notes.py, hotspots.py, chat.py}
│   │   ├── db.py
│   │   ├── models.py
│   │   ├── matcher.py         # note → hotspot matcher
│   │   ├── clip_utils.py
│   │   ├── mcp_server.py      # MCP tools
│   │   ├── agent/
│   │   │   ├── graph.py       # LangGraph state graph
│   │   │   └── tools.py       # tool call handlers
│   │   └── vlm.py             # Gemini Flash wrapper
│   └── tests/
├── web/                        # Next.js frontend
│   ├── package.json
│   ├── Dockerfile
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx
│   │   │   └── listing/[id]/
│   │   │       ├── page.tsx           # buyer view
│   │   │       └── admin/page.tsx     # realtor scrubber
│   │   ├── components/
│   │   │   ├── SplatViewer.tsx
│   │   │   ├── HotspotMarkers.tsx
│   │   │   ├── ChatOverlay.tsx
│   │   │   ├── ScrubberNotesUI.tsx
│   │   │   └── HighlightLayer.tsx
│   │   ├── lib/api.ts
│   │   └── stores/viewer.ts
│   └── public/
└── docs/
    └── superpowers/
        ├── specs/
        │   └── 2026-04-20-sceneagent-design.md   # this file
        └── plans/
            └── 2026-04-20-sceneagent-plan.md     # next: writing-plans output
```

---

## Appendix A: Reference Repositories

- [InteriorGS dataset](https://huggingface.co/datasets/spatialverse/InteriorGS) — **primary scene source.** 1,000 pre-trained Gaussian splats with per-object class labels, instance IDs, 3D OBBs, floorplans, occupancy maps.
- [InteriorGS GitHub](https://github.com/manycore-research/InteriorGS) — dataset toolkit.
- [SceneSplat-7K](https://huggingface.co/datasets/GaussianWorld/scene_splat_7k) — fallback dataset (ScanNet/Replica-based).
- [Gaussian Grouping](https://github.com/lkeab/gaussian-grouping) — ECCV 2024, referenced for v2 production upload flow.
- [SAGS](https://github.com/XuHu0529/SAGS) — training-free SAM lifting (backup plan).
- [LangSplat](https://github.com/minghanqin/LangSplat) — CVPR 2024, open-vocabulary splat querying (reference).
- [Nerfstudio](https://docs.nerf.studio/) — reference framework for splat workflows.
- [@mkkellogg/gaussian-splats-3d](https://github.com/mkkellogg/GaussianSplats3D) — web splat viewer.
- [pgvector](https://github.com/pgvector/pgvector) — vector similarity in Postgres.
- [mcp (Python SDK)](https://github.com/modelcontextprotocol/python-sdk) — MCP server implementation.
- [LangGraph](https://langchain-ai.github.io/langgraph/) — agent framework.
- [huggingface-cli](https://huggingface.co/docs/huggingface_hub/guides/cli) — used to download the InteriorGS scene.
