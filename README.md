# SceneAgent

Anyone records a short walkthrough of a space — or hands us a point cloud,
a set of photos, a BIM model, or a Gaussian-splat file — and SceneAgent
gives them back an interactive 3D viewer of that room with every object
detected, named, and addressable by an AI concierge that lives inside
the scene.

![Next.js](https://img.shields.io/badge/Next.js-14-black?logo=nextdotjs)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi)
![Postgres + pgvector](https://img.shields.io/badge/Postgres-16%20%2B%20pgvector-336791?logo=postgresql)
![Gaussian Splatting](https://img.shields.io/badge/3DGS-gsplat%20CUDA-ff6b35)
![SAM 3](https://img.shields.io/badge/Segmentation-SAM%203-5a67d8)
![open-clip](https://img.shields.io/badge/open--clip-ViT--B%2F32-1e40af)
![LangGraph](https://img.shields.io/badge/LangGraph-agent-34d399)
![MCP](https://img.shields.io/badge/MCP-Anthropic-8b5cf6)
![Qwen 3 on Groq](https://img.shields.io/badge/Agent%20LLM-Qwen%203%2032B%20%C2%B7%20Groq-4285f4)
![Kubernetes](https://img.shields.io/badge/Kubernetes-minikube-326ce5?logo=kubernetes)

The demo scene is an interior wine bar, served as a 3DGS Gaussian splat with
**151 objects detected and segmented in 3D** by the SceneAgent pipeline,
each clickable in the browser viewer with a wireframe bounding box and a
class label. The same pipeline is built to accept any of the four input
types above with a per-type sandbox UI (see [Roadmap](#roadmap)).




https://github.com/user-attachments/assets/46024f01-cde6-4ab4-a413-8e08ec244e3c










---

## Architecture

```
                   ┌───────────────────────┐
                   │     user browser      │
                   │  splat viewer + chat  │
                   └──────────┬────────────┘
                              │  HTTP / SSE
                              ▼
                   ┌───────────────────────┐
                   │  web  (Next.js 14)    │
                   │  splat viewport,      │
                   │  inventory sidebar,   │
                   │  bbox overlay         │
                   └──────────┬────────────┘
                              │  REST + SSE
                              ▼
                   ┌───────────────────────┐
                   │  api  (FastAPI)       │
                   │  /scenes /detections  │
                   │  /hotspots /metrics   │
                   │  /chat (SSE)          │
                   └──┬──────────┬─────────┘
                      │          │
           reads/     │          │  LangGraph agent
           writes     │          │  calls scene tools
                      ▼          ▼
          ┌─────────────────┐  ┌──────────────────────┐
          │ postgres 16     │  │ scene tools (MCP):   │
          │ + pgvector      │  │  list_objects        │
          │ (scenes, objects│  │  find_by_description │
          │  notes, hotspots│  │  measure_distance    │
          │  CLIP embedings)│  │  highlight_region    │
          └─────────▲───────┘  │  move_camera_to      │
                    │          │  plan_tour           │
        seeded by   │          └──────────┬───────────┘
        the CV      │                     │ image / text
        pipeline    │                     ▼
                    │          ┌──────────────────────┐
          ┌─────────┴───────┐  │  Qwen 3 32B (Groq)   │
          │ pipeline (CV)   │  │  pluggable, free     │
          │ splat ingest →  │  └──────────────────────┘
          │ render → SAM 3 →│
          │ 3D backproject  │
          │ → object DB     │
          └─────────────────┘
```

---

## What the pipeline does, end to end

The pipeline is **input-agnostic**. Whatever the user uploads — a phone
walkthrough video, a folder of phone photos, a LiDAR point cloud, a Revit
or IFC BIM model, or an already-trained Gaussian splat — the back end
converges on the same intermediate format (a standard 3DGS `.ply`) and
runs identical post-processing on top.

Stages, in order:

### 1. Input ingestion → Gaussian splat
- **Video.** Extract frames at a chosen FPS, run COLMAP or MASt3R for
  camera-pose recovery, train a 3DGS scene with `gsplat`. *(In progress;
  the upload sandbox for videos is on the roadmap below.)*
- **Photo set.** Same as above without the frame-extraction step. The
  user just drops 30–300 photos into the sandbox.
- **Point cloud (LAS / PLY / E57).** Densify and convert to a Gaussian
  splat by initializing one Gaussian per point and refining via the
  `gsplat` densification loop.
- **BIM model.** Render the BIM mesh from a synthetic camera array,
  then treat the renders as the photo-set path.
- **Pre-trained splat.** Skip directly to stage 2 — we already have the
  Gaussians, we just need to standardize the file format.

### 2. Splat standardisation
`pipeline/src/npz_to_ply.py` converts whatever splat format we received
(decoded compressed-PLY, NPZ, etc.) into the **standard 14-channel 3DGS
PLY**: `xyz`, `nx ny nz`, `f_dc_0..2` (DC spherical-harmonic colour),
logit `opacity`, log `scale_0..2`, and `rot_0..3` quaternion. From here
on every consumer in the pipeline reads the same thing.

### 3. Camera-array placement (the "where do we look?" problem)
Object-density driven by default: k-means over the scene-occupancy points
gives N camera-array centres, each placed 2 m back at eye height looking
at the cluster centre. For interactive control we ship a **Blender add-on**
under `Camera_array_tool/` (extension of Olli Huttunen's *Camera Array
Tool*) with prebuilt array shapes (HalfDome / Cylinder / InteriorTower /
MinAngle17 / MinAngle26). The workflow:

  1. Convert the splat to a colored point cloud
     (`pipeline/src/ply_to_pointcloud.py`) so any vanilla Blender PLY
     importer eats it.
  2. Spawn a pre-made array object in Blender and drag/scale it inside
     the splat where the user wants coverage.
  3. **SceneAgent Export → "Export to SceneAgent (camera_poses.json)"** —
     the addon writes a JSON file the rest of the pipeline picks up.

### 4. Photoreal view rendering
`pipeline/src/render_gsplat.py` runs **gsplat's CUDA rasterizer** at the
exported camera poses. 30 views at 800×600 RGB+depth render in ~3 s on
an RTX 4060 Laptop. OpenCV view convention; depth comes out in metres
for stage 6.

### 5. Open-vocabulary 2D segmentation
`pipeline/src/segment_sam3.py` feeds each rendered view to **SAM 3**
([facebookresearch/sam3](https://github.com/facebookresearch/sam3)).
SAM 3 is text-promptable: we hand it a class name string ("wine bottle",
"high chair", "chandelier", …) and it returns 2D instance masks for
every match in the image, with a confidence score. No separate
classification step needed; the prompt **is** the label.

### 6. Backprojection — 2D masks → 3D Gaussians
`pipeline/src/backproject.py` is the bridge from picture-space to
world-space. For every Gaussian in the splat (typically 700k+):

  - project its 3D centre into every view via the known camera matrix
  - if the projected pixel falls inside a SAM 3 mask, the Gaussian gets
    a vote for that mask's class
  - votes are weighted by `mask_area / median_area` so a giant ceiling
    mask doesn't drown out a small wine-bottle mask

After all views, each Gaussian's predicted class is whichever class won
the most votes.

### 7. Instance clustering
Same-class Gaussians that are close in 3D get grouped via **DBSCAN**
into one instance per cluster. We compute an axis-aligned bounding box
per instance from its member Gaussians' positions. The output is
`object_inventory.json` — a flat list of `{instance_id, class_name,
bbox_min, bbox_max, centroid, point_count}`.

### 8. Serving
`pipeline/src/seed_db.py` loads the inventory into Postgres alongside a
CLIP text embedding per object (used by the chat agent's
`find_by_description` tool for natural-language lookup like *"the green
wine bottle by the window"*). The web viewer's right-side **Inventory**
sidebar reads `/scenes/:slug/detections` directly and overlays the
hovered/selected object's bounding box on the splat in real time.

---

## The product surface

What ships in the browser, beyond the pipeline:

- **3D Gaussian-splat viewport** rendered in real time via
  `@mkkellogg/gaussian-splats-3d` with custom **Blender-style fly
  controls** — `W`/`S` forward, `A`/`D` strafe, `Q`/`E` world-up/down,
  drag-to-look. The viewport completely freezes camera input the moment
  any text input is focused (chat, filter), and resumes on blur or
  `Esc` — so typing never accidentally moves the scene.
- **Inventory sidebar** listing every detected object grouped by class,
  with a real-time **class filter**, per-row confidence chips, an
  optional auto-fly toggle, and an explicit ✈ fly-to button on each
  row. Clicking a row highlights without yanking the camera by default.
- **Live bounding-box + label overlay** rendered as an SVG/HTML layer
  over the WebGL canvas. The 12 edges of the active object's bbox plus
  a `class · NN%` text chip are projected from 3D to 2D on every frame
  using the viewer's camera matrix, so the highlight stays locked to
  the object as you fly around. Hover / click in the sidebar drives it.
- **Agentic AI concierge** — a LangGraph single-node agent inside the
  FastAPI process exposes scene tools through MCP (`list_objects`,
  `find_by_description`, `measure_distance`, `plan_tour`,
  `render_view`, `describe_image`, …). The agent does not carry the
  scene in its context — it queries the back end via tools.
- **Pluggable LLM provider.** Three env vars (`LLM_API_KEY`,
  `LLM_BASE_URL`, `LLM_MODEL`) point the agent at any
  OpenAI-compatible endpoint — Groq, OpenRouter, DeepSeek, Moonshot,
  Together AI, your own. Default config uses **Qwen 3 32B on Groq**
  (truly free, no card). Falls back to Gemini if those env vars are
  unset, then to a rule-based heuristic so the API surface always
  responds. Reasoning-model preambles (`<think>...</think>`) are
  stripped before JSON parsing.
- **Chat surface** collapsed behind a small icon by default; opens as a
  slide-up sheet that doesn't cover the splat.

---

## Roadmap

The current branch ships the splat → segmentation → viewer path against
a pre-trained input. Next, **per-input-type upload sandboxes** so a
non-technical user can bring their own scene:

- **Video sandbox.** Drag-drop a phone walkthrough; we extract frames,
  show pose recovery progress, train the splat, and drop the user into
  the same viewer flow.
- **Photo-set sandbox.** Same back end as video, skipping frame
  extraction. Live preview of the COLMAP graph as photos are added.
- **Point-cloud sandbox.** Upload a `.ply` / `.las` / `.e57`; the
  Gaussian initialiser runs server-side and the user picks a sampling
  density.
- **BIM sandbox.** Upload IFC or Revit, pick rooms to include, render a
  synthetic camera array, then converge on the photo-set path.
- **Pre-trained splat sandbox.** Direct upload of a 3DGS PLY (today's
  demo path) plus a browser-side camera array tool that mirrors the
  Blender add-on for users without Blender installed.

Other planned work:

- **Streaming chat.** SSE is wired; stream the agent's tokens through.
- **Per-Gaussian instance IDs** (Gaussian Grouping–style) so the splat
  itself carries object identity instead of needing a sidecar
  inventory file.

---

## Quick start

You need Docker + Docker Compose, a HuggingFace token (the demo splat
and SAM 3 weights sit behind one license accept each), and a free Groq
API key for the agent's LLM (or any OpenAI-compatible key).

1. **Clone the repo.**
   ```bash
   git clone https://github.com/<you>/sceneagent.git
   cd sceneagent
   ```
2. **Configure environment.**
   ```bash
   cp .env.example .env
   # edit .env:
   #   HF_TOKEN=hf_...
   #   LLM_API_KEY=gsk_...                     # from console.groq.com
   #   LLM_BASE_URL=https://api.groq.com/openai/v1
   #   LLM_MODEL=qwen/qwen3-32b
   ```
3. **Download the demo scene.**
   ```bash
   ./scripts/download_scene.sh 0003_839989
   ```
4. **Bring up the stack.**
   ```bash
   docker compose up --build
   ```
5. **Open the listing.**
   Visit <http://localhost:3000>. Hover an object in the right sidebar —
   its bounding box highlights in 3D. Open the chat to talk to the
   concierge.

---

## Kubernetes demo

The same stack runs on minikube with plain YAML — no Helm. See
[`k8s/README.md`](./k8s/README.md), or:

```bash
cp k8s/99-secrets.example.yaml k8s/99-secrets.yaml
$EDITOR k8s/99-secrets.yaml            # paste LLM_API_KEY (+ HF_TOKEN)
kubectl apply -f k8s/99-secrets.yaml
./k8s/apply.sh
```

`minikube start` must already be running.

---

## Credits

- **Dataset:** [InteriorGS](https://huggingface.co/datasets/spatialverse/InteriorGS) — pre-trained indoor Gaussian splats.
- **Splat rasterizer:** [`gsplat`](https://github.com/nerfstudio-project/gsplat) (Nerfstudio Project) — CUDA 3DGS renderer.
- **Web viewer:** [@mkkellogg/gaussian-splats-3d](https://github.com/mkkellogg/GaussianSplats3D).
- **2D segmentation:** [SAM 3](https://github.com/facebookresearch/sam3) (Meta Superintelligence Labs, 2026).
- **CLIP weights:** [OpenAI CLIP](https://github.com/openai/CLIP) ViT-B/32 via [open-clip](https://github.com/mlfoundations/open_clip).
- **Camera-array authoring:** Olli Huttunen's *Camera Array Tool* Blender add-on, extended with a SceneAgent JSON exporter.
- **Agent framework:** [LangGraph](https://langchain-ai.github.io/langgraph/).
- **Tool protocol:** Anthropic's [Model Context Protocol](https://modelcontextprotocol.io/).
- **Agent LLM:** [Qwen 3 32B](https://huggingface.co/Qwen/Qwen3-32B) (Alibaba) hosted free on [Groq](https://groq.com/) — pluggable to any OpenAI-compatible endpoint via three env vars.
- **Vector search:** [pgvector](https://github.com/pgvector/pgvector).

---

## License

MIT (TBD).
