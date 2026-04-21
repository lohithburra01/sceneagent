
https://github.com/user-attachments/assets/7ccb0068-5ded-4579-a503-195164e4443a
# SceneAgent

**An end-to-end pipeline that turns any 3D capture into a semantically queryable scene — with an agentic AI concierge that actually looks at the space when it needs to answer a question.**

Hand SceneAgent a phone walkthrough video, a photo set, a LiDAR point cloud, a BIM model, or a pre-trained Gaussian splat — and it gives you back a live 3D viewer where every object is detected, named, located in world space, and addressable by a conversational AI agent through a set of scene tools exposed via the Model Context Protocol (MCP).

The agent does not hallucinate from stale context. It queries the back end in real time — moving the camera, finding objects by natural-language description, measuring distances between points, and planning guided tours through the space.

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

---

## The problem this solves

Most 3D viewers stop at rendering. You can orbit the scene, zoom in, look around — but the viewer has no idea what it is looking at. The scene is opaque. You cannot ask it anything, filter by object category, find a specific item by description, or get a guided tour of what matters.

SceneAgent fixes that by treating the 3D scene as a structured database, not a render target. Every object gets a world-space bounding box, a class label, a confidence score, and a CLIP embedding. The AI agent queries that database in real time — and when a question requires visual grounding, it renders a view on demand and passes it to a vision-language model rather than guessing.

The pipeline is domain-agnostic. It works on commercial interiors, residential spaces, retail environments, industrial facilities, cultural heritage sites — any indoor space that can be captured as a Gaussian splat.

---

## Architecture

```
                   +----------------------------+
                   |       user browser         |
                   |  splat viewer + chat       |
                   +-------------+--------------+
                                 | HTTP / SSE
                                 v
                   +----------------------------+
                   |    web  (Next.js 14)       |
                   |  splat viewport,           |
                   |  inventory sidebar,        |
                   |  bbox overlay              |
                   +-------------+--------------+
                                 | REST + SSE
                                 v
                   +----------------------------+
                   |    api  (FastAPI)          |
                   |  /scenes /detections       |
                   |  /hotspots /metrics        |
                   |  /chat (SSE)               |
                   +--------+----------+--------+
                            |          |
               reads/       |          |  LangGraph agent
               writes       |          |  calls scene tools
                            v          v
          +-----------------+   +----------------------------+
          | postgres 16     |   | scene tools (MCP):        |
          | + pgvector      |   |  list_objects             |
          | (scenes, objects|   |  find_by_description      |
          |  notes, hotspots|   |  measure_distance         |
          |  CLIP embeddings|   |  highlight_region         |
          +---------^-------+   |  move_camera_to           |
                    |           |  plan_tour                |
        seeded by   |           |  render_view              |
        the CV      |           |  describe_image           |
        pipeline    |           +------------+--------------+
                    |                        | image / text
                    |                        v
                    |           +----------------------------+
          +---------+-------+   |  Qwen 3 32B (Groq)        |
          | pipeline (CV)   |   |  pluggable -- any         |
          | splat ingest -> |   |  OpenAI-compatible        |
          | render -> SAM 3 |   |  endpoint                 |
          | 3D backproject  |   +----------------------------+
          | -> object DB    |
          +-----------------+
```

---

## Pipeline -- what happens end to end

The pipeline is **input-agnostic**. Whatever arrives -- a phone walkthrough video, a photo set, a LiDAR point cloud, a BIM model, or a pre-trained Gaussian splat -- the back end normalises it into a standard 3DGS `.ply` and runs identical post-processing on top.

### 1. Input ingestion -> Gaussian splat

- **Video.** Extract frames at a chosen FPS, run COLMAP or MASt3R for camera-pose recovery, train a 3DGS scene with `gsplat`. *(Upload sandbox in roadmap.)*
- **Photo set.** Same as video without frame extraction. Drop 30-300 photos.
- **Point cloud (LAS / PLY / E57).** Densify and convert to a Gaussian splat by initialising one Gaussian per point and refining via the `gsplat` densification loop.
- **BIM model.** Render the BIM mesh from a synthetic camera array, then treat renders as the photo-set path.
- **Pre-trained splat.** Skip to stage 2 -- standardise the file format and continue.

### 2. Splat standardisation

`pipeline/src/npz_to_ply.py` converts any input splat format into the **standard 14-channel 3DGS PLY**: `xyz`, `nx ny nz`, `f_dc_0..2` (DC spherical-harmonic colour), logit `opacity`, log `scale_0..2`, and `rot_0..3` quaternion. Every downstream consumer reads the same format.

### 3. Camera-array placement

Object-density driven: k-means over the scene-occupancy points gives N cluster centres; cameras are placed 2 m back at eye height looking at each centre.

For interactive control, a **Blender add-on** ships under `Camera_array_tool/` with prebuilt array shapes (HalfDome, Cylinder, InteriorTower, MinAngle17/26). Workflow: convert the splat to a coloured point cloud, import into Blender, drag/scale the array inside the splat, then **SceneAgent Export -> Export to SceneAgent (camera_poses.json)**. The add-on writes the JSON the pipeline picks up directly.

### 4. Photoreal view rendering

`pipeline/src/render_gsplat.py` runs **gsplat's CUDA rasterizer** at the exported camera poses. 30 views at 800x600 RGB+depth render in ~3 s on an RTX 4060 Laptop. Depth comes out in metres for stage 6.

### 5. Open-vocabulary 2D segmentation

`pipeline/src/segment_sam3.py` feeds each rendered view to **SAM 3** (Meta Superintelligence Labs, 2026). SAM 3 is text-promptable: we pass a class name string -- "wine bottle", "high chair", "chandelier" -- and it returns 2D instance masks with confidence scores. No separate classification step; the prompt is the label.

### 6. Backprojection -- 2D masks -> 3D Gaussians

`pipeline/src/backproject.py` bridges picture-space to world-space. For every Gaussian in the splat (typically 700k+):

- Project its 3D centre into every view via the known camera matrix
- If the projected pixel falls inside a SAM 3 mask, the Gaussian gets a vote for that mask's class
- Votes are weighted by `mask_area / median_area` so large ceiling masks do not drown out small objects

Each Gaussian's predicted class is whichever class won the most votes across all views.

### 7. Instance clustering

Same-class Gaussians that are close in 3D get grouped via **DBSCAN** into one instance per cluster. An axis-aligned bounding box is computed per instance from its member Gaussians' positions. Output: `object_inventory.json` -- a flat list of `{instance_id, class_name, bbox_min, bbox_max, centroid, point_count}`.

### 8. Serving

`pipeline/src/seed_db.py` loads the inventory into Postgres alongside a CLIP text embedding per object. The web viewer reads `/scenes/:slug/detections` and overlays the selected object's bounding box on the splat in real time.

---

## Demo scene

The current demo runs against a pre-trained interior wine bar Gaussian splat from the [InteriorGS](https://huggingface.co/datasets/spatialverse/InteriorGS) dataset. Running the full pipeline against this scene -- 30 rendered views, SAM 3 segmentation, backprojection, DBSCAN clustering -- produces **151 detected object instances** across categories including seating, lighting, bar fixtures, glassware, and architectural elements, each with a world-space bounding box visible and clickable in the browser viewer.

https://github.com/user-attachments/assets/46024f01-cde6-4ab4-a413-8e08ec244e3c


https://github.com/user-attachments/assets/21a7d9e2-7e16-43c0-8184-7ca2a7e8491d


---

## What ships in the browser

- **3D Gaussian-splat viewport** via `@mkkellogg/gaussian-splats-3d` with custom **Blender-style fly controls** -- `W`/`S` forward-back, `A`/`D` strafe, `Q`/`E` world up-down, drag-to-look. Camera input freezes the moment any text field is focused and resumes on blur or `Esc`.
- **Inventory sidebar** listing every detected object grouped by class, with real-time class filter, per-row confidence chips, an optional auto-fly toggle, and an explicit fly-to button per row. Clicking a row highlights without moving the camera by default.
- **Live bounding-box + label overlay** -- an SVG/HTML layer over the WebGL canvas. The 12 edges of the active object's bounding box plus a `class . NN%` chip are projected from 3D to 2D on every frame using the viewer's camera matrix, so the highlight stays locked to the object as you fly.
- **Agentic AI concierge** -- a LangGraph agent inside the FastAPI process exposes scene tools through MCP. The agent can move the camera, find objects by free-text description, measure distances, list inventory by category, plan guided tours, and render views on demand for visual grounding. It does not carry the scene in its context window.
- **Pluggable LLM provider.** Three env vars (`LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`) point the agent at any OpenAI-compatible endpoint -- Groq, OpenRouter, DeepSeek, Together AI, your own. Default: **Qwen 3 32B on Groq** (free, no card required). Falls back to a rule-based heuristic so the API surface always responds.
- **Chat surface** collapsed behind a small icon by default; opens as a slide-up sheet that does not cover the splat.

---

## Roadmap

The current branch ships the splat -> segmentation -> viewer path against a pre-trained input. Next: **per-input-type upload sandboxes** so any user can bring their own scene.

- **Video sandbox.** Drag-drop a phone walkthrough; extract frames, show pose recovery progress, train the splat, drop into the viewer.
- **Photo-set sandbox.** Same back end, skip frame extraction. Live preview of the COLMAP graph as photos are added.
- **Point-cloud sandbox.** Upload `.ply` / `.las` / `.e57`; Gaussian initialiser runs server-side.
- **BIM sandbox.** Upload IFC or Revit, pick rooms, render synthetic camera array, converge on the photo-set path.
- **Pre-trained splat sandbox.** Direct upload of a 3DGS PLY plus a browser-side camera array tool that mirrors the Blender add-on.
- **Streaming chat.** SSE is wired; stream agent tokens through.
- **Per-Gaussian instance IDs** (Gaussian Grouping-style) so the splat itself carries object identity instead of a sidecar inventory file.

---

## Quick start

Requirements: Docker + Docker Compose, a HuggingFace token (demo splat and SAM 3 weights are gated), and a free Groq API key.

```bash
git clone https://github.com/lohithburra01/sceneagent.git
cd sceneagent

cp .env.example .env
# edit .env:
#   HF_TOKEN=hf_...
#   LLM_API_KEY=gsk_...                      # from console.groq.com
#   LLM_BASE_URL=https://api.groq.com/openai/v1
#   LLM_MODEL=qwen/qwen3-32b

./scripts/download_scene.sh 0003_839989

docker compose up --build
```

Visit `http://localhost:3000`. Hover an object in the inventory sidebar -- its bounding box highlights in 3D. Open the chat to talk to the concierge.

---

## Kubernetes

Runs on minikube with plain YAML -- no Helm.

```bash
cp k8s/99-secrets.example.yaml k8s/99-secrets.yaml
$EDITOR k8s/99-secrets.yaml   # paste LLM_API_KEY and HF_TOKEN
kubectl apply -f k8s/99-secrets.yaml
./k8s/apply.sh
```

---

## Credits

- **Dataset:** [InteriorGS](https://huggingface.co/datasets/spatialverse/InteriorGS) -- pre-trained indoor Gaussian splats.
- **Splat rasterizer:** [`gsplat`](https://github.com/nerfstudio-project/gsplat) -- CUDA 3DGS renderer.
- **Web viewer:** [@mkkellogg/gaussian-splats-3d](https://github.com/mkkellogg/GaussianSplats3D).
- **2D segmentation:** [SAM 3](https://github.com/facebookresearch/sam3) -- Meta Superintelligence Labs, 2026.
- **CLIP weights:** [OpenAI CLIP](https://github.com/openai/CLIP) ViT-B/32 via [open-clip](https://github.com/mlfoundations/open_clip).
- **Camera-array authoring:** Olli Huttunen's Camera Array Tool Blender add-on, extended with a SceneAgent JSON exporter.
- **Agent framework:** [LangGraph](https://langchain-ai.github.io/langgraph/).
- **Tool protocol:** Anthropic's [Model Context Protocol](https://modelcontextprotocol.io/).
- **Agent LLM:** [Qwen 3 32B](https://huggingface.co/Qwen/Qwen3-32B) on [Groq](https://groq.com/) -- pluggable to any OpenAI-compatible endpoint.
- **Vector search:** [pgvector](https://github.com/pgvector/pgvector).

---

## License

MIT
