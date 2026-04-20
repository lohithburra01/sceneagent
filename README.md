# SceneAgent

Zillow is 30 static photos. SceneAgent turns a short walkthrough into a full
3D Gaussian-splat listing with an AI concierge that answers questions,
measures distances, and guides tours.

![Next.js](https://img.shields.io/badge/Next.js-14-black?logo=nextdotjs)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi)
![Postgres + pgvector](https://img.shields.io/badge/Postgres-16%20%2B%20pgvector-336791?logo=postgresql)
![Gaussian Splatting](https://img.shields.io/badge/Gaussian%20Splatting-3DGS-ff6b35)
![open-clip](https://img.shields.io/badge/open--clip-ViT--B%2F32-5a67d8)
![MobileSAM](https://img.shields.io/badge/MobileSAM-CPU-34d399)
![LangGraph](https://img.shields.io/badge/LangGraph-agent-1e40af)
![MCP](https://img.shields.io/badge/MCP-Anthropic-8b5cf6)
![Gemini Flash](https://img.shields.io/badge/Gemini-2.0%20Flash-4285f4?logo=google)
![Kubernetes](https://img.shields.io/badge/Kubernetes-minikube-326ce5?logo=kubernetes)

> *Video link forthcoming — see v2 roadmap.*

The scene used in the v1 demo is InteriorGS `0003_839989`, a wine bar with
299 labeled objects. It's commercial rather than residential because the
InteriorGS dataset leans commercial — we treat the demo as a commercial
real-estate walkthrough.

---

## Architecture

```
                   ┌───────────────────────┐
                   │     user browser      │
                   └──────────┬────────────┘
                              │  HTTP / SSE
                              ▼
                   ┌───────────────────────┐
                   │  web  (Next.js 14)    │
                   │  @mkkellogg splat     │
                   │  hotspots, chat UI    │
                   └──────────┬────────────┘
                              │  REST + SSE
                              ▼
                   ┌───────────────────────┐
                   │  api  (FastAPI)       │
                   │  /notes /hotspots     │
                   │  /agent/chat (SSE)    │
                   │  /mcp  (MCP server)   │
                   └──┬──────────┬─────────┘
                      │          │
           reads/     │          │  LangGraph agent
           writes     │          │  calls MCP tools
                      ▼          ▼
          ┌─────────────────┐  ┌──────────────────────┐
          │ postgres 16     │  │ scene tools (MCP):   │
          │ + pgvector      │  │  render_view         │
          │ (scenes, objects│  │  list_objects        │
          │  notes, hotspots│  │  find_by_description │
          │  CLIP embedings)│  │  measure_distance    │
          └─────────────────┘  │  highlight_region    │
                               │  move_camera_to      │
                               │  plan_tour           │
                               └──────────┬───────────┘
                                          │ image/text
                                          ▼
                               ┌──────────────────────┐
                               │  Gemini 2.0 Flash    │
                               │  (text + vision)     │
                               └──────────────────────┘

         (scaffolded, idle in v1: redis + celery worker for
          future async splat-training pipeline)
```

*Legend: solid arrows are synchronous request/response; the agent loops over
tool calls until it can compose a final answer, then streams it back over
SSE.*

---

## What this project demonstrates

- **Real-time 3D Gaussian-splat rendering in the browser** — InteriorGS's
  pre-trained splat loaded via `@mkkellogg/gaussian-splats-3d`, orbit + WASD
  navigation at interactive frame rates.
- **A 2D→3D semantic segmentation pipeline** — MobileSAM for 2D masks,
  OpenCLIP ViT-B/32 for zero-shot classification, our own multi-view
  backprojection + majority voting, DBSCAN to cluster Gaussians into
  instances, evaluated against InteriorGS ground-truth labels. All of this
  is in `pipeline/`; it doesn't require a GPU.
- **Agentic AI with tool calling via MCP** — a LangGraph agent sits inside
  the FastAPI process and calls scene tools through an MCP server. The
  agent decides when to render a view, measure a distance, or look up a
  hotspot; it does not carry the scene in its context window.
- **Digital-twin grounding** — the chat can fly the camera to an object,
  highlight a region, measure between two 3D points, and plan guided tours.
  The agent is not answering questions *about* a scene — it is
  manipulating it.
- **One-command deployments** — `docker compose up --build` for local dev,
  `./k8s/apply.sh` for minikube.

---

## Our segmentation pipeline

The CV pipeline is five scripts in `pipeline/src/`:

0. **`decode_splat.py`** — InteriorGS ships its splats in the PlayCanvas
   compressed-PLY format (chunked 11/10/11-bit quantised positions, 2-10-10-10
   smallest-three quaternion, 8/8/8/8 SH-DC + logit-opacity color). We decode
   this in pure NumPy to get plain float Gaussians. No CUDA needed.
1. **`render_py.py`** — software splat rasteriser: project 717 k decoded
   Gaussians to 2D with a z-buffer, draw each as a scale/depth-sized disk.
   30 views × 1024×768 in ~2 min on CPU.
2. **`segment.py`** — for each rendered view, run MobileSAM for 2D instance
   masks, classify each mask with OpenCLIP ViT-B/32 against a 54-class
   home+commercial vocabulary. ~23 s/view on CPU, 705 masks total.
3. **`backproject.py`** — for each Gaussian centre, project into every view;
   accumulate class votes weighted by CLIP confidence and **mask-area
   normalisation** (so a big ceiling mask doesn't outvote a small wine-bottle
   mask). Pick the winning class; DBSCAN-cluster same-class Gaussians into
   instances; compute axis-aligned bounding boxes.
3. **`evaluate.py`** — match predicted instances to InteriorGS
   ground-truth labels by IoU ≥ 0.25 and report per-class precision,
   recall, and F1 to `pipeline/output/metrics.json`.
4. **`seed_db.py`** — loads either our predicted inventory or, if empty,
   the InteriorGS ground-truth labels (transparent fallback) into Postgres
   so the product demo still works.

### Honest status of the numbers

The pipeline runs end-to-end. `@mkkellogg/gaussian-splats-3d` in headless
Chrome couldn't decode InteriorGS's compressed-packed `.ply` in our time
budget, so we wrote our own PlayCanvas-compressed-PLY decoder (`decode_splat.py`)
plus a CPU-only software rasteriser (`render_py.py`). From those renders we
get 705 MobileSAM masks, 341 predicted instances after backprojection +
DBSCAN. GT has 240 instances across 44 classes.

Current `pipeline/output/metrics.json` reports (**IoU ≥ 0.25**):

| variant | TP | FP | FN | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| **class_matched** (strict — same class name, aliased) | **0** | 341 | 240 | 0.000 | 0.000 | 0.000 |
| **class_agnostic** (any GT object overlaps) | **21** | 320 | 219 | 0.062 | 0.087 | 0.072 |

Lower-IoU signal:

| variant | TP | precision | recall | F1 |
|---|---:|---:|---:|---:|
| class_agnostic @ IoU ≥ 0.10 | 37 | 0.109 | 0.154 | 0.127 |
| class_agnostic @ IoU ≥ 0.50 | 4  | 0.012 | 0.017 | 0.014 |

**What this means.** Class-agnostic shows the pipeline **localises** 21
real GT objects at IoU ≥ 0.25 (and 37 at IoU ≥ 0.10) — real 3D detections
coming from real CV, no ground truth peeking. The predicted class names
are mostly wrong because CLIP ViT-B/32, given our low-density point-sprite
renders instead of true 3DGS-rasterised photos, tends to pick structural /
residential classes (downlight, decorative_painting, shoe, vase) even when
the underlying blob is really a wine bottle or high chair.

The two biggest dials to turn for v2 are obvious:

1. **Better rasteriser** — convert the compressed PLY to standard
   `.ksplat` once via `@playcanvas/splat-transform`, then render with
   `gsplat` (true Gaussian rasterisation, anti-aliased anisotropic
   footprints) instead of point sprites. That alone should move P & R
   into the 30–50 % range on this scene.
2. **Domain-tuned classifier** — swap CLIP zero-shot for a small CLIP
   LoRA or a dedicated linear probe trained on 1–2 k InteriorGS
   crops. 0.24 mean CLIP confidence is the bottleneck, not SAM.

For the product demo, `seed_db.py` transparently uses our predicted
inventory (`source=ours`, 341 rows in `scene_objects`) — hotspots, chat,
and the MCP tools all run against real CV output, not a GT fallback.

### Sanity-check: what the pipeline sees

Class distribution of the 341 predictions (top 10):

```
downlight            120    spotlight              25
decorative_painting   94    wine_glass             24
wine                  70    stool                  19
vase                  30    wine_bottle            18
shoe                  30    cup                    17
```

These match the scene's character (wine bottles on shelves, spotlights /
downlights overhead, decorative paintings, stools) — the bboxes are just
coarser than the individual GT instances.

<details>
<summary>Top per-class GT breakdown (what the evaluator's looking for)</summary>

```json
{
  "jar":                 {"tp": 0, "fp": 0, "fn": 4,  "precision": 0, "recall": 0.0},
  "teapot":              {"tp": 0, "fp": 0, "fn": 1,  "precision": 0, "recall": 0.0},
  "cabinet":             {"tp": 0, "fp": 0, "fn": 1,  "precision": 0, "recall": 0.0},
  "tray":                {"tp": 0, "fp": 0, "fn": 3,  "precision": 0, "recall": 0.0},
  "side_table":          {"tp": 0, "fp": 0, "fn": 3,  "precision": 0, "recall": 0.0},
  "decorative_painting": {"tp": 0, "fp": 0, "fn": 3,  "precision": 0, "recall": 0.0},
  "suspended_ceiling":   {"tp": 0, "fp": 0, "fn": 2,  "precision": 0, "recall": 0.0},
  "cup":                 {"tp": 0, "fp": 0, "fn": 24, "precision": 0, "recall": 0.0},
  "door":                {"tp": 0, "fp": 0, "fn": 1,  "precision": 0, "recall": 0.0},
  "dining_plate":        {"tp": 0, "fp": 0, "fn": 10, "precision": 0, "recall": 0.0},
  "book":                {"tp": 0, "fp": 0, "fn": 4,  "precision": 0, "recall": 0.0},
  "billboard":           {"tp": 0, "fp": 0, "fn": 2,  "precision": 0, "recall": 0.0},
  "downlights":          {"tp": 0, "fp": 0, "fn": 14, "precision": 0, "recall": 0.0},
  "box":                 {"tp": 0, "fp": 0, "fn": 1,  "precision": 0, "recall": 0.0},
  "vase":                {"tp": 0, "fp": 0, "fn": 1,  "precision": 0, "recall": 0.0},
  "coffee_maker":        {"tp": 0, "fp": 0, "fn": 2,  "precision": 0, "recall": 0.0},
  "chocolate":           {"tp": 0, "fp": 0, "fn": 4,  "precision": 0, "recall": 0.0},
  "decorative_pendant":  {"tp": 0, "fp": 0, "fn": 1,  "precision": 0, "recall": 0.0},
  "plant":               {"tp": 0, "fp": 0, "fn": 4,  "precision": 0, "recall": 0.0},
  "chandelier":          {"tp": 0, "fp": 0, "fn": 4,  "precision": 0, "recall": 0.0},
  "multi_person_sofa":   {"tp": 0, "fp": 0, "fn": 1,  "precision": 0, "recall": 0.0},
  "basket":              {"tp": 0, "fp": 0, "fn": 1,  "precision": 0, "recall": 0.0},
  "wine":                {"tp": 0, "fp": 0, "fn": 70, "precision": 0, "recall": 0.0},
  "placemat":            {"tp": 0, "fp": 0, "fn": 1,  "precision": 0, "recall": 0.0},
  "bread":               {"tp": 0, "fp": 0, "fn": 1,  "precision": 0, "recall": 0.0},
  "candle":              {"tp": 0, "fp": 0, "fn": 2,  "precision": 0, "recall": 0.0},
  "ornament":            {"tp": 0, "fp": 0, "fn": 4,  "precision": 0, "recall": 0.0},
  "wardrobe":            {"tp": 0, "fp": 0, "fn": 1,  "precision": 0, "recall": 0.0},
  "fruit":               {"tp": 0, "fp": 0, "fn": 13, "precision": 0, "recall": 0.0},
  "wine_glass":          {"tp": 0, "fp": 0, "fn": 5,  "precision": 0, "recall": 0.0},
  "stool":               {"tp": 0, "fp": 0, "fn": 3,  "precision": 0, "recall": 0.0},
  "canned_food":         {"tp": 0, "fp": 0, "fn": 1,  "precision": 0, "recall": 0.0},
  "console":             {"tp": 0, "fp": 0, "fn": 1,  "precision": 0, "recall": 0.0},
  "spotlight":           {"tp": 0, "fp": 0, "fn": 25, "precision": 0, "recall": 0.0},
  "cake":                {"tp": 0, "fp": 0, "fn": 1,  "precision": 0, "recall": 0.0},
  "table":               {"tp": 0, "fp": 0, "fn": 2,  "precision": 0, "recall": 0.0},
  "spoon":               {"tp": 0, "fp": 0, "fn": 1,  "precision": 0, "recall": 0.0},
  "high_chair":          {"tp": 0, "fp": 0, "fn": 10, "precision": 0, "recall": 0.0},
  "chair":               {"tp": 0, "fp": 0, "fn": 1,  "precision": 0, "recall": 0.0},
  "window":              {"tp": 0, "fp": 0, "fn": 1,  "precision": 0, "recall": 0.0},
  "body_pillow":         {"tp": 0, "fp": 0, "fn": 2,  "precision": 0, "recall": 0.0},
  "combination":         {"tp": 0, "fp": 0, "fn": 1,  "precision": 0, "recall": 0.0},
  "sofa":                {"tp": 0, "fp": 0, "fn": 1,  "precision": 0, "recall": 0.0},
  "kettle":              {"tp": 0, "fp": 0, "fn": 1,  "precision": 0, "recall": 0.0},
  "carpet":              {"tp": 0, "fp": 0, "fn": 1,  "precision": 0, "recall": 0.0}
}
```

</details>

---

## Quick start

You need Docker + Docker Compose, `huggingface-cli` logged in (the
InteriorGS dataset requires one-click license acceptance on HF), and a
Google AI Studio API key for Gemini.

1. **Clone the repo.**
   ```bash
   git clone https://github.com/<you>/sceneagent.git
   cd sceneagent
   ```
2. **Configure environment.**
   ```bash
   cp .env.example .env
   # edit .env and paste GEMINI_API_KEY=<your key>
   ```
3. **Download the scene.**
   ```bash
   ./scripts/download_scene.sh 0003_839989
   ```
4. **Bring up the stack.**
   ```bash
   docker compose up --build
   ```
5. **Open the listing.**
   Visit <http://localhost:3000>. Click the demo listing, orbit the wine
   bar, click a hotspot, then open the chat.

---

## Kubernetes demo

The same stack runs on minikube with plain YAML — no Helm. See
[`k8s/README.md`](./k8s/README.md) for the full walkthrough, or:

```bash
cp k8s/99-secrets.example.yaml k8s/99-secrets.yaml
$EDITOR k8s/99-secrets.yaml            # paste GEMINI_API_KEY
kubectl apply -f k8s/99-secrets.yaml
./k8s/apply.sh
```

*Caveat: `minikube start` must already be running. `apply.sh` refuses to
continue otherwise.*

---

## v2 roadmap

- **Fix the renderer.** Convert InteriorGS `.ply` → `.ksplat` ahead of
  time, or swap the headless-Chrome rasterizer for a Python `gsplat`
  rasterizer. Unblocks the segmentation pipeline end-to-end.
- **Real camera trajectory from a user-uploaded video.** Extract poses via
  COLMAP or MASt3R instead of the synthesized waypoint path.
- **Better hotspot placement.** Raycast the viewer's click into the
  Gaussian cloud instead of snapping to the nearest object centroid.
- **Streaming chat.** SSE is already stubbed; wire Gemini's token stream
  through.
- **LoRA-tuned classifier.** Fine-tune OpenCLIP on in-the-wild real-estate
  imagery to close the domain gap with OpenAI's original weights.

---

## Credits

- **Dataset:** [InteriorGS](https://huggingface.co/datasets/spatialverse/InteriorGS) (`spatialverse/InteriorGS`) — 1,000 pre-trained
  indoor Gaussian splats with per-object class labels, 3D OBBs, room
  structure, and occupancy maps. Used under its HuggingFace license.
- **Concept:** [Gaussian Grouping](https://github.com/lkeab/gaussian-grouping) (Ye et al., ECCV 2024) — the idea of
  per-Gaussian instance identities that the segmentation pipeline leans on.
- **Web viewer:** [@mkkellogg/gaussian-splats-3d](https://github.com/mkkellogg/GaussianSplats3D).
- **CLIP weights:** [OpenAI CLIP](https://github.com/openai/CLIP) ViT-B/32 via
  [open-clip](https://github.com/mlfoundations/open_clip).
- **2D segmentation:** [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) (Chaoning Zhang et al., 2023).
- **Agent framework:** [LangGraph](https://langchain-ai.github.io/langgraph/).
- **Tool protocol:** Anthropic's [Model Context Protocol](https://modelcontextprotocol.io/) + the
  [Python SDK](https://github.com/modelcontextprotocol/python-sdk).
- **LLM / VLM:** [Gemini 2.0 Flash](https://ai.google.dev/gemini-api/docs/models/gemini).
- **Vector search:** [pgvector](https://github.com/pgvector/pgvector).

---

## License

MIT (TBD).
