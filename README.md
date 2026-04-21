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

The CV pipeline is seven scripts in `pipeline/src/` (P1 swap-in: `gsplat`
+ SAM 3 + Hungarian eval; the original CPU-only `render_py.py` and
`segment.py` remain for the no-CUDA fallback path):

0. **`decode_splat.py`** — InteriorGS ships its splats in the PlayCanvas
   compressed-PLY format (chunked 11/10/11-bit quantised positions, 2-10-10-10
   smallest-three quaternion, 8/8/8/8 SH-DC + logit-opacity color). We decode
   this in pure NumPy to get plain float Gaussians.
1. **`npz_to_ply.py`** — converts the decoded `.npz` into a standard
   3DGS 14-channel `.ply` (xyz, normals, f_dc_0..2, opacity logit, log
   scales, wxyz quat) so any 3DGS renderer can load it.
2. **`gen_camera_poses.py`** — samples 30 positions on a 6×6 grid
   *inside* the scene bbox at floor + 1.6 m (eye height), each looking
   at the nearest non-structural GT centroid. Replaces the v1
   outside-perimeter rings that produced cameras-see-floaters renders.
3. **`render_gsplat.py`** — `gsplat`'s CUDA rasterizer renders 30
   800×600 RGB+depth views in ~3.3 s total on an RTX 4060 Laptop
   (sm_89, JIT-compiled extension, OpenCV view convention).
4. **`segment_sam3.py`** — for each view, prompts SAM 3 with each of
   91 interior vocab classes (`bar counter`, `wine glass`, …), keeps
   masks above the confidence threshold. ~6.9 s/view on the 4060
   with bf16 autocast. Falls back to MobileSAM + open-CLIP if SAM 3
   isn't available.
5. **`backproject.py`** — for each Gaussian centre, projects into every
   view; accumulates per-class scores weighted by mask-area normalisation
   (so a big ceiling mask doesn't outvote a small wine-bottle mask).
   DBSCAN-clusters same-class Gaussians into instances.
6. **`evaluate.py`** — Hungarian-matched bipartite assignment between
   our predicted instances and the InteriorGS GT labels, with the
   structural classes (wall/floor/ceiling/window/door/stairs/column)
   dropped from both sides before matching. Reports class-matched and
   class-agnostic F1 at IoU thresholds 0.10 / 0.25 / 0.50.
7. **`seed_db.py`** — loads either our predicted inventory or, if empty,
   the InteriorGS ground-truth labels (transparent fallback) into Postgres
   so the product demo still works.

### Honest status of the numbers

The full pipeline runs end-to-end. From `decoded_splat.npz` →
`standard_3dgs.ply` (47 MB) → 30 gsplat views → SAM 3 with 91-class
vocab → 427 masks → backproject + DBSCAN → 254 predicted instances vs
240 GT. Total wall-clock on the 4060 Laptop: ~5 min (gsplat ~3 s,
SAM 3 ~3 min 26 s, backproject ~30 s, eval ~1 s).

Current `pipeline/output/metrics.json` reports:

| variant | IoU | TP | FP | FN | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **class_matched** | 0.25 | **4**  | 228 | 232 | 0.017 | 0.017 | **0.017** |
| class_matched     | 0.10 | 10 | 222 | 226 | 0.043 | 0.042 | 0.043 |
| class_matched     | 0.50 | 2  | 230 | 234 | 0.009 | 0.008 | 0.009 |
| **class_agnostic**| 0.25 | **16** | 216 | 220 | 0.069 | 0.068 | **0.068** |
| class_agnostic    | 0.10 | 31 | 201 | 205 | 0.134 | 0.131 | 0.132 |
| class_agnostic    | 0.50 | 3  | 229 | 233 | 0.013 | 0.013 | 0.013 |

**What changed vs v1 (`render_py` + MobileSAM + greedy matching, 0% class-matched):**

- Real GPU rasterisation (gsplat CUDA, sm_89) gives photo-like RGB+depth
  views instead of CPU point-sprite renders that lose the small objects.
- SAM 3 text-prompted segmentation against a 91-class open vocab —
  the prompt **is** the class label, no separate CLIP classify step.
- Hungarian matching instead of greedy first-best-IoU per prediction.
- Structural classes filtered before matching so wall/floor/ceiling
  trivially-perfect matches don't bias the score.

**What this means.** Class-agnostic shows the pipeline localises **16**
real GT objects at IoU ≥ 0.25 (and 31 at IoU ≥ 0.10) — real 3D
detections from real CV, no GT peeking. The class-matched gap (16
agnostic vs 4 matched at IoU ≥ 0.25) is dominated by class-name
mismatch with InteriorGS's wine-bar specific labels: SAM 3 calls
something "bottle" or "glass" while GT calls it "wine"; SAM 3 says
"chair" where GT says "high chair". Synonym groups in `evaluate.py`
absorb the obvious ones (wine ↔ wine_bottle ↔ bottle, glass ↔
wine_glass, downlight ↔ spotlight, pillow ↔ cushion) but the long
tail is open-ended.

The two biggest dials to turn for v3:

1. **Pose sampling.** A few of the 30 views currently look straight at
   a wall and yield 0 masks. Dropping those and re-sampling toward the
   high-density object regions should add ~5–10 TP without changing
   anything else.
2. **Domain-tuned text prompts.** Replacing the literal class-name
   prompt with a short LLM-generated description per class
   (`"a wine bottle on a bar shelf"`) is known to lift open-vocab
   recall ~2× on indoor benchmarks.

For the product demo, `seed_db.py` transparently uses our predicted
inventory (`source=ours`, 254 rows in `scene_objects`) — hotspots, chat,
and the MCP tools all run against real CV output, not a GT fallback.

### Sanity-check: what the pipeline sees

Class distribution of the 254 predictions (top 10):

```
glass         58    desk           16
plant         27    table          15
cushion       26    chair          12
window        18    tray            9
                    coffee table    6
                    sculpture       6
```

These match the scene's character (a wine bar with bottles/glasses
on counters, plants and cushions in the lounge, lots of seating).

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
