# P1 — Segmentation-Aware Demo (4-hour budget)

Branch: `feat/p1-segmentation-aware-demo`  (off `3fb5289`)

## Purpose

Turn the SceneAgent demo from a Sketchfab-style splat viewer into a
segmentation-aware 3D listing:

1. Lift the segmentation pipeline from 0% class-matched F1 to a
   real, honest, non-zero number by fixing the three root causes
   (render fidelity, vocab mismatch, greedy matching).
2. Surface segmentation in the UI — predicted objects become
   clickable, class-labelled, bbox-highlighted things, not a
   ring of emojis.
3. Replace the current WASD-up-down camera controls with
   Blender-style fly controls.

No use of GT labels at runtime. The vocabulary used for
classification is an open interior vocabulary (not derived from
`labels.json`). The same pipeline would run on a user-supplied
splat that has no GT at all.

## Scope in

- **Pipeline.** Convert `decoded_splat.npz` → standard 3DGS `.ply`;
  render 30 photorealistic views via `gsplat` from poses sampled
  inside the scene's occupancy volume looking at interior walls;
  segment each view with SAM 3 in open-vocabulary mode against a
  ~200-class interior vocab with prompt ensembling; backproject
  masks to Gaussians; cluster per class with DBSCAN; evaluate
  against `labels.json` with Hungarian matching (not greedy) and
  a structural-class filter.
- **Frontend.** Custom Blender-style fly controls on the splat
  viewer (W/S forward-back along view, A/D strafe, Q/E world-up
  / world-down, mouse-drag look), with `stopPropagation` on all
  text inputs. A right-docked sidebar (~340 px) listing predicted
  objects grouped by class, each row showing class, confidence,
  and any matched lister note; clicking flies the camera and
  highlights the object's 3D bounding box. Chat hidden behind a
  small bottom-right button that slides a sheet up; closed by
  default. Title chip top-left, controls hint chip bottom-centre,
  honest metrics badge at sidebar footer. Remove the circular
  emoji hotspot ring entirely.
- **Infra.** WSL2 Ubuntu for ML pipeline (venv with CUDA PyTorch,
  gsplat, SAM 3). Docker for API + web unchanged.

## Scope out (4h reality)

- SAGA / Gaussian Grouping / per-splat identity highlighting
  (stretch for session 2).
- SAM 3D Objects (it's image→mesh, not scene segmentation).
- User-records-video input pipeline.
- Any visual redesign round-trip; we ship a single coherent look
  (dark, minimal, one accent colour, Inter typography).
- Admin scrubber polish (separate concern).

## Architecture

### Pipeline data flow

```
InteriorGS 3dgs_compressed.ply
  → decode_splat.py → decoded_splat.npz           (already exists)
  → npz_to_standard_ply.py → standard_3dgs.ply    (NEW)
  → gen_camera_poses.py (rewritten)               (MODIFIED)
       poses from inside occupancy volume,
       looking at nearest interior wall,
       avoids floater viewpoints
  → render_gsplat.py → pipeline/output/views/*.png (NEW, replaces
       the Python point-sprite rasterizer for CLIP-input quality)
  → segment_sam3.py → pipeline/output/masks/*.json (NEW)
       SAM 3 automatic masks; open-vocab prompt-ensemble
       classification against a ~200-class interior vocab
  → backproject.py                                 (MODIFIED: reads
       new view/depth outputs)
  → evaluate.py                                    (MODIFIED:
       Hungarian matching, drop wall/floor/ceiling)
  → pipeline/output/object_inventory.json          (used by API)
```

The current `render_py.py` and `segment.py` are preserved but
unused for the new flow. We do not delete them in this session.

### Frontend module changes

- `web/src/components/SplatViewer.tsx` — disable mkkellogg's
  built-in keyboard controls, attach a custom fly-controller.
  Camera basis derived from the viewer's THREE.Camera.
- `web/src/components/HotspotMarkers.tsx` — renamed in spirit to
  `ObjectOverlay.tsx`; no persistent ring. Renders thin 3D bbox
  outlines for the active/hovered object only.
- `web/src/components/ChatOverlay.tsx` — chat panel collapsed
  into a small round icon button bottom-right; opens a slide-up
  sheet. `stopPropagation` on all key events from inputs.
- **NEW** `web/src/components/InventorySidebar.tsx` — right-docked
  340 px panel. Groups objects from `/api/hotspots/{slug}` by
  class, each row shows class name, count, confidence, and
  linked lister notes. Click → flyTo centroid + set
  `activeObjectId` for bbox highlight.
- `web/src/stores/viewer.ts` — add `activeObjectId`,
  `hoveredObjectId`, and a `setActiveObject` action.
- `web/src/app/listing/[slug]/page.tsx` — restructure layout:
  splat viewer fills screen minus sidebar; emoji ring gone;
  controls-hint chip added.

### Open vocabulary

Hardcoded in `segment_sam3.py`. Seeded from the LVIS-subset
interior vocab (~200 classes). Does **not** read `labels.json`
or any GT file at inference time. Includes common commercial
interior nouns (bar counter, stool, wine glass, etc.). Prompt
ensemble: `["a photo of a {c}", "a {c} in a room", "a close-up
photo of a {c}"]`.

## Success criteria

Hard acceptance:

1. End-to-end pipeline runs from the committed `decoded_splat.npz`
   to `metrics.json` on the branch, on the 4060, without manual
   intervention.
2. `class_matched@IoU>=0.25 F1 > 0` (any non-zero is a pass; stretch
   target: ≥10%).
3. Buyer page at `/listing/demo` renders with:
   - No emoji ring anywhere on screen.
   - Blender controls demonstrably working (W/S forward-back, Q/E
     up-down, A/D strafe, mouse-drag look; typing in chat does not
     orbit camera).
   - Right sidebar populated from live hotspots API, click flies +
     highlights bbox. Chat sheet opens via the icon button and
     dismisses via close button or outside-tap.
   - Metrics badge visible showing the real F1.
4. Spec and plan committed on branch, followed by implementation
   commits in small readable increments.

Soft:

- UI feels "not Claude-coded": dark, one accent, thin dividers,
  Inter, no rounded-2xl plastic, generous whitespace.

## Risks and mitigations

- **`gsplat` install on fresh WSL.** Mitigation: use the pip wheel
  with matching CUDA; fall back to `gaussian-splatting` repo's own
  rasterizer if wheel fails.
- **SAM 3 inference time on 8 GB 4060.** Mitigation: use the base
  checkpoint, not large; cap mask count per view.
- **Standard 3DGS `.ply` format mismatch.** Mitigation: write the
  converter against the exact 14-channel spec
  (xyz, nx ny nz, f_dc_0..2, f_rest_0..44, opacity, scale_0..2,
  rot_0..3); verify by loading into a known viewer before running
  gsplat.
- **WSL2 first-time setup burn.** Mitigation: session 1 plan
  assumes WSL2 ready; if not, fall back to a Python-native-on-
  Windows path for SAM 3 inference only (CPU tolerable for 30
  views) and skip gsplat — sparser but finishes.
- **Time blow-out.** Hard stop: if by 2h mark the pipeline is not
  producing a non-zero class-matched F1, accept current numbers
  and pivot all remaining time to UI; ship honestly.

## Setup (WSL2)

```
# one-time
wsl --install -d Ubuntu-22.04
# inside WSL
sudo apt update && sudo apt install -y python3.11-venv build-essential
python3 -m venv ~/venvs/sa && source ~/venvs/sa/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision
pip install gsplat open_clip_torch opencv-python Pillow numpy scipy
# SAM 3
git clone https://github.com/facebookresearch/sam3 && cd sam3
pip install -e .
# verify
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Deliverables

- `docs/superpowers/specs/2026-04-20-p1-segmentation-aware-demo-design.md`
  (this file).
- `docs/superpowers/plans/2026-04-20-p1-segmentation-aware-demo-plan.md`
  (next step, via writing-plans).
- Branch `feat/p1-segmentation-aware-demo` with small, reviewable
  commits in order of the plan.
- Updated `README.md` pipeline section with new metrics once the
  run completes.
