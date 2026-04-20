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
