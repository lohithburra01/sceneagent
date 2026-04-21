"""Convert a standard 3DGS .ply (14 channels, no faces) into a vanilla
mesh-PLY of colored vertices that Blender's built-in PLY importer will
accept.

Use this when KIRI's 3dgs-render addon isn't available — Blender will
import the result as a points-only mesh, which is perfectly adequate
for visually placing a camera array inside the scene's geometry.

Usage:
  python -m pipeline.src.ply_to_pointcloud \
      pipeline/output/standard_3dgs.ply \
      pipeline/output/scene_points.ply
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

SH_C0 = 0.28209479177387814


def convert(in_path: str, out_path: str, max_points: int = 400_000) -> None:
    src = PlyData.read(in_path)["vertex"]
    n = len(src)
    xyz = np.stack([src["x"], src["y"], src["z"]], axis=-1).astype(np.float32)
    # rgb from f_dc DC SH coefficient
    dc = np.stack([src["f_dc_0"], src["f_dc_1"], src["f_dc_2"]], axis=-1).astype(np.float32)
    rgb = np.clip(dc * SH_C0 + 0.5, 0, 1)
    rgb_u8 = (rgb * 255).astype(np.uint8)

    # alpha from logit opacity → keep dense (not too transparent) so Blender
    # actually renders the points
    op = 1.0 / (1.0 + np.exp(-src["opacity"].astype(np.float32)))
    alpha_u8 = (np.clip(op, 0.1, 1.0) * 255).astype(np.uint8)

    # Optionally subsample so Blender doesn't choke on 700k points
    if n > max_points:
        idx = np.random.RandomState(0).choice(n, max_points, replace=False)
        xyz = xyz[idx]
        rgb_u8 = rgb_u8[idx]
        alpha_u8 = alpha_u8[idx]
        n = max_points
        print(f"subsampled to {n} points (cap = {max_points})")

    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("alpha", "u1"),
    ]
    arr = np.zeros(n, dtype=dtype)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["red"], arr["green"], arr["blue"] = rgb_u8[:, 0], rgb_u8[:, 1], rgb_u8[:, 2]
    arr["alpha"] = alpha_u8

    el = PlyElement.describe(arr, "vertex")
    PlyData([el], text=False).write(out_path)
    print(f"wrote {n} colored points to {out_path}")


if __name__ == "__main__":
    in_path = sys.argv[1] if len(sys.argv) > 1 else "pipeline/output/standard_3dgs.ply"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "pipeline/output/scene_points.ply"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    convert(in_path, out_path)
