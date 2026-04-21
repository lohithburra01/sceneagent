"""Render views from a standard 3DGS .ply using gsplat's CUDA rasterizer.

Inputs:
  pipeline/output/standard_3dgs.ply
  pipeline/output/camera_poses.json
Outputs:
  pipeline/output/views/view_{i:02d}.png         (rgb)
  pipeline/output/views/depth_{i:02d}.npy        (depth, metres)
  pipeline/output/views/_intrinsics.json         (compat file for
                                                  backproject.py)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData
from gsplat import rasterization

PLY = Path("pipeline/output/standard_3dgs.ply")
POSES = Path("pipeline/output/camera_poses.json")
OUT = Path("pipeline/output/views")
OUT.mkdir(parents=True, exist_ok=True)

SH_C0 = 0.28209479177387814


def load_splat(path: Path, device: str):
    v = PlyData.read(str(path))["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
    dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1).astype(np.float32)
    rgb = np.clip(dc * SH_C0 + 0.5, 0, 1)
    op = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float32)))
    scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1).astype(np.float32))
    quats = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1).astype(np.float32)
    quats = quats / np.maximum(np.linalg.norm(quats, axis=-1, keepdims=True), 1e-8)
    return (torch.from_numpy(xyz).to(device),
            torch.from_numpy(quats).to(device),
            torch.from_numpy(scales).to(device),
            torch.from_numpy(op).to(device),
            torch.from_numpy(rgb).to(device))


def view_matrix_opencv(position, look_at, up):
    """World-to-camera matrix in OpenCV convention (z forward, y down, x right)."""
    position = np.asarray(position, np.float32)
    look_at = np.asarray(look_at, np.float32)
    up = np.asarray(up, np.float32)
    forward = look_at - position
    forward = forward / max(float(np.linalg.norm(forward)), 1e-8)
    right = np.cross(forward, up)
    right = right / max(float(np.linalg.norm(right)), 1e-8)
    down = np.cross(forward, right)
    R = np.stack([right, down, forward], axis=0)  # rows = camera axes in world
    t = -R @ position
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def intrinsics(fov_y_deg, width, height):
    fy = 0.5 * height / np.tan(np.deg2rad(fov_y_deg) / 2)
    fx = fy
    cx, cy = width / 2, height / 2
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


def main():
    device = "cuda"
    print("loading splat...")
    means, quats, scales, opacities, colors = load_splat(PLY, device)
    print(f"  {means.shape[0]} gaussians")
    poses = json.loads(POSES.read_text())

    # compat file for backproject.py
    intr_meta = {
        "width": poses[0]["width"],
        "height": poses[0]["height"],
        "fov_vertical_deg": poses[0]["fov_y_deg"],
        "poses": [
            {"position": p["position"], "lookAt": p["look_at"]}
            for p in poses
        ],
    }
    (OUT / "_intrinsics.json").write_text(json.dumps(intr_meta, indent=2))

    total = time.time()
    for p in poses:
        i = p["index"]
        W, H = p["width"], p["height"]
        V = view_matrix_opencv(p["position"], p["look_at"], p["up"])
        K = intrinsics(p["fov_y_deg"], W, H)

        viewmat = torch.from_numpy(V).to(device).unsqueeze(0)
        Kt = torch.from_numpy(K).to(device).unsqueeze(0)

        t0 = time.time()
        rgb, alpha, meta = rasterization(
            means=means, quats=quats, scales=scales,
            opacities=opacities, colors=colors,
            viewmats=viewmat, Ks=Kt,
            width=W, height=H, render_mode="RGB+D",
        )
        torch.cuda.synchronize()
        img = (rgb[0, ..., :3].clamp(0, 1) * 255).byte().cpu().numpy()
        depth = rgb[0, ..., 3].cpu().numpy()
        Image.fromarray(img).save(OUT / f"view_{i:02d}.png")
        np.save(OUT / f"depth_{i:02d}.npy", depth.astype(np.float32))
        print(f"[{i:02d}] {W}x{H} in {(time.time()-t0)*1000:.0f} ms  "
              f"rgb[min,max]=[{img.min()},{img.max()}]  depth[med]={np.median(depth[depth>0]):.2f}m")

    print(f"total: {time.time()-total:.1f}s for {len(poses)} views")


if __name__ == "__main__":
    main()
