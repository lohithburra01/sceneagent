"""Software splat rasterizer — project decoded Gaussians to 2D and write PNGs.

Not a real 3DGS rasterizer (no anisotropic footprints, no alpha compositing of
overlapping Gaussians). What it does:

  1. Read decoded Gaussians from pipeline/output/decoded_splat.npz
  2. For each pose in pipeline/render_views/camera_poses.json, project centers
     to 2D, discard points behind the camera / outside the frustum, paint each
     as a screen-space disk with the decoded SH-DC colour. Z-buffer so closer
     splats win.
  3. Write pipeline/output/views/view_XXX.png + _intrinsics.json

The output is crude but recognisable enough for SAM to segment furniture-sized
objects, which is all the downstream pipeline needs. Runs on CPU, ~2-3 s/view
on this machine at 1024x768.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

NPZ = Path("pipeline/output/decoded_splat.npz")
POSES = Path("pipeline/render_views/camera_poses.json")
OUT_DIR = Path("pipeline/output/views")

WIDTH = 1024
HEIGHT = 768
FOV_V_DEG = 60.0
OPACITY_MIN = 0.10
MAX_POINTS = 717_050  # all splats — sparser renders give SAM nothing to segment


def pose_to_w2c(pos, target, up=(0, 0, 1)):
    pos = np.asarray(pos, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)
    fwd = target - pos
    fwd /= np.linalg.norm(fwd) + 1e-9
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right) + 1e-9
    up2 = np.cross(right, fwd)
    R = np.stack([right, up2, -fwd], axis=0)
    t = -R @ pos
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def render_one(centers, colors, opacities, scale, pose):
    w2c = pose_to_w2c(pose["position"], pose["lookAt"])
    homo = np.concatenate([centers, np.ones((centers.shape[0], 1), dtype=np.float32)], axis=1)
    cam = homo @ w2c.T
    z = cam[:, 2]
    in_front = z < -0.05  # camera looks down -z (per the R = [right,up,-fwd] convention)
    depth = -z  # positive depth for in-front points

    f = (HEIGHT / 2) / math.tan(math.radians(FOV_V_DEG) / 2)
    u = cam[:, 0] * f / depth + WIDTH / 2
    v = -cam[:, 1] * f / depth + HEIGHT / 2

    valid = in_front & (u >= -20) & (u < WIDTH + 20) & (v >= -20) & (v < HEIGHT + 20) & (depth < 40.0)
    u = u[valid]
    v = v[valid]
    depth = depth[valid]
    rgb = colors[valid]
    sc = scale[valid]

    # Sort far→near so near points overwrite far ones in the z-buffer.
    order = np.argsort(-depth)
    u = u[order]; v = v[order]; depth = depth[order]; rgb = rgb[order]; sc = sc[order]

    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    zbuf = np.full((HEIGHT, WIDTH), 1e9, dtype=np.float32)

    # Screen-space radius: project scale/depth to pixel count. Keep within [1, 6].
    px_scale = (sc * f / depth)
    radii = np.clip(np.rint(px_scale + 1.0).astype(np.int32), 1, 6)

    # Vectorised disk rasterisation using OpenCV circle per point is slow (~50k/s).
    # For speed: assign each point to its nearest pixel with z-buffer, then dilate.
    ui = np.clip(u.astype(np.int32), 0, WIDTH - 1)
    vi = np.clip(v.astype(np.int32), 0, HEIGHT - 1)

    # Pass 1: z-buffered splat with per-point disk (radius from scale/depth).
    # Vectorised on the outer index — OpenCV circle is still the bottleneck but
    # with 700k points it's bounded to ~15 s per view on CPU.
    for idx in range(u.shape[0]):
        px, py = ui[idx], vi[idx]
        d = depth[idx]
        r = int(radii[idx])
        if d < zbuf[py, px]:
            zbuf[py, px] = d
            cv2.circle(img, (px, py), r, (int(rgb[idx, 0]), int(rgb[idx, 1]), int(rgb[idx, 2])), -1)

    # Pass 2: small dilation to merge point-sprite edges — helps SAM find
    # connected regions. One iteration of 3x3 is enough with dense splats.
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)

    return img


def main():
    data = np.load(NPZ)
    centers = data["centers"]
    colors = data["colors"]
    opacity = data["opacity"]
    scale = data["scale"].max(axis=1)  # use largest axis as isotropic proxy

    keep = opacity > OPACITY_MIN
    centers = centers[keep]
    colors = colors[keep]
    opacity = opacity[keep]
    scale = scale[keep]

    if centers.shape[0] > MAX_POINTS:
        # Keep the highest-opacity subset
        idx = np.argsort(-opacity)[:MAX_POINTS]
        centers = centers[idx]; colors = colors[idx]; opacity = opacity[idx]; scale = scale[idx]

    poses = json.loads(POSES.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, pose in enumerate(poses):
        img = render_one(centers, colors, opacity, scale, pose)
        cv2.imwrite(str(OUT_DIR / f"view_{i:03d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"view {i+1:02d}/{len(poses)} written")

    (OUT_DIR / "_intrinsics.json").write_text(json.dumps({
        "width": WIDTH, "height": HEIGHT, "fov_vertical_deg": FOV_V_DEG, "poses": poses,
    }, indent=2))
    print(f"done: {len(poses)} views in {OUT_DIR}")


if __name__ == "__main__":
    main()
