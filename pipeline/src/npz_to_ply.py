"""Convert decoded_splat.npz to standard 3DGS .ply (14 channels + 0-order SH).

Minimal spec: xyz, nx/ny/nz (zeros), f_dc_0..2 (DC spherical-harmonic coefficient,
derived from linear RGB), opacity (logit), scale_0..2 (log), rot_0..3 (wxyz
quaternion). We skip f_rest_* (higher SH orders); gsplat and most other
renderers treat them as zero when absent.
"""
from __future__ import annotations

import numpy as np
from plyfile import PlyData, PlyElement

SH_C0 = 0.28209479177387814


def _rgb_to_dc(rgb: np.ndarray) -> np.ndarray:
    return (rgb.astype(np.float32) - 0.5) / SH_C0


def _pick(z, *names):
    for n in names:
        if n in z.files:
            return z[n]
    raise KeyError(f"none of {names} in npz (have: {z.files})")


def npz_to_standard_ply(npz_path: str, ply_path: str) -> None:
    z = np.load(npz_path)
    xyz = _pick(z, "centers", "means", "xyz").astype(np.float32)
    rgb_raw = _pick(z, "colors", "rgb", "color")
    # Accept either uint8 0..255 or float 0..1
    rgb = rgb_raw.astype(np.float32)
    if rgb_raw.dtype == np.uint8 or rgb.max() > 1.5:
        rgb = rgb / 255.0
    op = _pick(z, "opacities", "opacity").astype(np.float32).reshape(-1)
    scale = _pick(z, "scales", "scale").astype(np.float32)
    rot = _pick(z, "rotations", "rotation", "quats", "quat").astype(np.float32)

    n = xyz.shape[0]
    assert rgb.shape == (n, 3), rgb.shape
    assert scale.shape == (n, 3), scale.shape
    assert rot.shape == (n, 4), rot.shape

    dc = _rgb_to_dc(rgb)
    eps = 1e-6
    op_clip = np.clip(op, eps, 1 - eps)
    op_logit = np.log(op_clip / (1 - op_clip)).astype(np.float32)
    scale_log = np.log(np.maximum(scale, 1e-8)).astype(np.float32)

    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]
    arr = np.zeros(n, dtype=dtype)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = dc[:, 0], dc[:, 1], dc[:, 2]
    arr["opacity"] = op_logit
    arr["scale_0"], arr["scale_1"], arr["scale_2"] = scale_log[:, 0], scale_log[:, 1], scale_log[:, 2]
    arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"] = rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]

    el = PlyElement.describe(arr, "vertex")
    PlyData([el], text=False).write(ply_path)


if __name__ == "__main__":
    import sys
    npz_to_standard_ply(sys.argv[1], sys.argv[2])
    print(f"wrote {sys.argv[2]}")
