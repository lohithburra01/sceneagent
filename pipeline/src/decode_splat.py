"""Decode the PlayCanvas compressed-PLY Gaussian-splat format used by InteriorGS.

The file header has:
  element chunk N              — per-chunk quantization ranges (256 splats/chunk)
    min_x/y/z, max_x/y/z, min_scale_x/y/z, max_scale_x/y/z, min_r/g/b, max_r/g/b  (18 float32)
  element vertex M             — packed_position, packed_rotation, packed_scale, packed_color (4 uint32)
  element sh M                 — 45 uchar (SH coefficients, unused by us)

Bit layouts (PlayCanvas engine gsplat-compressed-data.js):
  packed_position : x:11 | y:10 | z:11   (unsigned unorm; lerp between chunk min/max)
  packed_scale    : sx:11 | sy:10 | sz:11 (unsigned unorm; log-scale; apply exp)
  packed_rotation : largest:2 | a:10 | b:10 | c:10  (smallest-3 quaternion, norm √2)
  packed_color    : r:8 | g:8 | b:8 | a:8 (rgb lerp between chunk min/max in SH-DC space;
                                           a is inverse-log-sigmoid opacity)

Output is a NumPy npz:
  centers   (N, 3) float32
  colors    (N, 3) uint8     — sRGB 0..255 approximation of f_dc projected to linear
  opacity   (N,)   float32   — sigmoid(opacity_logit)
  scale     (N, 3) float32   — linear (exp of stored log-scale)
  rotation  (N, 4) float32   — quaternion (w,x,y,z)

Usage:
  python -m pipeline.src.decode_splat data/scene/demo/3dgs_compressed.ply pipeline/output/decoded_splat.npz
"""
from __future__ import annotations

import argparse
import math
import struct
import sys
from pathlib import Path

import numpy as np

SH_C0 = 0.28209479177387814


def _parse_header(data: bytes) -> tuple[int, int, int, int]:
    """Return (header_end_offset, chunk_count, vertex_count, sh_count)."""
    header_end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:header_end].decode("ascii", errors="replace")
    chunk_n = vertex_n = sh_n = 0
    for line in header.splitlines():
        if line.startswith("element chunk "):
            chunk_n = int(line.split()[-1])
        elif line.startswith("element vertex "):
            vertex_n = int(line.split()[-1])
        elif line.startswith("element sh "):
            sh_n = int(line.split()[-1])
    return header_end, chunk_n, vertex_n, sh_n


def _unorm(v: np.ndarray, bits: int) -> np.ndarray:
    mask = (1 << bits) - 1
    return (v & mask).astype(np.float32) / mask


def decode(ply_path: Path) -> dict[str, np.ndarray]:
    data = ply_path.read_bytes()
    header_end, n_chunks, n_verts, n_sh = _parse_header(data)
    if n_sh and n_sh != n_verts:
        print(f"warn: sh count ({n_sh}) != vertex count ({n_verts})", file=sys.stderr)

    offset = header_end
    # chunk element: 18 float32 per chunk
    chunk_bytes = n_chunks * 18 * 4
    chunks = np.frombuffer(data, dtype=np.float32, count=n_chunks * 18, offset=offset)
    chunks = chunks.reshape(n_chunks, 18)
    offset += chunk_bytes

    # vertex element: 4 uint32 per vertex
    vert_bytes = n_verts * 4 * 4
    verts = np.frombuffer(data, dtype=np.uint32, count=n_verts * 4, offset=offset)
    verts = verts.reshape(n_verts, 4)
    offset += vert_bytes
    # sh element skipped — we don't need directional SH for 2D rendering of the mean colour.

    # Per-splat chunk index
    ci = (np.arange(n_verts) // 256).astype(np.int64)
    # Clamp for the tail partial chunk
    ci = np.clip(ci, 0, n_chunks - 1)

    pp = verts[:, 0]
    pr = verts[:, 1]
    ps = verts[:, 2]
    pc = verts[:, 3]

    # Position: 11|10|11
    px = _unorm(pp >> 21, 11)
    py = _unorm(pp >> 11, 10)
    pz = _unorm(pp, 11)
    min_p = chunks[ci, 0:3]
    max_p = chunks[ci, 3:6]
    centers = (min_p + (max_p - min_p) * np.stack([px, py, pz], axis=1)).astype(np.float32)

    # Scale: 11|10|11 — stored as log
    sx = _unorm(ps >> 21, 11)
    sy = _unorm(ps >> 11, 10)
    sz = _unorm(ps, 11)
    min_s = chunks[ci, 6:9]
    max_s = chunks[ci, 9:12]
    log_scale = (min_s + (max_s - min_s) * np.stack([sx, sy, sz], axis=1)).astype(np.float32)
    scale = np.exp(log_scale)

    # Rotation: smallest-three quaternion
    norm = math.sqrt(2.0)
    a = (_unorm(pr >> 20, 10) - 0.5) * norm
    b = (_unorm(pr >> 10, 10) - 0.5) * norm
    c = (_unorm(pr, 10) - 0.5) * norm
    m = np.sqrt(np.maximum(0.0, 1.0 - a * a - b * b - c * c))
    sel = (pr >> 30).astype(np.int64)
    quat = np.empty((n_verts, 4), dtype=np.float32)  # (w,x,y,z)
    # sel=0 → (a,b,c,m) means (x,y,z,w) per playcanvas
    # sel=1 → (m,b,c,a)  — largest is X
    # sel=2 → (b,m,c,a)  — largest is Y
    # sel=3 → (b,c,m,a)  — largest is Z
    xyzw = np.empty((n_verts, 4), dtype=np.float32)
    idx = sel == 0
    xyzw[idx] = np.stack([a[idx], b[idx], c[idx], m[idx]], axis=1)
    idx = sel == 1
    xyzw[idx] = np.stack([m[idx], b[idx], c[idx], a[idx]], axis=1)
    idx = sel == 2
    xyzw[idx] = np.stack([b[idx], m[idx], c[idx], a[idx]], axis=1)
    idx = sel == 3
    xyzw[idx] = np.stack([b[idx], c[idx], m[idx], a[idx]], axis=1)
    # Reorder to (w,x,y,z) for consistency with 3DGS convention
    quat[:, 0] = xyzw[:, 3]
    quat[:, 1] = xyzw[:, 0]
    quat[:, 2] = xyzw[:, 1]
    quat[:, 3] = xyzw[:, 2]

    # Color: 8|8|8|8
    r = _unorm(pc >> 24, 8)
    g = _unorm(pc >> 16, 8)
    bb = _unorm(pc >> 8, 8)
    aa = _unorm(pc, 8)
    min_c = chunks[ci, 12:15]
    max_c = chunks[ci, 15:18]
    f_dc = (min_c + (max_c - min_c) * np.stack([r, g, bb], axis=1)).astype(np.float32)
    # f_dc is in SH-DC space. Convert to 0..1 RGB by multiplying by SH_C0 + 0.5 (DC term contribution).
    rgb01 = np.clip(f_dc * SH_C0 + 0.5, 0.0, 1.0)
    colors = (rgb01 * 255.0).astype(np.uint8)

    # Opacity logit → probability
    aa_clamped = np.clip(aa, 1e-6, 1.0 - 1e-6)
    opacity_logit = -np.log(1.0 / aa_clamped - 1.0)
    opacity = (1.0 / (1.0 + np.exp(-opacity_logit))).astype(np.float32)

    return {
        "centers": centers,
        "colors": colors,
        "opacity": opacity,
        "scale": scale.astype(np.float32),
        "rotation": quat,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ply", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()

    out = decode(args.ply)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    n = out["centers"].shape[0]
    print(f"decoded {n} splats -> {args.out}")
    print(f"  centers range: {out['centers'].min(0).tolist()} .. {out['centers'].max(0).tolist()}")
    print(f"  colors mean: {out['colors'].mean(0).tolist()}")
    print(f"  opacity range: {out['opacity'].min():.3f} .. {out['opacity'].max():.3f}  mean: {out['opacity'].mean():.3f}")
    print(f"  scale (linear) range: {out['scale'].min():.4f} .. {out['scale'].max():.4f}  mean: {out['scale'].mean():.4f}")


if __name__ == "__main__":
    main()
