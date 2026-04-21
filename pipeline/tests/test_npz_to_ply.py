import numpy as np
from plyfile import PlyData

from pipeline.src.npz_to_ply import npz_to_standard_ply


def test_produces_valid_3dgs_ply(tmp_path):
    npz = tmp_path / "fake.npz"
    n = 64
    np.savez(
        npz,
        centers=np.random.randn(n, 3).astype(np.float32),
        colors=np.random.rand(n, 3).astype(np.float32),
        opacities=np.random.rand(n).astype(np.float32),
        scales=np.abs(np.random.randn(n, 3)).astype(np.float32) * 0.01,
        rotations=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
    )
    out = tmp_path / "out.ply"
    npz_to_standard_ply(str(npz), str(out))
    pd = PlyData.read(str(out))
    v = pd["vertex"]
    assert len(v) == n
    for f in (
        "x", "y", "z",
        "f_dc_0", "f_dc_1", "f_dc_2",
        "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ):
        assert f in v.data.dtype.names, f"missing field {f}"
