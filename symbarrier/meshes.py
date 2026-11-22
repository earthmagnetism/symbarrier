"""Idealised test meshes: BCC lattice clipped to the shape, Delaunay tetrahedralisation, sliver filter.
Units are micrometres, as MERRILL expects."""
import numpy as np
from scipy.spatial import Delaunay


def _tetrahedralise(pts, h):
    sim = Delaunay(pts).simplices
    v0, v1, v2, v3 = (pts[sim[:, i]] for i in range(4))
    vol = np.einsum("ij,ij->i", np.cross(v1 - v0, v2 - v0), v3 - v0) / 6.0
    keep = np.abs(vol) > 1e-6 * h ** 3
    sim, vol = sim[keep], vol[keep]
    neg = vol < 0
    sim[neg, 1], sim[neg, 2] = sim[neg, 2].copy(), sim[neg, 1].copy()
    used = np.unique(sim)
    remap = np.full(len(pts), -1)
    remap[used] = np.arange(len(used))
    return pts[used], remap[sim]


def _bcc(n):
    g = [np.arange(-k, k + 1) for k in n]
    sc = np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3)
    b = [np.arange(-k, k) + 0.5 for k in n]
    bc = np.stack(np.meshgrid(*b, indexing="ij"), -1).reshape(-1, 3)
    return np.vstack([sc, bc]).astype(float)


def truncated_octahedron(size_nm=100.0, n_sub=3):
    """|x|,|y|,|z| <= 2s and |x|+|y|+|z| <= 3s with 4s = size (square-face separation); h = size / (4 n_sub)."""
    s = size_nm * 1e-3 / 4.0
    h = s / n_sub
    p = _bcc((2 * n_sub,) * 3) * h
    tol = 1e-9 * s
    keep = (np.abs(p).max(1) <= 2 * s + tol) & (np.abs(p).sum(1) <= 3 * s + tol)
    return _tetrahedralise(p[keep], h)


def box(lx_nm, ly_nm, lz_nm, h_nm):
    """Rectangular prism centred at the origin; the half-lengths must be multiples of h."""
    half = np.array([lx_nm, ly_nm, lz_nm]) * 1e-3 / 2
    h = h_nm * 1e-3
    n = np.rint(half / h).astype(int)
    assert np.allclose(n * h, half), "half-lengths must be multiples of h"
    return _tetrahedralise(_bcc(n) * h, h)


def analytic_vortex(coords, axis, core_radius=0.02, handed=1):
    """Smooth single-vortex field with core along `axis` (unit), used as a seed and in tests."""
    u = np.asarray(axis, float) / np.linalg.norm(axis)
    r = coords - coords.mean(0)
    rho = r - (r @ u)[:, None] * u
    phi = handed * np.cross(u, rho)
    phi /= np.linalg.norm(phi, axis=1, keepdims=True) + 1e-30
    wgt = np.exp(-0.5 * (np.linalg.norm(rho, axis=1) / core_radius) ** 2)[:, None]
    m = wgt * u + np.sqrt(np.clip(1 - wgt ** 2, 0, None)) * phi
    return m / np.linalg.norm(m, axis=1, keepdims=True)
