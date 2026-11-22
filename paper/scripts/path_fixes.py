"""
Helpers for initial paths.

1. select_c2_axis_*: choose the covariant rotation axis only among C2 axes of the
   cubic point group (<100>, <110>), which map both the cube-aligned grain and the
   magnetite crystal onto themselves, and pick the one whose R(pi) image of m0 is
   closest to -m0 (endpoint mismatch eps). The anisotropy proxy is only a tie-break.
   (Assumes the mesh/grid axes coincide with the crystal axes.)

2. geodesic_band_smooth_axis_torch: a fair pointwise (non-covariant) baseline for
   m0 -> -m0. Each spin rotates by pi*s about a smooth axis field a(r) perpendicular to
   m0(r), built from a tangent field on S^2 whose only zero sits in a cap that m0 never
   visits. Unlike the notebook's build_slerp_random_band (which reduces to
   cos(pi s) m0 + noise and is pure noise at s = 0.5), this band is smooth.
"""
import math

import numpy as np

C2_AXES = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, -1, 0),
           (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1)]


def _rodrigues(axis, ang):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def select_c2_axis_mesh(coords, m0, k=4, tol_deg=1.0):
    """For unstructured meshes (MERRILL). Returns (axis, eps_rms_deg, table)."""
    from scipy.spatial import cKDTree
    c = coords - coords.mean(axis=0, keepdims=True)
    tree = cKDTree(c)
    table = []
    for ax in C2_AXES:
        R = _rodrigues(ax, math.pi)
        d, idx = tree.query((R.T @ c.T).T, k=k)
        w = 1.0 / (d + 1e-12)
        w /= w.sum(axis=1, keepdims=True)
        mr = (R @ np.einsum("ij,ijk->ik", w, m0[idx]).T).T
        mr /= np.linalg.norm(mr, axis=1, keepdims=True)
        ang = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", mr, -m0), -1, 1)))
        # path-integrated cubic-anisotropy proxy (K1 < 0): tie-break only
        proxy = 0.0
        for s in np.linspace(0.1, 0.9, 9):
            mm = (_rodrigues(ax, math.pi * s) @ m0.T).T
            proxy -= float(np.mean(mm[:, 0]**2 * mm[:, 1]**2 + mm[:, 1]**2 * mm[:, 2]**2 + mm[:, 2]**2 * mm[:, 0]**2))
        table.append((ax, float(np.sqrt(np.mean(ang**2))), proxy))
    best_eps = min(t[1] for t in table)
    cands = [t for t in table if t[1] <= best_eps + tol_deg]
    ax, eps, _ = min(cands, key=lambda t: t[2])
    if eps > 5.0:
        print(f"WARNING: no C2 axis maps m0 to -m0 (best eps = {eps:.1f} deg); "
              "the covariant band will need a non-rotational final segment.")
    return list(ax), eps, table


def largest_empty_cap(m, n_dir=20000):
    """Centre p and angular radius (deg) of the largest spherical cap containing no spin of m (N,3 numpy)."""
    i = np.arange(n_dir) + 0.5
    phi, th = np.arccos(1 - 2 * i / n_dir), np.pi * (1 + 5 ** 0.5) * i
    d = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
    closest = (m @ d.T).max(0)
    j = int(np.argmin(closest))
    return d[j], float(np.degrees(np.arccos(np.clip(closest[j], -1, 1))))


def geodesic_band_smooth_axis_torch(m0, n_images, p=None, sigma=0.0, seed=0):
    """Fair pointwise baseline for m0 -> -m0 (FD grids, m: [Nx,Ny,Nz,3] torch tensor).

    Every spin is rotated by pi*s about a(r) = v(m0(r))/|v|, where
        v(m) = (1 - m.p) c + (c.m)(p - m),  c perpendicular to p,
    is a tangent field on S^2 whose only zero is at p (pulled-back constant field of the
    stereographic projection from p). If p lies in a cap that m0 never visits, a(r) is
    smooth everywhere, so the band is singularity-free and ends exactly at -m0.
    If m0 covers all of S^2, no such smooth pointwise scheme exists (hairy-ball theorem).
    """
    import torch
    dev, dt = m0.device, m0.dtype
    if p is None:
        p, cap = largest_empty_cap(m0.reshape(-1, 3).detach().cpu().numpy())
        print(f"empty cap: centre {np.round(p, 3)}, radius {cap:.1f} deg")
    p = np.asarray(p, float) / np.linalg.norm(p)
    c = np.cross(p, [1.0, 0, 0] if abs(p[0]) < 0.9 else [0, 1.0, 0])
    c /= np.linalg.norm(c)
    P = torch.tensor(p, device=dev, dtype=dt)
    C = torch.tensor(c, device=dev, dtype=dt)
    mp = (m0 * P).sum(-1, keepdim=True)
    mc = (m0 * C).sum(-1, keepdim=True)
    a = (1 - mp) * C + mc * (P - m0)
    a = a / a.norm(dim=-1, keepdim=True)
    g = torch.Generator(device=dev).manual_seed(seed)
    band = [m0.clone()]
    for k in range(1, n_images - 1):
        th = math.pi * k / (n_images - 1)
        m = m0 * math.cos(th) + torch.cross(a, m0, dim=-1) * math.sin(th)
        if sigma > 0:
            m = m + sigma * math.sin(th) * torch.randn(m.shape, generator=g, device=dev, dtype=dt)
        band.append(m / m.norm(dim=-1, keepdim=True))
    band.append(-m0.clone())
    return band
