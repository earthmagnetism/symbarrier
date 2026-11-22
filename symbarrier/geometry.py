"""Tetrahedral-mesh utilities: PATRAN / TecPlot I/O, P1 interpolation, volume weights, rms angle distance."""
import re
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def read_patran(path):
    """Read a PATRAN neutral file (MERRILL mesh). Returns coords (N,3) in the file's units and tets (E,4), 0-based.
    Node ids need not be contiguous or start at 1."""
    L = Path(path).read_text().split("\n")
    ids, X, T = [], [], []
    i = 0
    while i < len(L):
        line = L[i]
        if len(line) < 2 or not line[:2].strip().isdigit():
            i += 1
            continue
        k = int(line[:2])
        if k == 1:
            ids.append(int(line[2:10]))
            X.append([float(L[i + 1][j:j + 16]) for j in (0, 16, 32)])
            i += 3
        elif k == 2:
            T.append([int(v) for v in L[i + 2].split()[:4]])
            i += 3
        elif k in (25, 26):
            i += 2
        elif k == 99:
            break
        else:
            i += 1
    index = {v: j for j, v in enumerate(ids)}
    return np.array(X), np.array([[index[v] for v in t] for t in T])


def write_patran(path, pts, tets, title="mesh"):
    """Write nodes (N,3) and tets (E,4, 0-based) as a PATRAN neutral file readable by MERRILL."""
    with open(path, "w") as f:
        f.write(f"{25:2d}{0:8d}{0:8d}{1:8d}{0:8d}{0:8d}{0:8d}{0:8d}{0:8d}\n{title}\n")
        f.write(f"{26:2d}{0:8d}{0:8d}{1:8d}{len(pts):8d}{len(tets):8d}{1:8d}{1:8d}{0:8d}\n01/01/26    00:00:00         5.0\n")
        for i, (x, y, z) in enumerate(pts, start=1):
            f.write(f"{1:2d}{i:8d}{0:8d}{2:8d}{0:8d}{0:8d}{0:8d}{0:8d}{0:8d}\n{x:16.9e}{y:16.9e}{z:16.9e}\n0G       6       0       0  000000\n")
        for i, (a, b, c, d) in enumerate(np.asarray(tets) + 1, start=1):
            f.write(f"{2:2d}{i:8d}{5:8d}{2:8d}{0:8d}{0:8d}{0:8d}{0:8d}{0:8d}\n")
            f.write(f"{4:8d}{0:8d}{1:8d}{0:8d}{0.0:16.9e}{0.0:16.9e}{0.0:16.9e}\n{a:8d}{b:8d}{c:8d}{d:8d}\n")
        f.write(f"{99:2d}{0:8d}{0:8d}{1:8d}{0:8d}{0:8d}{0:8d}{0:8d}{0:8d}\n")


def read_tecplot(path):
    """Parse a MERRILL FEBLOCK TecPlot file -> coords (N,3), tets (E,4), list of unit m (N,3), one per zone."""
    lines = Path(path).read_text().splitlines()
    zone_idx = [i for i, l in enumerate(lines) if l.strip().startswith("ZONE")]
    head = lines[zone_idx[0]]
    N = int(re.search(r"\bN\s*=\s*(\d+)", head).group(1))
    E = int(re.search(r"\bE\s*=\s*(\d+)", head).group(1))
    mags, coords, tets = [], None, None
    for z, start in enumerate(zone_idx):
        end = zone_idx[z + 1] if z + 1 < len(zone_idx) else len(lines)
        body = lines[start + 2:end]
        if z == 0:
            conn = body[-E:]
            nums = np.array(" ".join(body[:-E]).split(), dtype=float)
            coords = nums[:3 * N].reshape(3, N).T
            m = nums[3 * N:6 * N].reshape(3, N).T
            tets = np.array(" ".join(conn).split(), dtype=int).reshape(E, 4) - 1
        else:
            m = np.array(" ".join(body).split(), dtype=float).reshape(3, N).T
        mags.append(m / np.linalg.norm(m, axis=1, keepdims=True))
    return coords, tets, mags


def tet_volumes(coords, tets):
    P = coords[tets]
    return np.abs(np.linalg.det(P[:, 1:, :] - P[:, :1, :])) / 6.0


def node_weights(coords, tets):
    """Lumped (P1) volume weight of every node, normalised to 1."""
    vol = tet_volumes(coords, tets)
    w = np.zeros(len(coords))
    np.add.at(w, tets.ravel(), np.repeat(vol / 4.0, 4))
    return w / w.sum()


def rms_angle_deg(a, b, w):
    """Volume-weighted rms angle (degrees) between two unit fields."""
    ang = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", a, b), -1, 1)))
    return float(np.sqrt(np.sum(w * ang ** 2)))


class P1Interpolator:
    """Barycentric (P1) interpolation on a tetrahedral mesh; query points outside the grain use the clamped
    barycentric weights of the best nearby tetrahedron."""

    def __init__(self, coords, tets, k=24):
        self.tets = tets
        P = coords[tets]
        self.T0 = P[:, 0, :]
        self.Dinv = np.linalg.inv((P[:, 1:, :] - P[:, :1, :]).transpose(0, 2, 1))
        self.tree = cKDTree(P.mean(axis=1))
        self.k = k

    def __call__(self, pts, values):
        _, cand = self.tree.query(pts, k=self.k)
        rel = pts[:, None, :] - self.T0[cand]
        lam123 = np.einsum("nkij,nkj->nki", self.Dinv[cand], rel)
        lam = np.concatenate([1 - lam123.sum(-1, keepdims=True), lam123], axis=-1)
        best = np.argmax(lam.min(-1), axis=1)
        idx = np.arange(len(pts))
        L = np.clip(lam[idx, best], 0, None)
        L /= L.sum(axis=1, keepdims=True)
        return np.einsum("na,nak->nk", L, values[self.tets[cand[idx, best]]])
