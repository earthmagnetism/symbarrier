#!/usr/bin/env python3
"""
Diagnostics for SO(3) covariant GNEB paths from MERRILL TecPlot output.

For each mesh:
  1. Endpoint mismatch eps: compare R(pi) m0(R(pi)^T r) with -m0 (the
     endpoint actually used), for the rotation axis used in the run and for
     the C2 axes of the cubic group (O_h), which also map the grain to itself.
  2. Helicity H(s) along initial / converged paths (P1 tetrahedral gradients).
  3. Max edge angle per image, and the geodesic gap between consecutive images.
  4. Distance between the SO(3) and staged saddle images.

Usage: python3 merrill_diag.py <run_dir> <tag> <init.tec> <final.tec> <staged.tec> <axis_x,axis_y,axis_z>
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def read_tec(path):
    """Parse MERRILL FEBLOCK TecPlot path file -> coords (N,3), tets (E,4), list of m (N,3)."""
    lines = Path(path).read_text().splitlines()
    zone_idx = [i for i, l in enumerate(lines) if l.strip().startswith("ZONE")]
    import re
    head = lines[zone_idx[0]]
    N = int(re.search(r"\bN\s*=\s*(\d+)", head).group(1))
    E = int(re.search(r"\bE\s*=\s*(\d+)", head).group(1))
    mags = []
    coords = tets = None
    for z, start in enumerate(zone_idx):
        end = zone_idx[z + 1] if z + 1 < len(zone_idx) else len(lines)
        body = lines[start + 2:end]
        if z == 0:
            # 6 nodal vars + 1 cell var as whitespace stream, then E lines of connectivity
            conn = body[-E:]
            nums = np.array(" ".join(body[:-E]).split(), dtype=float)
            assert nums.size == 6 * N + E, (nums.size, 6 * N + E)
            xyz = nums[:3 * N].reshape(3, N).T
            m = nums[3 * N:6 * N].reshape(3, N).T
            coords = xyz
            tets = np.array(" ".join(conn).split(), dtype=int).reshape(E, 4) - 1
        else:
            nums = np.array(" ".join(body).split(), dtype=float)
            assert nums.size == 3 * N, (nums.size, 3 * N)
            m = nums.reshape(3, N).T
        mags.append(m / np.linalg.norm(m, axis=1, keepdims=True))
    return coords, tets, mags


def rodrigues(axis, ang):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def covariant_image(coords, m0, R):
    """Same construction as run_merrill_benchmark.py (centroid, 4-NN IDW)."""
    c = coords - coords.mean(axis=0, keepdims=True)
    tree = cKDTree(c)
    d, idx = tree.query((R.T @ c.T).T, k=4)
    w = 1.0 / (d + 1e-12)
    w /= w.sum(axis=1, keepdims=True)
    ms = np.einsum("ij,ijk->ik", w, m0[idx])
    mr = (R @ ms.T).T
    # how far outside the grain the pulled-back points fall (nearest-node distance)
    return mr / np.linalg.norm(mr, axis=1, keepdims=True), d[:, 0]


def tet_geometry(coords, tets):
    P = coords[tets]  # E,4,3
    D = P[:, 1:, :] - P[:, :1, :]  # E,3,3 rows = edges
    vol = np.abs(np.linalg.det(D)) / 6.0
    Dinv = np.linalg.inv(D)  # grad of barycentric coords 1..3 are columns of Dinv
    G = np.zeros((len(tets), 4, 3))
    G[:, 1:, :] = np.transpose(Dinv, (0, 2, 1))
    G[:, 0, :] = -G[:, 1:, :].sum(axis=1)
    return vol, G


def helicity(m, tets, vol, G):
    """H = sum_tet V * mean(m) . curl(m), with P1 interpolation (curl constant per tet)."""
    mt = m[tets]  # E,4,3
    J = np.einsum("eak,eal->ekl", mt, G)  # J[k,l] = d m_k / d x_l
    curl = np.stack([J[:, 2, 1] - J[:, 1, 2], J[:, 0, 2] - J[:, 2, 0], J[:, 1, 0] - J[:, 0, 1]], axis=1)
    return float(np.sum(vol * np.einsum("ek,ek->e", mt.mean(axis=1), curl)))


def edges_of(tets):
    e = np.concatenate([tets[:, [i, j]] for i in range(4) for j in range(i + 1, 4)])
    e.sort(axis=1)
    return np.unique(e, axis=0)


def ang_deg(a, b):
    return np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", a, b), -1, 1)))


def node_weights(coords, tets, vol):
    w = np.zeros(len(coords))
    np.add.at(w, tets.ravel(), np.repeat(vol / 4.0, 4))
    return w / w.sum()


def mismatch(mrot, mtarget, w):
    a = ang_deg(mrot, mtarget)
    return {"rms_deg": float(np.sqrt(np.sum(w * a ** 2))),
            "frac_vol_gt90deg": float(np.sum(w[a > 90])),
            "max_deg": float(a.max())}


def main():
    run_dir, tag, f_init, f_final, f_staged, axis = sys.argv[1:7]
    axis = [float(v) for v in axis.split(",")]
    run_dir = Path(run_dir)
    coords, tets, init = read_tec(run_dir / f_init)
    _, _, final = read_tec(run_dir / f_final)
    _, _, staged = read_tec(run_dir / f_staged)
    vol, G = tet_geometry(coords, tets)
    w = node_weights(coords, tets, vol)
    E = edges_of(tets)
    m0, m1 = init[0], init[-1]
    out = {"tag": tag, "n_nodes": len(coords), "n_tets": len(tets),
           "endpoint_is_minus_m0_deg": float(ang_deg(m1, -m0).max())}

    # (1) endpoint mismatch for used axis and all O_h C2 axes (<100>, <110>)
    c2 = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1)]
    R_used = rodrigues(axis, math.pi)
    mr, dnear = covariant_image(coords, m0, R_used)
    h = np.median(np.linalg.norm(coords[E[:, 0]] - coords[E[:, 1]], axis=1))
    out["used_axis"] = {"axis": axis, **mismatch(mr, -m0, w),
                        "frac_nodes_pulled_outside(>1 edge)": float(np.mean(dnear > h))}
    rows = []
    for ax in c2:
        mr2, dn2 = covariant_image(coords, m0, rodrigues(ax, math.pi))
        rows.append({"axis": ax, **mismatch(mr2, -m0, w), "frac_nodes_pulled_outside(>1 edge)": float(np.mean(dn2 > h))})
    rows.sort(key=lambda r: r["rms_deg"])
    out["C2_axes"] = rows

    # (2-4) path diagnostics
    H0 = helicity(m0, tets, vol, G)
    def path_stats(path):
        Hs = [helicity(m, tets, vol, G) / H0 for m in path]
        maxedge = [float(ang_deg(m[E[:, 0]], m[E[:, 1]]).max()) for m in path]
        gaps = [float(np.sqrt(np.sum(w * ang_deg(path[i], path[i + 1]) ** 2))) for i in range(len(path) - 1)]
        return {"H_over_H0": [round(x, 3) for x in Hs], "max_edge_deg": [round(x, 1) for x in maxedge],
                "rms_gap_deg": [round(x, 1) for x in gaps]}
    out["H0"] = H0
    out["init_so3"] = path_stats(init)
    out["final_so3"] = path_stats(final)
    out["final_staged"] = path_stats(staged)
    json.dump(out, sys.stdout, indent=1)
    return out


if __name__ == "__main__":
    main()
