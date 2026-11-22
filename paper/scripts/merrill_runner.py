#!/usr/bin/env python3
"""
Run MERRILL PathMinimize from a prescribed 19-image initial path.

Initial-path builders (all on the MERRILL node set, end images m0 and -m0):
  covariant   : R(s) m0(R(s)^T r), 4-NN IDW, cosine schedule
  smoothaxis  : pointwise rotation by pi*theta(s) about a smooth axis field a(r) perpendicular to m0(r)
                (tangent field with its single zero inside the empty cap of m0), same cosine schedule
  tec         : take images from an existing TecPlot path (continuation runs)

Usage:
  merrill_runner.py <merrill_bin> <mesh.pat> <start_mag.dat> <workdir> covariant <ax,ay,az> [max_evals]
  merrill_runner.py <merrill_bin> <mesh.pat> <start_mag.dat> <workdir> smoothaxis - [max_evals]
  merrill_runner.py <merrill_bin> <mesh.pat> <start_mag.dat> <workdir> tec <path.tec> [max_evals]
  merrill_runner.py <merrill_bin> <mesh.pat> <start_mag.dat> <workdir> covariant_p1 <ax,ay,az> <max_evals> <any_tec_on_same_mesh>
"""
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from merrill_diag import read_tec  # noqa: E402
from path_fixes import largest_empty_cap  # noqa: E402

KBT_300 = 1.380649e-23 * 300.0
N_IMG = 19


def rodrigues(axis, ang):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def theta(k):
    """Cosine-clustered rotation angle; total angle from env ANGLE_DEG (default 180)."""
    import os
    total = math.radians(float(os.environ.get("ANGLE_DEG", "180")))
    s = k / (N_IMG - 1)
    return 0.5 * total * (1.0 - math.cos(math.pi * s))


class P1Interpolator:
    """Barycentric (P1) interpolation on the tetrahedral mesh; points outside the grain
    use the clamped barycentric weights of the best nearby tetrahedron."""

    def __init__(self, coords, tets, k=24):
        self.tets = tets
        P = coords[tets]
        self.T0 = P[:, 0, :]
        D = (P[:, 1:, :] - P[:, :1, :]).transpose(0, 2, 1)  # columns = edges
        self.Dinv = np.linalg.inv(D)
        self.tree = cKDTree(P.mean(axis=1))
        self.k = k

    def __call__(self, pts, values):
        _, cand = self.tree.query(pts, k=self.k)  # (n,k)
        rel = pts[:, None, :] - self.T0[cand]  # (n,k,3)
        lam123 = np.einsum("nkij,nkj->nki", self.Dinv[cand], rel)
        lam = np.concatenate([1 - lam123.sum(-1, keepdims=True), lam123], axis=-1)  # (n,k,4)
        best = np.argmax(lam.min(-1), axis=1)
        idx = np.arange(len(pts))
        L = np.clip(lam[idx, best], 0, None)
        L /= L.sum(axis=1, keepdims=True)
        nodes = self.tets[cand[idx, best]]  # (n,4)
        return np.einsum("na,nak->nk", L, values[nodes])


def covariant_path(coords, m0, axis, interp=None):
    """interp=None -> 4-NN inverse-distance weighting (as in run_merrill_benchmark.py);
    otherwise a callable(points, values) such as P1Interpolator."""
    c = coords - coords.mean(axis=0, keepdims=True)
    centre = coords.mean(axis=0, keepdims=True)
    tree = cKDTree(c)
    path = [m0]
    for k in range(1, N_IMG - 1):
        R = rodrigues(axis, theta(k))
        src = (R.T @ c.T).T
        if interp is None:
            d, idx = tree.query(src, k=4)
            w = 1.0 / (d + 1e-12)
            w /= w.sum(axis=1, keepdims=True)
            ms = np.einsum("ij,ijk->ik", w, m0[idx])
        else:
            ms = interp(src + centre, m0)
        mr = (R @ ms.T).T
        path.append(mr / np.linalg.norm(mr, axis=1, keepdims=True))
    path.append(-m0)
    return path


def smoothaxis_path(m0):
    p, cap = largest_empty_cap(m0)
    c = np.cross(p, [1.0, 0, 0] if abs(p[0]) < 0.9 else [0, 1.0, 0])
    c /= np.linalg.norm(c)
    a = (1 - m0 @ p)[:, None] * c + (m0 @ c)[:, None] * (p - m0)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    path = [m0]
    for k in range(1, N_IMG - 1):
        th = theta(k)
        m = m0 * math.cos(th) + np.cross(a, m0) * math.sin(th)
        path.append(m / np.linalg.norm(m, axis=1, keepdims=True))
    path.append(-m0)
    return path, {"cap_centre": p.round(3).tolist(), "cap_radius_deg": round(cap, 1)}


def main():
    mbin, mesh, start, wd, mode, arg = sys.argv[1:7]
    max_evals = int(sys.argv[7]) if len(sys.argv) > 7 else 1000
    wd = Path(wd)
    (wd / "path").mkdir(parents=True, exist_ok=True)
    data = np.loadtxt(start)
    coords, m0 = data[:, :3], data[:, 3:6] / np.linalg.norm(data[:, 3:6], axis=1, keepdims=True)
    end_dat = os.environ.get("END_DAT")
    if end_dat:
        e = np.loadtxt(end_dat)
        assert np.allclose(e[:, :3], coords, atol=1e-6), "END_DAT nodes differ"
        m_end = e[:, 3:6] / np.linalg.norm(e[:, 3:6], axis=1, keepdims=True)
    else:
        m_end = -m0
    info = {"mode": mode, "arg": arg, "max_evals": max_evals, "end": end_dat or "-m0"}
    if mode in ("covariant", "covariant_p1"):
        interp = None
        if mode == "covariant_p1":
            mesh_tec = Path(__file__).resolve().parent / "_mesh_topology.tec"
            _, tets, _ = read_tec(sys.argv[8] if len(sys.argv) > 8 else mesh_tec)
            interp = P1Interpolator(coords, tets)
        path = covariant_path(coords, m0, [float(v) for v in arg.split(",")], interp)
        info["total_angle_deg"] = float(os.environ.get("ANGLE_DEG", "180"))
        path[-1] = m_end
    elif mode == "smoothaxis":
        path, extra = smoothaxis_path(m0)
        info.update(extra)
    elif mode == "tec":
        tc, _, path = read_tec(arg)
        assert np.allclose(tc, coords, atol=1e-6), "tec nodes differ from start_mag nodes"
    elif mode in ("staged", "default"):
        path = [m0, m_end]
    else:
        raise SystemExit(f"unknown mode {mode}")
    for i, m in enumerate(path, 1):
        np.savetxt(wd / "path" / f"img_{i:02d}.dat", np.hstack([coords, m]), fmt="%.10e")
    load = "\n".join(f"ReadMagnetization: filename=path/img_{i:02d}.dat\nMagnetizationToPath: {i}" for i in range(1, N_IMG + 1))
    head = f"""Set: MaxMeshNumber = 1
ReadMesh: mesh=1, filename={Path(mesh).name}
Set: MaxEnergyEvaluations = 10000
Set: MaxPathEvaluations = {max_evals}
Material: Type=Magnetite, TempC=20
""" + "".join(l + "\n" for l in os.environ.get("EXTRA_MATERIAL", "").split(";") if l)
    tail = """WriteTecPlotPath: filename=finalpath.tec
PathStructureEnergies: filename=energies.txt
End
"""
    if mode in ("staged", "default"):
        ends = """Set: PathN = 2
ReadMagnetization: filename=path/img_01.dat
MagnetizationToPath: 1
ReadMagnetization: filename=path/img_02.dat
MagnetizationToPath: 2
"""
        if mode == "default":
            body = ends + f"""RefinePathTo: {N_IMG}
ConjugateGradient
Set: ExchangeCalculator = 1
MakeInitialPath
WriteTecPlotPath: filename=initpath.tec
PathLogfile: filename=neb
PathMinimize
"""
        else:
            body = ends + """RefinePathTo: 5
ConjugateGradient
Set: ExchangeCalculator = 1
MakeInitialPath
WriteTecPlotPath: filename=initpath.tec
PathLogfile: filename=neb5
PathMinimize
RefinePathTo: 9
RenewPath
PathLogfile: filename=neb9
PathMinimize
RefinePathTo: 19
RenewPath
PathLogfile: filename=neb
PathMinimize
"""
    else:
        body = f"""Set: PathN = {N_IMG}
{load}
RenewPath
ConjugateGradient
Set: ExchangeCalculator = 1
WriteTecPlotPath: filename=initpath.tec
PathLogfile: filename=neb
PathMinimize
"""
    (wd / "run.mscr").write_text(head + body + tail)


    import shutil
    if not (wd / Path(mesh).name).exists():
        shutil.copy(mesh, wd / Path(mesh).name)
    t0 = time.time()
    with open(wd / "run.log", "w") as lf:
        subprocess.run([str(Path(mbin).resolve()), "run.mscr"], cwd=wd, stdout=lf, stderr=subprocess.STDOUT, check=True)
    info["wall_s"] = round(time.time() - t0, 1)
    e = np.array([float(l.split()[-1]) for l in open(wd / "energies.txt") if l.strip()])
    prof = (e - min(e[0], e[-1])) / KBT_300
    info["barrier_kBT"] = round(float(prof.max()), 2)
    info["saddle_image"] = int(prof.argmax()) + 1
    info["profile_kBT"] = [round(float(v), 1) for v in prof]
    enlog = wd / "neb.enlog"
    info["path_evals_by_stage"] = {f.stem: sum(1 for l in open(f) if l.strip()) for f in sorted(wd.glob("neb*.enlog"))}
    (wd / "summary.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k != "profile_kBT"}))
    print("profile:", info["profile_kBT"])


if __name__ == "__main__":
    main()
