#!/usr/bin/env python3
"""
Symmetry-decomposed barrier calculation for MERRILL (v2.2.1 syntax).

Protocol (fixed in advance, no tuning per grain):
  1. orbit   : g.m0 for g in G x {1,T}: G = the elements of O_h that map the mesh onto itself (detected
               automatically; SUBGROUP=proper keeps only proper rotations), T = time reversal m -> -m.
               Improper elements act with det R (m axial). P1 interpolation on the mesh; de-duplicated
               (rms < 5 deg); identify the node equivalent to -m0.
  2. classes : edge classes = orbits of the stabiliser H = Stab(m0) on the orbit states; one
               representative per class, excluding m0 and -m0.
  3. relax   : MERRILL Minimize of every representative (one MERRILL run).
  4. hops    : MERRILL default NEB (RefinePathTo 19 + MakeInitialPath + PathMinimize) m0 -> each representative.
  5. network : edges g.(m0 -> rep_c) for all g, weight = hop barrier of class c; minimax path m0 -> -m0.
Wall time of every step is recorded (single process; run pinned to one core for benchmarking).

Usage: merrill_symdecomp.py <merrill_bin> <mesh.pat> <m0.dat> <workdir> [max_evals]
"""
import heapq
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from merrill_diag import read_tec, ang_deg, tet_geometry, node_weights  # noqa: E402
from merrill_runner import P1Interpolator  # noqa: E402

RUNNER = str(Path(__file__).resolve().parent / "merrill_runner.py")


def rotations():
    """Candidate elements: the 48 signed permutation matrices (O_h, the point group of cubic anisotropy
    in the crystal frame); improper ones act on the axial vector m with det(R)."""
    out = []
    for perm in itertools.permutations(range(3)):
        for sg in itertools.product([1, -1], repeat=3):
            R = np.zeros((3, 3))
            for i, j in enumerate(perm):
                R[i, j] = sg[i]
            if os.environ.get("SUBGROUP", "full") == "proper" and np.linalg.det(R) < 0:
                continue  # SUBGROUP=proper: the 24 proper rotations only
            out.append(R)  # full point group incl. improper elements (m is an axial vector)
    return out


def main():
    mbin, mesh, m0file, wd = sys.argv[1:5]
    max_evals = sys.argv[5] if len(sys.argv) > 5 else "1000"
    mbin = str(Path(mbin).resolve())
    wd = Path(wd)
    wd.mkdir(parents=True, exist_ok=True)
    shutil.copy(mesh, wd / Path(mesh).name)
    timing = {}
    t0 = time.time()
    # mesh topology: one MERRILL write of m0 gives a .tec with connectivity
    head = ["Set: MaxMeshNumber = 1", f"ReadMesh: mesh=1, filename={Path(mesh).name}",
            "Set: MaxEnergyEvaluations = 20000", "Material: Type=Magnetite, TempC=20"] + [l for l in os.environ.get("EXTRA_MATERIAL", "").split(";") if l] + ["ConjugateGradient"]
    shutil.copy(m0file, wd / "m0.dat")
    (wd / "topo.mscr").write_text("\n".join(head + ["ReadMagnetization: filename=m0.dat", "WriteMagnetization: filename=m0copy", "End"]) + "\n")
    subprocess.run([mbin, "topo.mscr"], cwd=wd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True)
    c, t, _ = read_tec(wd / "m0copy_mult.tec")
    a = np.loadtxt(wd / "m0.dat")
    m0 = a[:, 3:6] / np.linalg.norm(a[:, 3:6], axis=1, keepdims=True)
    vol, G = tet_geometry(c, t)
    w = node_weights(c, t, vol)
    d = lambda x, y: float(np.sqrt(np.sum(w * ang_deg(x, y) ** 2)))
    P = P1Interpolator(c, t)
    cen = c.mean(0)
    # symmetry group of the grain: elements of O_h that map the node set onto itself (automatic detection),
    # times time reversal T: m -> -m (exact in zero field). Group element = (R, t), acting as t det(R) R m(R^-1 r).
    from scipy.spatial import cKDTree
    tree = cKDTree(c - cen)
    hmin = float(np.min(np.linalg.norm(c[t[:, 1]] - c[t[:, 0]], axis=1)))
    shape_group = [R for R in rotations() if tree.query((c - cen) @ R.T)[0].max() < 1e-3 * hmin]
    Rs = [(R, tr) for R in shape_group for tr in (1, -1)]
    imgs = []
    for R, tr in Rs:
        mr = tr * np.linalg.det(R) * (R @ P((c - cen) @ R + cen, m0).T).T
        imgs.append(mr / np.linalg.norm(mr, axis=1, keepdims=True))
    nodes, node_of = [], []
    for mi in imgs:
        k = next((j for j, s in enumerate(nodes) if d(mi, s) < 5.0), None)
        if k is None:
            nodes.append(mi)
            k = len(nodes) - 1
        node_of.append(k)
    n_m0 = node_of[[np.allclose(R, np.eye(3)) and tr == 1 for R, tr in Rs].index(True)]
    dneg = [d(s, -m0) for s in nodes]
    n_neg = int(np.argmin(dneg))
    timing["orbit_s"] = round(time.time() - t0, 2)
    dist = [d(s, m0) for s in nodes]
    # exact edge classes: orbits of the stabiliser of m0 acting on the orbit states
    stab = [gi for gi in range(len(Rs)) if node_of[gi] == n_m0]
    rep_rot = {k: node_of.index(k) for k in range(len(nodes))}
    def compose(i, j):
        P_, t_ = Rs[i][0] @ Rs[j][0], Rs[i][1] * Rs[j][1]
        return next(q for q, (Q, tq) in enumerate(Rs) if np.allclose(Q, P_) and tq == t_)
    classes = {}
    seen = set()
    for k in range(len(nodes)):
        if k in (n_m0, n_neg) or k in seen:
            continue
        members = sorted({node_of[compose(g, rep_rot[k])] for g in stab})
        seen.update(members)
        classes[min(members)] = members
    if os.environ.get("ONLY_ANGLE"):  # restrict to the class(es) within 5 deg of this rms angle
        ang = float(os.environ["ONLY_ANGLE"])
        classes = {cl: mem for cl, mem in classes.items() if abs(dist[cl] - ang) < 5.0}
    reps = {cl: cl for cl in classes}

    info = {"subgroup": os.environ.get("SUBGROUP", "full"), "n_shape_group": len(shape_group), "n_group_with_T": len(Rs), "n_orbit_states": len(nodes), "m0_node": n_m0, "minus_m0_node": n_neg,
            "minus_m0_match_rms_deg": round(dneg[n_neg], 2),
            "classes": {f"class{cl}": {"rep": k, "members": classes[cl], "rms_to_m0_deg": round(dist[k], 1), "stabiliser_size": len(stab)} for cl, k in reps.items()}}
    # relax representatives
    t1 = time.time()
    lines = list(head)
    for cl, k in reps.items():
        np.savetxt(wd / f"rep{k}.dat", np.hstack([c, nodes[k]]), fmt="%.10e")
        lines += [f"ReadMagnetization: filename=rep{k}.dat", "Minimize", f"WriteMagnetization: filename=rep{k}_relaxed"]
    (wd / "relax.mscr").write_text("\n".join(lines + ["End"]) + "\n")
    with open(wd / "relax.log", "w") as lf:
        subprocess.run([mbin, "relax.mscr"], cwd=wd, stdout=lf, stderr=subprocess.STDOUT, check=True)
    timing["relax_s"] = round(time.time() - t1, 2)
    # hops
    hop = {}
    t2 = time.time()
    for cl, k in reps.items():
        env = dict(os.environ, END_DAT=str(wd / f"rep{k}_relaxed.dat"))
        th = time.time()
        with open(wd / f"hop{k}.out", "w") as f:
            subprocess.run([sys.executable, "-I", RUNNER, mbin, str(wd / Path(mesh).name), str(wd / "m0.dat"),
                            str(wd / f"hop{k}"), "default", "-", max_evals], env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
        s = json.load(open(wd / f"hop{k}" / "summary.json"))
        txt = (wd / f"hop{k}" / "run.log").read_text()
        import re
        n = re.findall(r"CalcPathAction Calls:\s*(\d+)", txt)
        hop[k] = {"class": f"class{cl}_{dist[k]:.0f}deg", "barrier_kBT": s["barrier_kBT"], "evals": int(n[-1]),
                  "wall_s": round(time.time() - th, 2), "profile_kBT": s["profile_kBT"]}
    timing["hops_s"] = round(time.time() - t2, 2)
    # network minimax m0 -> -m0
    adj = {}
    for gi in range(len(Rs)):
        a_ = node_of[gi]
        for cl, k in reps.items():
            # rep k = h.m0 for some h; g.rep = (g h).m0
            hi = node_of.index(k)
            b_ = node_of[compose(gi, hi)]
            wgt = hop[k]["barrier_kBT"]
            adj.setdefault(a_, {})[b_] = min(wgt, adj.get(a_, {}).get(b_, 1e18))
            adj.setdefault(b_, {})[a_] = min(wgt, adj.get(b_, {}).get(a_, 1e18))
    best = {n_m0: 0.0}
    pq = [(0.0, n_m0)]
    prev = {}
    while pq:
        cost, u = heapq.heappop(pq)
        if cost > best.get(u, 1e18):
            continue
        for v, wgt in adj.get(u, {}).items():
            nc = max(cost, wgt)
            if nc < best.get(v, 1e18):
                best[v] = nc
                prev[v] = u
                heapq.heappush(pq, (nc, v))
    path = [n_neg]
    while path[-1] != n_m0 and path[-1] in prev:
        path.append(prev[path[-1]])
    info["hops"] = hop
    info["minimax_barrier_m0_to_minus_m0_kBT"] = best.get(n_neg)
    info["minimax_path_nodes"] = path[::-1]
    info["lowest_escape_barrier_kBT"] = min((h["barrier_kBT"] for h in hop.values()), default=None)
    timing["total_s"] = round(time.time() - t0, 2)
    info["timing"] = timing
    info["total_path_evals"] = sum(h["evals"] for h in hop.values())
    (wd / "symdecomp.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k != "hops"}, indent=1))
    for k, h in hop.items():
        print(k, {kk: vv for kk, vv in h.items() if kk != "profile_kBT"})


if __name__ == "__main__":
    main()
