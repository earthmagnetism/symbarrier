#!/usr/bin/env python3
"""
Symmetry-reduced barrier network over several LEM families (MERRILL v2.2.1).

  1. group    : G = elements of O_h mapping the mesh onto itself (automatic), times time reversal T
                (exact in zero field). Element (R, t) acts as t det(R) R m(R^-1 r). For an irregular grain
                G = {e} and only T remains.
  2. nodes    : orbits of every seed LEM under G x {1,T}, de-duplicated (rms < 5 deg). Seeds come from any
                search (random starts, covariant band, earlier runs).
  3. edges    : short-edge graph: every node joined to its K nearest nodes (rms distance), symmetrised.
  4. classes  : orbits of G x {1,T} on unordered node pairs, from the node permutation table.
  5. NEB      : one MERRILL default NEB per edge class between relaxed endpoints.
  6. split    : an interior image of a band that lies > SPLIT_KBT below both neighbours and is a local
                minimum of the profile is relaxed; if it is a new state it is reported (and, with
                ITERATE=1, added as a new seed family for a second pass).
  7. network  : minimax barrier m0 -> -m0 (upper bound by the mountain-pass property), escape barrier of
                every family, Arrhenius rate matrix k_ab = f0 exp(-(E_s - E_a)/kT), f0 = 1e9 s^-1, T = 300 K.

Usage: merrill_network.py <merrill_bin> <mesh.pat> <workdir> <m0.dat> [seed2.dat ...]
Env:   K (default 4), K_CROSS (default 2, nearest nodes of each other family), SPLIT_KBT (default 2), MAX_EVALS (default 1000), SUBGROUP=proper (optional)
"""
import heapq
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from merrill_diag import read_tec, ang_deg, tet_geometry, node_weights  # noqa: E402
from merrill_runner import P1Interpolator  # noqa: E402

KBT = 1.380649e-23 * 300.0
F0 = 1e9
RUNNER = str(HERE / "merrill_runner.py")


def head(mesh):
    return ["Set: MaxMeshNumber = 1", f"ReadMesh: mesh=1, filename={mesh}", "Set: MaxEnergyEvaluations = 20000",
            "Material: Type=Magnetite, TempC=20", "ConjugateGradient"]


def merrill(mbin, wd, name, lines):
    (wd / f"{name}.mscr").write_text("\n".join(lines + ["End"]) + "\n")
    with open(wd / f"{name}.log", "w") as lf:
        subprocess.run([mbin, f"{name}.mscr"], cwd=wd, stdout=lf, stderr=subprocess.STDOUT, check=True)


def candidates():
    out = []
    for perm in itertools.permutations(range(3)):
        for sg in itertools.product([1, -1], repeat=3):
            R = np.zeros((3, 3))
            for i, j in enumerate(perm):
                R[i, j] = sg[i]
            if os.environ.get("SUBGROUP") == "proper" and np.linalg.det(R) < 0:
                continue
            out.append(R)
    return out


def energies(mbin, wd, mname, files, tag):
    lines = head(mname) + [f"Set: PathN = {len(files)}"]
    for i, f in enumerate(files, 1):
        lines += [f"ReadMagnetization: filename={f}", f"MagnetizationToPath: {i}"]
    merrill(mbin, wd, tag, lines + ["RenewPath", f"PathStructureEnergies: filename={tag}.txt"])
    return np.array([float(l.split()[-1]) for l in open(wd / f"{tag}.txt") if l.strip()])


def main():
    mbin, mesh, wd = str(Path(sys.argv[1]).resolve()), sys.argv[2], Path(sys.argv[3])
    seeds = sys.argv[4:]
    K, split_kbt, max_evals = int(os.environ.get("K", "4")), float(os.environ.get("SPLIT_KBT", "2")), os.environ.get("MAX_EVALS", "1000")
    wd.mkdir(parents=True, exist_ok=True)
    mname = Path(mesh).name
    shutil.copy(mesh, wd / mname)
    for i, s in enumerate(seeds):
        shutil.copy(s, wd / f"seed{i}.dat")
    T0 = time.time()
    timing = {}
    merrill(mbin, wd, "topo", head(mname) + ["ReadMagnetization: filename=seed0.dat", "WriteMagnetization: filename=topo"])
    c, t, _ = read_tec(wd / "topo_mult.tec")
    cen = c.mean(0)
    vol, _ = tet_geometry(c, t)
    w = node_weights(c, t, vol)
    d = lambda x, y: float(np.sqrt(np.sum(w * ang_deg(x, y) ** 2)))
    P = P1Interpolator(c, t)
    L = lambda f: (lambda a: a[:, 3:6] / np.linalg.norm(a[:, 3:6], axis=1, keepdims=True))(np.loadtxt(f))

    # 1. group
    tree = cKDTree(c - cen)
    hmin = float(np.min(np.linalg.norm(c[t[:, 1]] - c[t[:, 0]], axis=1)))
    shape = [R for R in candidates() if tree.query((c - cen) @ R.T)[0].max() < 1e-3 * hmin]
    G = [(R, tr) for R in shape for tr in (1, -1)]
    act = lambda g, m: (lambda r: r / np.linalg.norm(r, axis=1, keepdims=True))(
        g[1] * np.linalg.det(g[0]) * (g[0] @ P((c - cen) @ g[0] + cen, m).T).T)

    # 2. nodes (orbits of every seed family); family energy from the relaxed seed
    Es = energies(mbin, wd, mname, [f"seed{i}.dat" for i in range(len(seeds))], "seedE")
    nodes, fam = [], []
    for i in range(len(seeds)):
        m = L(wd / f"seed{i}.dat")
        for g in G:
            mi = act(g, m)
            if all(d(mi, s) >= 5.0 for s in nodes):
                nodes.append(mi)
                fam.append(i)
    n = len(nodes)
    E = np.array([(Es[f] - Es.min()) / KBT for f in fam])  # kBT_300, family energy of the relaxed seed
    # permutation table perm[g][a] = index of g.node_a
    perm = []
    for g in G:
        row = []
        for a in range(n):
            ga = act(g, nodes[a])
            row.append(int(np.argmin([d(ga, s) for s in nodes])))
        perm.append(row)
    timing["orbits_s"] = round(time.time() - T0, 1)
    D = np.array([[d(nodes[a], nodes[b]) for b in range(n)] for a in range(n)])
    m0, mneg = 0, int(np.argmin([d(-nodes[0], s) for s in nodes]))

    # 3. short-edge graph
    edges = set()
    fam_arr = np.array(fam)
    kx = int(os.environ.get("K_CROSS", "2"))
    for a in range(n):
        for b in np.argsort(D[a])[1:K + 1]:
            edges.add(tuple(sorted((a, int(b)))))
        for f in set(fam) - {fam[a]}:  # and its kx nearest nodes of every other family
            idx = np.where(fam_arr == f)[0]
            for b in idx[np.argsort(D[a, idx])[:kx]]:
                edges.add(tuple(sorted((a, int(b)))))
    edges.add(tuple(sorted((m0, mneg))))  # always include the direct reversal edge (the single band), so minimax <= single band
    # 4. classes of unordered pairs
    canon = lambda e: min(tuple(sorted((perm[gi][e[0]], perm[gi][e[1]]))) for gi in range(len(G)))
    classes = {}
    for e in sorted(edges):
        classes.setdefault(canon(e), []).append(e)
    # all images of every class representative (the full edge orbit), not only the short edges found
    edge_orbit = {r: sorted({tuple(sorted((perm[gi][r[0]], perm[gi][r[1]]))) for gi in range(len(G))}) for r in classes}

    # 5. relax endpoints of class representatives, NEB per class
    ends = sorted({x for r in classes for x in r})
    lines = head(mname)
    for a in ends:
        np.savetxt(wd / f"node{a}.dat", np.hstack([c, nodes[a]]), fmt="%.10e")
        lines += [f"ReadMagnetization: filename=node{a}.dat", "Minimize", f"WriteMagnetization: filename=node{a}_r"]
    t1 = time.time()
    merrill(mbin, wd, "relax", lines)
    timing["relax_s"] = round(time.time() - t1, 1)
    moved = {a: round(d(L(wd / f"node{a}_r.dat"), nodes[a]), 2) for a in ends}
    hop = {}
    t2 = time.time()
    for r in classes:
        a, b = r
        env = dict(os.environ, END_DAT=str(wd / f"node{b}_r.dat"))
        hw = wd / f"hop_{a}_{b}"
        th = time.time()
        with open(wd / f"hop_{a}_{b}.out", "w") as f:
            subprocess.run([sys.executable, "-I", RUNNER, mbin, str(wd / mname), str(wd / f"node{a}_r.dat"), str(hw),
                            "default", "-", max_evals], env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
        e_img = np.array([float(l.split()[-1]) for l in open(hw / "energies.txt") if l.strip()]) / KBT
        txt = (hw / "run.log").read_text()
        stop = ("failed" if "FAILED TO CONVERGE" in txt else "max_evals" if "MAX Path" in txt
                else "stagnation" if "DeltaS negligible" in txt else "?")
        evals = sum(int(x) for x in re.findall(r"CalcPathAction Calls:\s*(\d+)", txt))
        prof = e_img - e_img[0] + (E[a] * 1.0)  # profile on the global energy scale (kBT), anchored at node a
        smax = float(prof.max())
        # 6. interior minima well below both neighbours
        splits = []
        for i in range(1, len(prof) - 1):
            if prof[i] < prof[i - 1] - split_kbt and prof[i] < prof[i + 1] - split_kbt:
                splits.append(i + 1)
        hop[f"{a}-{b}"] = {"saddle_E_kBT": round(smax, 2), "barrier_from_a": round(smax - E[a], 2), "barrier_from_b": round(smax - E[b], 2),
                           "stop": stop, "evals": evals, "wall_s": round(time.time() - th, 1), "interior_minima_images": splits,
                           "profile_kBT": [round(float(v), 1) for v in prof]}
    timing["hops_s"] = round(time.time() - t2, 1)

    # split check: relax interior minima, report whether they are new states
    new_states = []
    for key, h in hop.items():
        if not h["interior_minima_images"]:
            continue
        a, b = map(int, key.split("-"))
        hw = wd / f"hop_{a}_{b}"
        _, _, path = read_tec(hw / "finalpath.tec")
        for i in h["interior_minima_images"]:
            f = f"split_{a}_{b}_{i}"
            np.savetxt(wd / f"{f}.dat", np.hstack([c, path[i - 1]]), fmt="%.10e")
            merrill(mbin, wd, f, head(mname) + [f"ReadMagnetization: filename={f}.dat", "Minimize", f"WriteMagnetization: filename={f}_r"])
            ms = L(wd / f"{f}_r.dat")
            dmin = min(d(ms, s) for s in nodes)
            new_states.append({"hop": key, "image": i, "rms_to_nearest_node_deg": round(dmin, 1), "new": bool(dmin >= 5.0),
                               "file": f"{f}_r.dat"})

    # 7. network over all images of every computed class
    W = {}
    for r, imgs in edge_orbit.items():
        h = hop[f"{r[0]}-{r[1]}"]
        for (a, b) in imgs:
            # orientation of the image relative to the representative: energies are G-invariant, so the
            # saddle energy is the same for every image
            W[(a, b)] = W[(b, a)] = min(W.get((a, b), 1e18), h["saddle_E_kBT"])
    adj = {}
    for (a, b), s in W.items():
        adj.setdefault(a, {})[b] = s

    def minimax(src):
        best = {src: E[src]}
        pq = [(E[src], src)]
        while pq:
            cost, u = heapq.heappop(pq)
            if cost > best.get(u, 1e18):
                continue
            for v, s in adj.get(u, {}).items():
                nc = max(cost, s)
                if nc < best.get(v, 1e18):
                    best[v] = nc
                    heapq.heappush(pq, (nc, v))
        return best

    b0 = minimax(m0)
    res = {"n_shape_group": len(shape), "n_group_with_T": len(G), "n_families": len(seeds), "n_nodes": n,
           "family_energy_kBT": {i: round(float((Es[i] - Es.min()) / KBT), 2) for i in range(len(seeds))},
           "nodes_per_family": {i: fam.count(i) for i in range(len(seeds))}, "K": K, "K_CROSS": kx, "n_short_edges": len(edges),
           "n_edge_classes": len(classes), "n_edges_in_network": len(W) // 2, "endpoint_relax_shift_deg_max": max(moved.values()),
           "minus_m0_node": mneg, "minus_m0_match_deg": round(d(-nodes[0], nodes[mneg]), 2),
           "minimax_m0_to_minus_m0_kBT": round(b0.get(mneg, float("nan")) - E[m0], 2) if mneg in b0 else None,
           "hops": {k: {kk: vv for kk, vv in v.items() if kk != "profile_kBT"} for k, v in hop.items()},
           "split_states": new_states}
    # escape barrier of every family representative: lowest saddle on any edge leaving its node
    esc = {}
    for i in range(len(seeds)):
        a = fam.index(i)
        out = [s - E[a] for (x, y), s in W.items() if x == a]
        esc[i] = round(min(out), 2) if out else None
    res["escape_barrier_by_family_kBT"] = esc
    # rate matrix (Arrhenius, f0 = 1e9 s^-1) over the nodes; written as CSV
    Kmat = np.zeros((n, n))
    for (a, b), s in W.items():
        Kmat[a, b] = F0 * np.exp(-(s - E[a]))
    np.savetxt(wd / "rate_matrix_per_s.csv", Kmat, delimiter=",", fmt="%.6e")
    np.savetxt(wd / "node_energy_kBT.csv", np.column_stack([np.arange(n), fam, E]), delimiter=",", fmt=["%d", "%d", "%.4f"],
               header="node,family,E_kBT", comments="")
    timing["total_s"] = round(time.time() - T0, 1)
    res["timing"] = timing
    res["total_path_evals"] = sum(h["evals"] for h in hop.values())
    (wd / "network.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "hops"}, indent=1, default=float))
    for k, v in res["hops"].items():
        print(k, v)


if __name__ == "__main__":
    main()
