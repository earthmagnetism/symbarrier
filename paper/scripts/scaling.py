"""Cost of the symmetry/network bookkeeping against mesh size and group order (no solver calls).

For each mesh: point-group detection, P1 interpolator set-up, orbit + permutation table (|G| actions of one seed),
distance matrix, short-edge graph and edge classes. Seed: analytic <111> vortex. Counts: nodes, short edges, classes,
all unordered pairs. Times are wall-clock on one pinned core (median of 3)."""
import json, sys, time
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from _paths import DATA, FIGURES, MERRILL, RUNS  # noqa: F401
from symbarrier import group as grp, network as net
from symbarrier.geometry import P1Interpolator, node_weights, read_patran, rms_angle_deg
from symbarrier.meshes import analytic_vortex, box


def bench(c, t, label, proper_only=False, reps=3):
    out = []
    for _ in range(reps):
        T = {}
        t0 = time.perf_counter(); P = grp.detect_point_group(c, t, proper_only=proper_only); T["detect"] = time.perf_counter() - t0
        G = grp.with_time_reversal(P)
        t0 = time.perf_counter()
        if t is None:
            tree = cKDTree(c); ip = lambda p, v: v[tree.query(p)[1]]; w = np.full(len(c), 1 / len(c))
        else:
            ip = P1Interpolator(c, t); w = node_weights(c, t)
        T["interp_setup"] = time.perf_counter() - t0
        act = grp.make_action(c, ip)
        m0 = analytic_vortex(c, [1, 1, 1], core_radius=0.12 * np.ptp(c[:, 0]))
        t0 = time.perf_counter(); nodes, fam, perm = net.build_nodes([m0], G, act, w); T["orbit_perm"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        D = net.distance_matrix(nodes, w)
        neg = int(np.argmin([rms_angle_deg(-nodes[0], s, w) for s in nodes]))
        edges = net.short_edge_graph(D, fam, 4, 2, always=[(0, neg)])
        cl, orb = net.edge_classes(edges, perm)
        T["graph_classes"] = time.perf_counter() - t0
        T["total"] = sum(T.values())
        out.append(T)
    med = {k: round(float(np.median([o[k] for o in out])), 3) for k in out[0]}
    n = len(nodes)
    row = {"mesh": label, "n_mesh_nodes": len(c), "n_tets": 0 if t is None else len(t), "G": len(P), "G_T": len(G), "orbit": n,
           "short_edges": len(edges), "classes": len(cl), "all_pairs": n * (n - 1) // 2,
           "network_edges": sum(len(v) for v in orb.values()), "time_s": med}
    print(json.dumps(row), flush=True)
    return row


rows = []
for s in ["080", "100", "150", "180", "250"]:
    c, t = read_patran(f"{RUNS}/sweep/S{s}/to_{int(s)}nm.pat")
    rows.append(bench(c, t, f"truncated octahedron {int(s)} nm"))
    if s == "150":
        rows.append(bench(c, t, "truncated octahedron 150 nm, proper only", proper_only=True))
        R = Rotation.random(random_state=1).as_matrix()
        rows.append(bench(c @ R.T, t, "truncated octahedron 150 nm, rotated"))
cb, tb = box(120, 120, 200, 10)
rows.append(bench(cb, tb, "prism 120x120x200 nm (FE)"))
for n in [(24, 24, 40), (36, 36, 60)]:
    d = 120e-9 / n[0]
    g = [(np.arange(k) + 0.5) * d - 0.5 * k * d for k in n]
    cg = np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3)
    rows.append(bench(cg, None, f"prism grid {n[0]}x{n[1]}x{n[2]} (FD)"))
json.dump(rows, open(f"{RUNS}/scaling.json", "w"), indent=1)
