#!/usr/bin/env python3
"""
Symmetry-decomposed barrier calculation for the FD (magnum.np) systems, same protocol as
merrill_symdecomp.py, plus a 'band' mode that times a single covariant band with identical solver settings.

Systems: truncoct (voxelised, O, masked), cube (O), prism (D4; start from the transverse ground state).
Solver: GNEB-FIRE (fd_defs.run_gneb_fire) with ci_gate=5e3, dt_max=6e-5 for every run.

Usage:
  fd_symdecomp.py decomp <system> <workdir>
  fd_symdecomp.py band   <system> <ax,ay,az> <workdir>
"""
import heapq
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from _paths import DATA, FIGURES, MERRILL, RUNS  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent))
torch.set_default_dtype(torch.float64)
KW = dict(ci_gate=5e3, dt_max=6e-5, max_iter=3000)
R_ = RUNS / "benchmarks"


def load(system):
    if system == "truncoct":
        import fd_truncoct as T
        st, tm, mn = T.setup()
        g = T.__dict__
        m0 = torch.load(R_ / "tofd" / "m0.pt")
        return dict(st=st, tm=tm, mn=mn, fix=T.fix, inside=(T.MASK > 0), m0=m0, N=T.N, dx=T.DX, ns=g)
    ns = {}
    exec(open(Path(__file__).resolve().parent / "fd_defs.py").read(), ns)
    N = (30, 30, 30) if system == "cube" else (24, 24, 40)
    dx = (5e-9,) * 3
    st, tm, mn = ns["setup_magnetite"](N, dx)
    if system == "cube":
        import os
        if os.environ.get("START_NPY"):
            st.m = torch.tensor(np.load(os.environ["START_NPY"]).astype(np.float64))
        else:
            st.m = ns["make_seed_vortex"](N, dx, [0, 0, 1], st.m.device, st.m.dtype)
        mn.minimize(st)
        m0 = st.m.clone()
    else:
        st.m = torch.tensor(np.load(R_ / "fd_c2" / "B2_ground.npy").astype(np.float64))
        mn.minimize(st)
        m0 = st.m.clone()
    return dict(st=st, tm=tm, mn=mn, fix=lambda m: m / m.norm(dim=-1, keepdim=True), inside=torch.ones(N, dtype=bool),
                m0=m0, N=N, dx=dx, ns=ns)


def rotations(N):
    out = []
    for perm in itertools.permutations(range(3)):
        if tuple(N[p] for p in perm) != tuple(N):
            continue
        for sg in itertools.product([1, -1], repeat=3):
            R = np.zeros((3, 3))
            for i, j in enumerate(perm):
                R[i, j] = sg[i]
            out.append(R)  # full point group incl. improper elements (m is an axial vector)
    return out


def rot_grid(m, R):
    perm = [int(np.nonzero(R[a])[0][0]) for a in range(3)]
    sgn = [int(R[a, perm[a]]) for a in range(3)]
    arr = m.permute(*perm, 3)
    for a in range(3):
        if sgn[a] < 0:
            arr = torch.flip(arr, dims=[a])
    return float(np.linalg.det(R)) * torch.einsum("ij,...j->...i", torch.tensor(R, dtype=torch.float64), arr)


def main():
    mode, system = sys.argv[1], sys.argv[2]
    S = load(system)
    st, tm, mn, fix, inside, m0, N, dx = (S[k] for k in ["st", "tm", "mn", "fix", "inside", "m0", "N", "dx"])
    ns = S["ns"]
    run_gneb_fire, build_slerp_random_band, build_so3_covariant_band = ns["run_gneb_fire"], ns["build_slerp_random_band"], ns["build_so3_covariant_band"]
    KB = ns["KBT_300"]
    neg = fix(-m0 * ns["M4"] + ns["OUT"]) if system == "truncoct" else -m0

    def E(m):
        st.m = m
        return float(sum(t.E(st) for t in tm))

    def d(a, b):
        x = torch.acos((a * b).sum(-1).clamp(-1, 1))[inside]
        return float(torch.rad2deg(torch.sqrt((x ** 2).mean())))

    E0 = E(m0)
    if mode == "band":
        axis, wd = [float(v) for v in sys.argv[3].split(",")], Path(sys.argv[4])
        wd.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        band = [fix(m) for m in build_so3_covariant_band(m0, neg, dx, 19, axis)]
        band[-1] = neg
        r = run_gneb_fire(st, tm, band, f"{system}_band", verbose=False, **KW)
        r["wall_s"] = round(time.time() - t0, 1)
        r["profile_rel_m0_kBT"] = [round((E(fix(m)) - E0) / KB, 2) for m in band]
        r["image_iterations"] = r["iters"] * (len(band) - 2)
        (wd / "band.json").write_text(json.dumps({k: r[k] for k in ["conv", "iters", "barrier_kBT", "final_maxF", "wall_s", "image_iterations", "profile_rel_m0_kBT"]}, indent=1))
        print(json.dumps({k: r[k] for k in ["conv", "iters", "barrier_kBT", "final_maxF", "wall_s", "image_iterations"]}))
        return
    wd = Path(sys.argv[3])
    wd.mkdir(parents=True, exist_ok=True)
    timing = {}
    t0 = time.time()
    Rs = rotations(N)
    imgs = [fix(rot_grid(m0, R)) for R in Rs]
    nodes, node_of = [], []
    for mi in imgs:
        k = next((j for j, s in enumerate(nodes) if d(mi, s) < 5.0), None)
        if k is None:
            nodes.append(mi)
            k = len(nodes) - 1
        node_of.append(k)
    n_m0 = node_of[[np.allclose(R, np.eye(3)) for R in Rs].index(True)]
    dneg = [d(s, neg) for s in nodes]
    n_neg = int(np.argmin(dneg))
    dist = [d(s, m0) for s in nodes]
    # exact edge classes: orbits of the stabiliser of m0 acting on the orbit states
    stab = [gi for gi in range(len(Rs)) if node_of[gi] == n_m0]
    rep_rot = {k: node_of.index(k) for k in range(len(nodes))}
    def compose(i, j):
        P_ = Rs[i] @ Rs[j]
        return next(q for q, Q in enumerate(Rs) if np.allclose(Q, P_))
    classes = {}
    seen = set()
    for k in range(len(nodes)):
        if k in (n_m0, n_neg) or k in seen:
            continue
        members = sorted({node_of[compose(g, rep_rot[k])] for g in stab})
        seen.update(members)
        classes[min(members)] = members
    import os
    if os.environ.get("MAX_CLASSES"):
        keep = sorted(classes, key=lambda cl: d(nodes[cl], m0))[: int(os.environ["MAX_CLASSES"])]
        classes = {cl: classes[cl] for cl in keep}
    reps = {cl: cl for cl in classes}

    timing["orbit_s"] = round(time.time() - t0, 2)
    t1 = time.time()
    relaxed = {}
    for cl, k in reps.items():
        st.m = nodes[k]
        mn.minimize(st)
        relaxed[k] = fix(st.m.clone())
    timing["relax_s"] = round(time.time() - t1, 2)
    hop = {}
    t2 = time.time()
    for cl, k in reps.items():
        th = time.time()
        band = [fix(m) for m in build_slerp_random_band(m0, relaxed[k], 19, sigma=0.0, seed=0)]
        r = run_gneb_fire(st, tm, band, f"hop{k}", verbose=False, **KW)
        prof = [round((E(fix(m)) - E0) / KB, 2) for m in band]
        hop[k] = {"class": f"class{cl}_{dist[k]:.0f}deg", "rms_to_m0_deg": round(dist[k], 1), "conv": r["conv"], "iters": r["iters"],
                  "image_iterations": r["iters"] * 17, "barrier_from_m0_kBT": round(max(prof), 2),
                  "E_end_rel_m0_kBT": prof[-1], "final_maxF": r["final_maxF"], "wall_s": round(time.time() - th, 1),
                  "profile_rel_m0_kBT": prof}
        torch.save(torch.stack(band), wd / f"band_hop{k}.pt")
    timing["hops_s"] = round(time.time() - t2, 2)
    adj = {}
    for gi, R in enumerate(Rs):
        a_ = node_of[gi]
        for cl, k in reps.items():
            hi = node_of.index(k)
            gh = R @ Rs[hi]
            gj = next(j for j, Q in enumerate(Rs) if np.allclose(Q, gh))
            b_ = node_of[gj]
            wgt = hop[k]["barrier_from_m0_kBT"]
            for x, y in ((a_, b_), (b_, a_)):
                adj.setdefault(x, {})[y] = min(wgt, adj.get(x, {}).get(y, 1e18))
    best, prev, pq = {n_m0: 0.0}, {}, [(0.0, n_m0)]
    while pq:
        cost, u = heapq.heappop(pq)
        if cost > best.get(u, 1e18):
            continue
        for v, wgt in adj.get(u, {}).items():
            nc = max(cost, wgt)
            if nc < best.get(v, 1e18):
                best[v], prev[v] = nc, u
                heapq.heappush(pq, (nc, v))
    path = [n_neg]
    while path[-1] != n_m0 and path[-1] in prev:
        path.append(prev[path[-1]])
    timing["total_s"] = round(time.time() - t0, 2)
    info = {"system": system, "n_rotations": len(Rs), "n_orbit_states": len(nodes), "minus_m0_match_rms_deg": round(dneg[n_neg], 3),
            "classes": {f"class{cl}": {"rep": k, "members": classes[cl], "rms_to_m0_deg": round(dist[k], 1), "stabiliser_size": len(stab)} for cl, k in reps.items()},
            "hops": hop, "minimax_barrier_m0_to_minus_m0_kBT": best.get(n_neg), "minimax_path_nodes": path[::-1],
            "total_image_iterations": sum(h["image_iterations"] for h in hop.values()), "timing": timing}
    (wd / "symdecomp.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k != "hops"}, indent=1))
    for k, h in hop.items():
        print(k, {kk: vv for kk, vv in h.items() if kk != "profile_rel_m0_kBT"})


if __name__ == "__main__":
    main()
