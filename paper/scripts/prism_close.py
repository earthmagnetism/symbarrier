"""Close the FD prism network over the two states found on the 180-degree band (Algorithm 2, one round, paths
computed by two worker processes). Modes: plan | work <k> <n> | assemble"""
import json, sys, time
import numpy as np
from _paths import DATA, FIGURES, MERRILL, RUNS  # noqa: F401
from symbarrier.backends.magnumnp_backend import MagnumNPBackend, MAGNETITE_MERRILL_20C
from symbarrier import group as grp, network as net
from symbarrier.geometry import rms_angle_deg
from symbarrier.workflow import run_network, save_cache, load_cache

W = f"{RUNS}/prism_close"
be = MagnumNPBackend((24, 24, 40), (5e-9,) * 3, material=MAGNETITE_MERRILL_20C, gneb_kw=dict(max_iter=3000),
                     relax_kw=dict(maxiter=50000))
m0 = np.load(f"{RUNS}/prism_m0.npy")
seeds = [m0, np.load(f"{RUNS}/prism_state_9.npy"), np.load(f"{RUNS}/prism_state_4.npy")]
G = grp.with_time_reversal(be.point_group())
act = be.action()
w = be.weights


def old_cache():
    nodes0, _, _ = net.build_nodes([m0], G, act, w)
    E0 = float(be.energy([m0])[0])
    r1 = json.load(open(f"{RUNS}/prism_fd/round1/network.json"))["hops"]
    return [{"start": nodes0[int(k.split("-")[0])], "end": nodes0[int(k.split("-")[1])], "saddle_abs_kT": v["saddle_E_kT"] + E0}
            for k, v in r1.items()]


def plan():
    Es = be.energy(seeds)
    nodes, fam, perm = net.build_nodes(seeds, G, act, w)
    D = net.distance_matrix(nodes, w)
    neg = int(np.argmin([rms_angle_deg(-nodes[0], s, w) for s in nodes]))
    edges = net.short_edge_graph(D, fam, 4, 2, always=[(0, neg)])
    classes, orbit = net.edge_classes(edges, perm)
    cache = old_cache()
    cp = {}
    for k, c in enumerate(cache):
        a = int(np.argmin([rms_angle_deg(c["start"], s, w) for s in nodes])); b = int(np.argmin([rms_angle_deg(c["end"], s, w) for s in nodes]))
        cp[tuple(sorted((a, b)))] = k
    todo = [list(map(int, r)) for r in classes if not any(p in cp for p in orbit[r])]
    print("nodes", len(nodes), "per family", [fam.count(i) for i in range(3)], "E", np.round(Es - Es.min(), 2), "classes", len(classes), "todo", todo)
    json.dump({"todo": todo}, open(f"{W}/plan.json", "w"))
    np.save(f"{W}/nodes.npy", np.stack(nodes))


def work(k, n, only=None):
    todo = json.load(open(f"{W}/plan.json"))["todo"][k::n] if only is None else [only]
    tag = str(k) if only is None else f"_pair_{only[0]}_{only[1]}"
    nodes = np.load(f"{W}/nodes.npy")
    E0 = float(be.energy([m0])[0])
    out = []
    for a, b in todo:
        t = time.time()
        h = be.neb(nodes[a], nodes[b], n_images=19)
        p = np.array(h["profile_kT"])
        sp = max(rms_angle_deg(h["path"][i], h["path"][i + 1], w) for i in range(18))
        idx = [i for i in range(1, 18) if p[i] < p[i - 1] - 2 and p[i] < p[i + 1] - 2] if np.all(np.isfinite(p)) else []
        mins = []
        for i, m in zip(idx, be.relax([h["path"][i] for i in idx])):
            mins.append({"image": i, "E_kT": round(float(be.energy([m])[0]) - E0, 2),
                         "rms_to_nodes_deg": round(min(rms_angle_deg(m, s, w) for s in nodes), 1)})
        out.append({"pair": [a, b], "saddle_abs_kT": float(np.max(p)), "profile_rel_m0_kT": np.round(p - E0, 2).tolist(),
                    "max_neighbour_rms_deg": round(sp, 1), "info": h["info"], "minima": mins, "wall_s": round(time.time() - t, 1)})
        np.save(f"{W}/ends_{a}_{b}.npy", np.stack([nodes[a], nodes[b]]))
        json.dump(out, open(f"{W}/work{tag}.json", "w"), indent=1)
        print(a, b, round(float(np.max(p)) - E0, 2), h["info"]["converged"], flush=True)


def assemble():
    nodes = np.load(f"{W}/nodes.npy")
    cache = old_cache()
    for f in sorted(__import__("glob").glob(f"{W}/work*.json")):
        for r in json.load(open(f)):
            if np.isfinite(r["saddle_abs_kT"]):
                a, b = r["pair"]
                cache.append({"start": nodes[a], "end": nodes[b], "saddle_abs_kT": r["saddle_abs_kT"]})
    res = run_network(be, seeds, outdir=f"{W}/out", cache=cache, relax_nodes=False)
    assert res["n_new_paths"] == 0
    print(json.dumps({k: v for k, v in res.items() if k not in ("hops", "paths", "split_candidates", "nodes", "node_family")}, indent=1, default=float))


if __name__ == "__main__":
    import os; os.makedirs(W, exist_ok=True)
    {"plan": plan, "work": lambda: work(int(sys.argv[2]), int(sys.argv[3])), "pair": lambda: work(0, 1, [int(sys.argv[2]), int(sys.argv[3])]), "assemble": assemble}[sys.argv[1]]()
