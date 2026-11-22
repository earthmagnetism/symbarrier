"""End-to-end workflow on any backend: group detection -> orbits -> edge classes -> one path per class -> network
-> rates, plus the adaptive exploration loop that turns intermediate minima found on bands into new families.

Every quantity is computed; nothing is fitted. Energies are in units of kT of the backend.
"""
import json
import time
from pathlib import Path

import numpy as np

from . import group as grp
from . import network as net
from .geometry import rms_angle_deg


def _match(field, nodes, w, tol_deg):
    d = [rms_angle_deg(field, s, w) for s in nodes]
    k = int(np.argmin(d))
    return k if d[k] < tol_deg else None


def run_network(backend, seeds, outdir=None, K=4, K_cross=2, split_kT=2.0, proper_only=False,
                include_direct_edge=True, f0=1e9, n_images=19, relax_nodes=True, cache=None, tol_deg=5.0,
                verbose=True, cache_only=False):
    """Barrier network of the seed families on `backend`.

    seeds  list of relaxed states (arrays (n, 3)); seeds[0] is m0.
    cache  list of earlier paths, each {"start", "end", "saddle_abs_kT"} (from a previous call, or any other band,
           e.g. a repeated or transferred path). A class whose orbit contains a cached pair is not recomputed and takes
           the lowest cached saddle (every path maximum is an upper bound). The list is extended in place.
    Returns a dict (JSON-serialisable except for the keys 'paths' and 'split_candidates')."""
    t0 = time.time()
    cache = [] if cache is None else cache
    w = backend.weights
    P = backend.point_group(proper_only=proper_only)
    G = grp.with_time_reversal(P)
    act = backend.action()
    Es = np.asarray(backend.energy(seeds))
    nodes, fam, perm = net.build_nodes(seeds, G, act, w, tol_deg=tol_deg)
    E = np.array([Es[f] - Es.min() for f in fam])
    D = net.distance_matrix(nodes, w)
    m0 = 0
    neg = int(np.argmin([rms_angle_deg(-nodes[0], s, w) for s in nodes]))
    # cached paths -> node pairs in the present node list
    cached_pair = {}
    for k, c in enumerate(cache):
        a, b = _match(c["start"], nodes, w, tol_deg), _match(c["end"], nodes, w, tol_deg)
        if a is not None and b is not None and a != b:
            cached_pair.setdefault(tuple(sorted((a, b))), []).append(k)
    # every stored path is an upper bound on its edge, so it always enters the edge set
    always = ([(m0, neg)] if include_direct_edge else []) + list(cached_pair)
    edges = net.short_edge_graph(D, fam, K, K_cross, always=always)
    classes, orbit = net.edge_classes(edges, perm)
    timing = {"bookkeeping_s": round(time.time() - t0, 2)}

    todo = [r for r in classes if not any(p in cached_pair for p in orbit[r])]
    missing = []
    if cache_only:  # re-assemble from stored paths only; classes without a stored path are left out (no bound)
        missing, todo = [list(map(int, r)) for r in todo], []
    ends = sorted({x for r in todo for x in r})
    t1 = time.time()
    relaxed = dict(zip(ends, backend.relax([nodes[a] for a in ends]) if relax_nodes else [nodes[a] for a in ends]))
    timing["relax_s"] = round(time.time() - t1, 1)

    hops, saddle, paths, cands = {}, {}, {}, []
    t1 = time.time()
    for r in classes:
        hits = [k for p in orbit[r] for k in cached_pair.get(p, [])]
        if hits:  # every earlier path between images of this pair is an upper bound; keep the lowest
            saddle[r] = min(cache[k]["saddle_abs_kT"] for k in hits) - Es.min()
            hops[f"{r[0]}-{r[1]}"] = {"reused": True, "n_bounds": len(hits), "saddle_E_kT": round(saddle[r], 2)}
            continue
        if list(map(int, r)) in missing:
            saddle[r] = np.inf
            hops[f"{r[0]}-{r[1]}"] = {"missing": True}
            continue
        a, b = r
        if verbose:
            print(f"[symbarrier] path {a}-{b} ({len(todo)} new classes)", flush=True)
        h = backend.neb(relaxed[a], relaxed[b], n_images=n_images, tag=f"hop_{a}_{b}")
        prof_abs = np.asarray(h["profile_kT"], dtype=float)
        prof = prof_abs - prof_abs[0] + E[a]  # anchor the band to the energy of its start node
        saddle[r] = float(prof.max()) if np.all(np.isfinite(prof)) else np.inf  # a non-finite band bounds nothing
        usable = np.all(np.isfinite(prof)) and not h.get("info", {}).get("diverged", False)
        interior = [] if not usable else [i for i in range(1, len(prof) - 1) if prof[i] < prof[i - 1] - split_kT and prof[i] < prof[i + 1] - split_kT
                    and prof[i] < max(prof[:i]) - split_kT and prof[i] < max(prof[i + 1:]) - split_kT]
        cands += [h["path"][i] for i in interior]
        spacing = max(rms_angle_deg(h["path"][i], h["path"][i + 1], w) for i in range(len(h["path"]) - 1))
        hops[f"{a}-{b}"] = {"saddle_E_kT": round(saddle[r], 2), "max_neighbour_rms_deg": round(spacing, 1),
                            "barrier_from_a": round(saddle[r] - E[a], 2),
                            "barrier_from_b": round(saddle[r] - E[b], 2), "profile_kT": np.round(prof, 2).tolist(),
                            "interior_minima": interior, "info": h.get("info", {})}
        paths[r] = h["path"]
        if np.isfinite(saddle[r]):
            cache.append({"start": nodes[a], "end": nodes[b], "saddle_abs_kT": saddle[r] + Es.min()})
    timing["paths_s"] = round(time.time() - t1, 1)

    adj = net.assemble(orbit, saddle)
    best = net.minimax(adj, m0, E)
    esc = {}
    for i in range(len(seeds)):
        a = fam.index(i)
        out = [s - E[a] for s in adj.get(a, {}).values() if np.isfinite(s)]
        esc[i] = round(min(out), 2) if out else None
    Kr = net.rate_matrix(adj, E, f0)
    if not np.isfinite(best.get(neg, np.inf)):
        best.pop(neg, None)
    res = {"backend": backend.name, "n_point_group": len(P), "n_group_with_T": len(G), "n_nodes": len(nodes),
           "nodes_per_family": {i: fam.count(i) for i in range(len(seeds))},
           "family_energy_kT": {i: round(float(Es[i] - Es.min()), 2) for i in range(len(seeds))},
           "n_short_edges": len(edges), "edges": [list(map(int, e)) for e in edges], "n_classes": len(classes), "n_new_paths": len(todo), "missing_classes": missing,
           "n_network_edges": sum(len(v) for v in orbit.values()),
           "minimax_m0_to_minus_m0_kT": round(best[neg] - E[m0], 2) if neg in best else None,
           "escape_barrier_by_family_kT": esc, "hops": hops, "timing": timing}
    if outdir is not None:
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "network.json").write_text(json.dumps(res, indent=1, default=float))
        np.savetxt(out / "rate_matrix_per_s.csv", Kr, delimiter=",", fmt="%.6e")
    res["paths"], res["split_candidates"], res["nodes"], res["node_family"] = paths, cands, nodes, fam
    return res


def save_cache(cache, path):
    """Store computed paths (end fields and saddle energy) so that a later run can reuse them."""
    np.savez_compressed(path, start=np.stack([c["start"] for c in cache]), end=np.stack([c["end"] for c in cache]),
                        saddle_abs_kT=np.array([c["saddle_abs_kT"] for c in cache]))


def load_cache(path):
    z = np.load(path)
    return [{"start": a, "end": b, "saddle_abs_kT": float(e)} for a, b, e in zip(z["start"], z["end"], z["saddle_abs_kT"])]


def explore(backend, seeds, outdir=None, max_rounds=5, tol_deg=5.0, verbose=True, cache=None, **kw):
    """Adaptive exploration: run the network, relax the interior minima of its bands, add every relaxed state that
    is not in the orbit of a known family as a new family, and repeat until no new state appears. Paths are reused
    across rounds through the endpoint cache, so each round only computes the classes that involve new families.
    `cache` may carry paths from earlier runs (see load_cache); the lowest bound per edge is used."""
    fams, history = [np.asarray(s) for s in seeds], []
    cache = [] if cache is None else cache
    for rnd in range(1, max_rounds + 1):
        res = run_network(backend, fams, outdir=None if outdir is None else Path(outdir) / f"round{rnd}",
                          cache=cache, tol_deg=tol_deg, verbose=verbose, **kw)
        w, nodes = backend.weights, list(res["nodes"])
        new = []
        if res["split_candidates"]:
            G = grp.with_time_reversal(backend.point_group(proper_only=kw.get("proper_only", False)))
            act = backend.action()
            relaxed = backend.relax(res["split_candidates"])
            Er = np.asarray(backend.energy(relaxed), dtype=float)
            for m, e in zip(relaxed, Er):
                if not (np.all(np.isfinite(m)) and np.isfinite(e)):
                    continue  # failed relaxation: not a state
                if _match(m, nodes, w, tol_deg) is None:
                    new.append(m)
                    nodes += [act(g, m) for g in G]  # its whole orbit is now known
        history.append({"round": rnd, "n_families": len(fams), "n_nodes": res["n_nodes"], "n_classes": res["n_classes"],
                        "n_new_paths": res["n_new_paths"], "n_split_candidates": len(res["split_candidates"]),
                        "n_new_families": len(new), "minimax_kT": res["minimax_m0_to_minus_m0_kT"],
                        "escape_kT": res["escape_barrier_by_family_kT"], "timing": res["timing"]})
        if verbose:
            print(f"[symbarrier] round {rnd}: {history[-1]}", flush=True)
        if outdir is not None:  # checkpoint: states and paths so far, so an interrupted run can be resumed
            rd = Path(outdir) / f"round{rnd}"
            rd.mkdir(parents=True, exist_ok=True)
            if new:
                np.save(rd / "new_families.npy", np.stack(new))
                (rd / "new_family_energy_kT.json").write_text(json.dumps(
                    [float(e) for e in backend.energy(new) - np.min(backend.energy(fams))]))
            if cache:
                save_cache(cache, rd / "path_cache.npz")
        if not new:
            break
        fams += new
    res["history"], res["families"] = history, fams
    if outdir is not None:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        (Path(outdir) / "explore.json").write_text(json.dumps(history, indent=1, default=float))
        np.save(Path(outdir) / "families.npy", np.stack(fams))
        if cache:
            save_cache(cache, Path(outdir) / "path_cache.npz")
    return res
