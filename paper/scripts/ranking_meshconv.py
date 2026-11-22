"""Mesh convergence of the state ranking at 80 and 100 nm (truncated octahedron, MERRILL 20 C):
relax <100>, <110> and <111> vortex seeds at several mesh sizes; energies relative to the lowest."""
import json, sys, time
import numpy as np
from _paths import DATA, FIGURES, MERRILL, RUNS  # noqa: F401
from symbarrier.backends.merrill_backend import MerrillBackend
from symbarrier.geometry import write_patran
from symbarrier.meshes import truncated_octahedron, analytic_vortex
out = []
for size, subs in ((80, (2, 3, 4, 5)), (100, (3, 4, 5))):
    for n in subs:
        c, t = truncated_octahedron(size, n)
        wd = f"{RUNS}/rank/S{size}_n{n}"
        import os; os.makedirs(wd, exist_ok=True)
        write_patran(f"{wd}/to.pat", c, t)
        be = MerrillBackend(MERRILL, f"{wd}/to.pat", wd)
        seeds = {ax: analytic_vortex(be.coords, a, core_radius=0.15 * np.ptp(be.coords[:, 0]))
                 for ax, a in (("100", [1, 0, 0]), ("110", [1, 1, 0]), ("111", [1, 1, 1]))}
        t0 = time.time()
        rel = be.relax(list(seeds.values()))
        E = be.energy(rel)
        row = {"size_nm": size, "n_sub": n, "h_nm": round(size / (4 * n), 2), "nodes": len(be.coords), "relax_s": round(time.time() - t0, 1)}
        for (ax, _), m, e in zip(seeds.items(), rel, E):
            mm = (be.weights[:, None] * m).sum(0)
            row[ax] = {"E_rel_kT": round(float(e - E.min()), 2), "mean_m": round(float(np.linalg.norm(mm)), 3),
                       "dir": np.round(mm / np.linalg.norm(mm), 2).tolist()}
        print(json.dumps(row), flush=True)
        out.append(row)
json.dump(out, open(f"{RUNS}/ranking_meshconv.json", "w"), indent=1)
