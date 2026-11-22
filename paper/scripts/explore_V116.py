"""Adaptive exploration on natural grain V116 (MERRILL backend, crystal = mesh axes), from the three LEM families."""
import json, sys, time
import numpy as np
from _paths import DATA, FIGURES, MERRILL, RUNS  # noqa: F401
from symbarrier.backends.merrill_backend import MerrillBackend
from symbarrier.workflow import explore
src = f"{RUNS}/natural/V116/run/net"
be = MerrillBackend(MERRILL, f"{src}/V116_mesh.pat", f"{RUNS}/V116_explore")
seeds = []
for i in range(3):
    a = np.loadtxt(f"{src}/seed{i}.dat")
    assert np.allclose(a[:, :3], be.coords, atol=1e-6)
    seeds.append(a[:, 3:6] / np.linalg.norm(a[:, 3:6], axis=1, keepdims=True))
t = time.time()
r = explore(be, seeds, outdir=f"{RUNS}/V116_explore/out")
json.dump({"hops": r["hops"], "history": r["history"], "wall_s": time.time() - t}, open(f"{RUNS}/V116_explore/out/summary.json", "w"), indent=1, default=float)
print("wall", time.time() - t)
