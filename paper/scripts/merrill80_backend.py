"""MERRILL backend check (Section 5.1): network of the 80 nm truncated octahedron from the <111> vortex m0.
Inputs in <RUNS>/merrill80: to_80nm.pat (make_meshes.py to 80 2) and m0.dat (x y z mx my mz per node)."""
import json, sys, time
import numpy as np
from _paths import DATA, FIGURES, MERRILL, RUNS  # noqa: F401
from symbarrier.backends.merrill_backend import MerrillBackend
from symbarrier.workflow import run_network
be = MerrillBackend(MERRILL, f"{RUNS}/merrill80/to_80nm.pat", f"{RUNS}/merrill80")
m0 = np.loadtxt(f"{RUNS}/merrill80/m0.dat")[:, 3:6]
assert np.allclose(np.loadtxt(f"{RUNS}/merrill80/m0.dat")[:, :3], be.coords, atol=1e-6)
t = time.time()
r = run_network(be, [m0 / np.linalg.norm(m0, axis=1, keepdims=True)], outdir=f"{RUNS}/merrill80/out")
print(json.dumps({k: v for k, v in r.items() if k not in ("hops", "paths", "split_candidates", "nodes", "node_family")}, default=float))
print("wall", time.time() - t)
