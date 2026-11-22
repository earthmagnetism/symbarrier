"""Full network on the FD backend: prism 120x120x200 nm, 5 nm cells, MERRILL 20 C constants + K2 (validation
against the stand-alone FD alignment run, 1064.4 kT for the 96.6 deg class)."""
import json, sys, time
import numpy as np
from _paths import DATA, FIGURES, MERRILL, RUNS  # noqa: F401
from symbarrier.backends.magnumnp_backend import MagnumNPBackend, MAGNETITE_MERRILL_20C
from symbarrier.workflow import explore
be = MagnumNPBackend((24, 24, 40), (5e-9,) * 3, material=MAGNETITE_MERRILL_20C, gneb_kw=dict(max_iter=3000))
m0 = np.load(f"{RUNS}/prism_m0.npy")
t = time.time()
r = explore(be, [m0], outdir=f"{RUNS}/prism_fd", relax_nodes=False)
json.dump({"hops": r["hops"], "history": r["history"], "wall_s": time.time() - t}, open(f"{RUNS}/prism_fd/summary.json", "w"), indent=1, default=float)
print("wall", time.time() - t)
