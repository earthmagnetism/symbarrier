"""Quotient network for the Chen, Chang and Williams (2025) state set on the FD backend: vortex family only (4 nodes, 3 classes), 105 nm, 24 x 20 x 16 cells."""
import json, sys, time
import numpy as np
import os
from _paths import DATA, FIGURES, MERRILL, RUNS  # noqa: F401
os.makedirs(RUNS / "chen", exist_ok=True)
os.chdir(RUNS / "chen")  # inputs and outputs of the cuboid runs live in <RUNS>/chen
from symbarrier.backends.magnumnp_backend import MagnumNPBackend, MAGNETITE_MERRILL_20C
from symbarrier.workflow import run_network, save_cache
Lnm = float(sys.argv[1]); N = (24, 20, 16); dx = Lnm * 1e-9 / 24
be = MagnumNPBackend(N, (dx,) * 3, material=MAGNETITE_MERRILL_20C, gneb_kw=dict(max_iter=3000))
sd, sv = np.load(f"chen_seeds_{int(Lnm)}.npy")
cache = []
t = time.time()
r = run_network(be, [sv], outdir=f"chen_fd_sv_{int(Lnm)}", K=3, K_cross=0, relax_nodes=False, cache=cache)
save_cache(cache, f"chen_fd_sv_{int(Lnm)}/cache.npz")
json.dump({"wall_s": time.time() - t, "hops": r["hops"], "minimax": r["minimax_m0_to_minus_m0_kT"]},
          open(f"chen_fd_sv_{int(Lnm)}/summary.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
print("wall", time.time() - t)
