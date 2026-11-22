"""Quotient network for the Chen 6:5:4 cuboid at 105 nm on an nx grid. Families from chen_grid_states.py.
Usage: chen_fd_network_grid.py nx fam1,fam2,...  (e.g. vortex  or  vortex,perturbed_flower,tilted_x)"""
import json, sys, time
import numpy as np
import os
from _paths import DATA, FIGURES, MERRILL, RUNS  # noqa: F401
os.makedirs(RUNS / "chen", exist_ok=True)
os.chdir(RUNS / "chen")  # inputs and outputs of the cuboid runs live in <RUNS>/chen
from symbarrier.backends.magnumnp_backend import MagnumNPBackend, MAGNETITE_MERRILL_20C
from symbarrier.workflow import run_network, save_cache, load_cache
nx = int(sys.argv[1]); fams = sys.argv[2].split(","); N = (nx, nx * 5 // 6, nx * 4 // 6)
be = MagnumNPBackend(N, (105e-9 / nx,) * 3, material=MAGNETITE_MERRILL_20C, gneb_kw=dict(max_iter=3000))
S = np.load(f"chen_states_{nx}.npz"); seeds = [S[f] for f in fams]
tag = f"chen_net_{nx}_{'+'.join(fams)}"
cache = []
for c in sys.argv[3:]:
    cache += load_cache(c)
t = time.time()
r = run_network(be, seeds, outdir=tag, K=4, K_cross=2, relax_nodes=False, cache=cache)
save_cache(cache, f"{tag}/cache.npz")
json.dump({"families": fams, "N": N, "wall_s": round(time.time() - t, 1), "n_classes": r["n_classes"],
           "minimax": r["minimax_m0_to_minus_m0_kT"], "escape": r["escape_barrier_by_family_kT"],
           "hops": {k: {kk: v.get(kk) for kk in ("saddle_E_kT", "barrier_from_a", "barrier_from_b", "max_neighbour_rms_deg", "reused", "info")} for k, v in r["hops"].items()}},
          open(f"{tag}/summary.json", "w"), indent=1, default=str)
print("wall", time.time() - t)
