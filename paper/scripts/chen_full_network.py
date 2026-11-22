"""Full quotient network for the Chen 6:5:4 cuboid (105 nm): short-axis vortex (m0), long-axis vortex, tilted flower.
Reuses the vortex-only classes from the cache; records the Bloch-point charge along every new path.
Usage: chen_full_network.py nx [cache.npz]"""
import json, sys, time
import numpy as np
import os
from _paths import DATA, FIGURES, MERRILL, RUNS  # noqa: F401
os.makedirs(RUNS / "chen", exist_ok=True)
os.chdir(RUNS / "chen")  # inputs and outputs of the cuboid runs live in <RUNS>/chen
from symbarrier.backends.magnumnp_backend import MagnumNPBackend, MAGNETITE_MERRILL_20C
from symbarrier.workflow import run_network, save_cache, load_cache
from chen_bloch_point_lib import max_cube_charge
nx = int(sys.argv[1]); N = (nx, nx * 5 // 6, nx * 4 // 6)
be = MagnumNPBackend(N, (105e-9 / nx,) * 3, material=MAGNETITE_MERRILL_20C, gneb_kw=dict(max_iter=3000))
S = np.load(f"chen_states_{nx}.npz"); fams = ["vortex", "perturbed_flower", "tilted_x"]
cache = load_cache(sys.argv[2]) if len(sys.argv) > 2 else []
tag = f"chen_full_{nx}"; t = time.time()
r = run_network(be, [S[f] for f in fams], outdir=tag, K=4, K_cross=2, relax_nodes=False, cache=cache)
save_cache(cache, f"{tag}/cache.npz")
bp = {f"{a}-{b}": [round(float(max_cube_charge(np.asarray(m).reshape(*N, 3))), 2) for m in path] for (a, b), path in r["paths"].items()}
fam = r["node_family"]
json.dump({"families": fams, "N": N, "wall_s": round(time.time() - t, 1), "n_nodes": r["n_nodes"], "n_classes": r["n_classes"],
           "family_energy_kT": r["family_energy_kT"], "node_family": fam, "escape": r["escape_barrier_by_family_kT"],
           "minimax": r["minimax_m0_to_minus_m0_kT"], "bloch_charge_per_image": bp,
           "hops": {k: {kk: v.get(kk) for kk in ("saddle_E_kT", "barrier_from_a", "barrier_from_b", "max_neighbour_rms_deg", "reused", "info")} for k, v in r["hops"].items()}},
          open(f"{tag}/summary.json", "w"), indent=1, default=str)
print("wall", time.time() - t)
