"""Chen 6:5:4 cuboid at 105 nm: relax vortex and flower seeds on a given grid; report energy, orbit size,
stabiliser and net moment of each relaxed state. Usage: chen_grid_states.py nx (ny = 5nx/6, nz = 4nx/6)."""
import json, sys, time
import numpy as np
import os
from _paths import DATA, FIGURES, MERRILL, RUNS  # noqa: F401
os.makedirs(RUNS / "chen", exist_ok=True)
os.chdir(RUNS / "chen")  # inputs and outputs of the cuboid runs live in <RUNS>/chen
from symbarrier.backends.magnumnp_backend import MagnumNPBackend, MAGNETITE_MERRILL_20C
from symbarrier.meshes import analytic_vortex
from symbarrier import network as net, group as grp
from symbarrier.geometry import rms_angle_deg
nx = int(sys.argv[1]); N = (nx, nx * 5 // 6, nx * 4 // 6); L = 105e-9; dx = L / nx
be = MagnumNPBackend(N, (dx,) * 3, material=MAGNETITE_MERRILL_20C)
X = be.coords; n = len(X); w = be.weights
G = grp.with_time_reversal(be.point_group()); act = be.action()
rng = np.random.default_rng(0)
def unit(m): return m / np.linalg.norm(m, axis=1, keepdims=True)
seeds = {"vortex": analytic_vortex(X, [0.02, 0.01, 1], core_radius=0.15 * np.ptp(X[:, 2]), handed=1),
         "uniform_x": np.tile([1.0, 0, 0], (n, 1)),
         "tilted_x": unit(np.tile([1.0, 0.05, 0.03], (n, 1)))}
t = time.time()
names = list(seeds); R = dict(zip(names, be.relax([seeds[k] for k in names])))
R["perturbed_flower"] = be.relax([unit(R["uniform_x"] + 0.05 * rng.standard_normal((n, 3)))])[0]
for s in range(1, 4):  # further random restarts around the flower
    R[f"perturbed_flower_{s}"] = be.relax([unit(R["uniform_x"] + 0.15 * rng.standard_normal((n, 3)))])[0]
E = dict(zip(R, np.asarray(be.energy(list(R.values())))))
def lab(g):
    Rm, tt = g; return ("T·" if tt < 0 else "") + ("diag" + str(tuple(int(x) for x in np.diag(Rm))) if np.allclose(Rm, np.diag(np.diag(Rm))) else str(Rm.astype(int).tolist()))
out = {"N": N, "dx_nm": dx * 1e9, "relax_s": round(time.time() - t, 1), "states": {}}
for k, m in R.items():
    stab = [lab(g) for g in G if rms_angle_deg(act(g, m), m, w) < 2.0]
    out["states"][k] = {"E_minus_vortex_kT": round(float(E[k] - E["vortex"]), 2), "orbit": len(G) // len(stab),
                        "stabiliser": stab, "mean_m": np.round(m.mean(0), 3).tolist()}
np.savez(f"chen_states_{nx}.npz", **R)
json.dump(out, open(f"chen_states_{nx}.json", "w"), indent=1)
print(json.dumps({k: (v["E_minus_vortex_kT"], v["orbit"], v["mean_m"]) for k, v in out["states"].items()}), out["relax_s"], "s")
