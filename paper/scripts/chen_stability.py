"""Small-noise stability test of the relaxed flower states (Chen cuboid, 105 nm). Usage: chen_stability.py nx"""
import sys, json, numpy as np
import os
from _paths import DATA, FIGURES, MERRILL, RUNS  # noqa: F401
os.makedirs(RUNS / "chen", exist_ok=True)
os.chdir(RUNS / "chen")  # inputs and outputs of the cuboid runs live in <RUNS>/chen
from symbarrier.backends.magnumnp_backend import MagnumNPBackend, MAGNETITE_MERRILL_20C
from symbarrier.geometry import rms_angle_deg
nx = int(sys.argv[1]); N = (nx, nx * 5 // 6, nx * 4 // 6)
be = MagnumNPBackend(N, (105e-9 / nx,) * 3, material=MAGNETITE_MERRILL_20C)
S = np.load(f"chen_states_{nx}.npz"); w = be.weights; rng = np.random.default_rng(1)
Ev = float(be.energy([S["vortex"]])[0]); res = {}
for k in ["uniform_x", "tilted_x", "perturbed_flower"]:
    for amp in (0.005, 0.02):
        p = S[k] + amp * rng.standard_normal(S[k].shape); p /= np.linalg.norm(p, axis=1, keepdims=True)
        (r,) = be.relax([p])
        res[f"{k}@{amp}"] = {"E": round(float(be.energy([r])[0]) - Ev, 2), "rms_to_start_deg": round(rms_angle_deg(r, S[k], w), 2)}
print(json.dumps(res, indent=0)); json.dump(res, open(f"chen_stability_{nx}.json", "w"), indent=1)
