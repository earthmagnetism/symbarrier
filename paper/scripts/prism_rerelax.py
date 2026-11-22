"""Relax the interior band minima of the prism (prism_fd_split.py) further and compare them with the known states."""
import json, sys
import numpy as np
from _paths import DATA, FIGURES, MERRILL, RUNS  # noqa: F401
from symbarrier.backends.magnumnp_backend import MagnumNPBackend, MAGNETITE_MERRILL_20C
from symbarrier import group as grp, network as net
from symbarrier.geometry import rms_angle_deg
be = MagnumNPBackend((24, 24, 40), (5e-9,) * 3, material=MAGNETITE_MERRILL_20C, relax_kw=dict(maxiter=50000))
m0 = np.load(f"{RUNS}/prism_m0.npy")
nodes, fam, perm = net.build_nodes([m0], grp.with_time_reversal(be.point_group()), be.action(), be.weights)
E0 = be.energy([m0])[0]
out = {}
for i in (4, 9):
    m = np.load(f"{RUNS}/prism_split_0_1_{i}.npy")
    r = be.relax([m])[0]
    np.save(f"{RUNS}/prism_state_{i}.npy", r)
    out[i] = {"E_before": round(float(be.energy([m])[0] - E0), 2), "E_after": round(float(be.energy([r])[0] - E0), 2),
              "moved_deg": round(rms_angle_deg(m, r, be.weights), 2), "rms_to_orbit_deg": round(min(rms_angle_deg(r, n, be.weights) for n in nodes), 1),
              "mean_m": np.round(r.mean(0), 3).tolist()}
print(json.dumps(out, indent=1))
json.dump(out, open(f"{RUNS}/prism_rerelax.json", "w"), indent=1)
