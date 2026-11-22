"""Recompute one prism class (FD backend, 3000 iterations) and relax its interior band minima; report energies and
distances of the relaxed states to the orbit of m0."""
import json, sys
import numpy as np
from _paths import DATA, FIGURES, MERRILL, RUNS  # noqa: F401
from symbarrier.backends.magnumnp_backend import MagnumNPBackend, MAGNETITE_MERRILL_20C
from symbarrier import group as grp, network as net
from symbarrier.geometry import rms_angle_deg
b = int(sys.argv[1])
be = MagnumNPBackend((24, 24, 40), (5e-9,) * 3, material=MAGNETITE_MERRILL_20C, gneb_kw=dict(max_iter=3000))
m0 = np.load(f"{RUNS}/prism_m0.npy")
G = grp.with_time_reversal(be.point_group())
act = be.action()
nodes, fam, perm = net.build_nodes([m0], G, act, be.weights)
h = be.neb(nodes[0], nodes[b], n_images=19)
p = np.array(h["profile_kT"]); p -= p[0]
idx = [i for i in range(1, 18) if p[i] < p[i - 1] - 2 and p[i] < p[i + 1] - 2]
rel = be.relax([h["path"][i] for i in idx])
E0 = be.energy([m0])[0]
out = {"class": f"0-{b}", "barrier_kT": round(float(p.max()), 2), "info": h["info"], "profile_kT": np.round(p, 2).tolist(),
       "max_neighbour_rms_deg": round(max(rms_angle_deg(h["path"][i], h["path"][i + 1], be.weights) for i in range(18)), 1), "minima": []}
for i, m in zip(idx, rel):
    e = float(be.energy([m])[0] - E0)
    d = min(rms_angle_deg(m, n, be.weights) for n in nodes)
    mm = m.mean(0)
    out["minima"].append({"image": i, "band_E_kT": round(float(p[i]), 2), "relaxed_E_kT": round(e, 2), "finite": bool(np.isfinite(e)),
                          "rms_to_orbit_deg": round(d, 1), "mean_m": np.round(mm, 3).tolist()})
    np.save(f"{RUNS}/prism_split_0_{b}_{i}.npy", m)
json.dump(out, open(f"{RUNS}/prism_split_0_{b}.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "profile_kT"}, indent=1))
