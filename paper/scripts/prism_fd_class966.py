"""Single class 0-6 (96.6 deg) of the FD prism network through the backend API (parallel to the full run)."""
import json, sys, time
import numpy as np
from _paths import DATA, FIGURES, MERRILL, RUNS  # noqa: F401
from symbarrier.backends.magnumnp_backend import MagnumNPBackend, MAGNETITE_MERRILL_20C
from symbarrier import group as grp, network as net
be = MagnumNPBackend((24, 24, 40), (5e-9,) * 3, material=MAGNETITE_MERRILL_20C, gneb_kw=dict(max_iter=8000))
m0 = np.load(f"{RUNS}/prism_m0.npy")
G = grp.with_time_reversal(be.point_group())
nodes, fam, perm = net.build_nodes([m0], G, be.action(), be.weights)
t = time.time()
h = be.neb(nodes[0], nodes[6], n_images=19)
p = np.array(h["profile_kT"]); p = p - p[0]
out = {"barrier_kT": round(float(p.max()), 2), "end_rel_kT": round(float(p[-1]), 2), "info": h["info"], "profile_kT": np.round(p, 2).tolist(), "wall_s": round(time.time() - t, 1)}
json.dump(out, open(f"{RUNS}/prism_fd_class966.json", "w"), indent=1)
print(out)
