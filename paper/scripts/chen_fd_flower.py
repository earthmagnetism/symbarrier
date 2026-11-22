"""Flower sector of the 6:5:4 cuboid at 105 nm (Supplementary S8): symmetric relaxation and a perturbed restart."""
import sys, numpy as np
import os
from _paths import DATA, FIGURES, MERRILL, RUNS  # noqa: F401
os.makedirs(RUNS / "chen", exist_ok=True)
os.chdir(RUNS / "chen")  # inputs and outputs of the cuboid runs live in <RUNS>/chen
from symbarrier.backends.magnumnp_backend import MagnumNPBackend, MAGNETITE_MERRILL_20C
from symbarrier import network as net, group as grp
be = MagnumNPBackend((24, 20, 16), (105e-9 / 24,) * 3, material=MAGNETITE_MERRILL_20C)
sd_tw, sv = np.load("chen_seeds_105.npy")          # from chen_fd_relax.py (flower seed tilted by 0.05, 0.03)
(sym,) = be.relax([np.tile([1.0, 0, 0], (len(be.coords), 1))])
rng = np.random.default_rng(0); p = sym + 0.05 * rng.standard_normal(sym.shape); p /= np.linalg.norm(p, axis=1, keepdims=True)
(pert,) = be.relax([p])
G = grp.with_time_reversal(be.point_group()); act = be.action(); w = be.weights
E = np.asarray(be.energy([sym, sd_tw, pert, sv]))
for name, m, e in zip(["uniform start", "tilted start", "perturbed symmetric flower"], [sym, sd_tw, pert], E[:3] - E[3]):
    print(name, "orbit", len(net.build_nodes([m], G, act, w)[0]), "E - E_vortex %.1f" % e)
