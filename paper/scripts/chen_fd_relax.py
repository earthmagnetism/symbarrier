"""Relax the Chen, Chang and Williams (2025) seeds on the FD backend: 6:5:4 cuboid, MERRILL 20 C constants + K2."""
import json, sys, time
import numpy as np
import os
from _paths import DATA, FIGURES, MERRILL, RUNS  # noqa: F401
os.makedirs(RUNS / "chen", exist_ok=True)
os.chdir(RUNS / "chen")  # inputs and outputs of the cuboid runs live in <RUNS>/chen
from symbarrier.backends.magnumnp_backend import MagnumNPBackend, MAGNETITE_MERRILL_20C
from symbarrier.meshes import analytic_vortex
Lnm = float(sys.argv[1]); N = (24, 20, 16); dx = Lnm * 1e-9 / 24
be = MagnumNPBackend(N, (dx,) * 3, material=MAGNETITE_MERRILL_20C)
X = be.coords
sd = np.tile([1.0, 0.05, 0.03], (len(X), 1)); sd /= np.linalg.norm(sd, axis=1, keepdims=True)
sv = analytic_vortex(X, [0.02, 0.01, 1], core_radius=0.15 * (X.max(0) - X.min(0))[2], handed=1)
t = time.time(); r = be.relax([sd, sv]); dt = time.time() - t
E = np.asarray(be.energy(r))
mm = [np.asarray(m).mean(0).round(3).tolist() for m in r]
print(Lnm, "nm  dx %.2f nm  relax %.0f s" % (dx * 1e9, dt), "E_kT", (E - E.min()).round(2).tolist(), "<m>", mm)
np.save(f"chen_seeds_{int(Lnm)}.npy", np.stack(r))
