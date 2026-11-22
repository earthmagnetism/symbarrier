"""MERRILL (finite elements) behind the array interface. MERRILL runs as an external program; this class writes and
reads its files in a working directory so that callers only handle arrays."""
import itertools

import numpy as np

from ..geometry import P1Interpolator, node_weights, read_tecplot
from ..merrill import Merrill
from .base import Backend


class MerrillBackend(Backend):
    name = "MERRILL"

    def __init__(self, binary, mesh, workdir, material="Type=Magnetite, TempC=20", extra=(), temperature_K=300.0,
                 max_evals=1000):
        self.m = Merrill(binary, mesh, workdir, material=material, extra=extra, temperature_K=temperature_K)
        self.coords, self.tets = self.m.topology()
        self.weights = node_weights(self.coords, self.tets)
        self.kT = self.m.kT
        self.max_evals = max_evals
        self._n = itertools.count()

    @property
    def workdir(self):
        return self.m.wd

    def interpolator(self):
        return P1Interpolator(self.coords, self.tets)

    def _write(self, fields, stem):
        k = next(self._n)
        return [self.m.write_field(f"{stem}{k}_{i}", self.coords, f) for i, f in enumerate(fields)]

    def read(self, fname):
        return self.m.read_field(fname)[1]

    def energy(self, fields):
        files = self._write(fields, "e")
        return self.m.energies(files, tag=f"E{next(self._n)}")

    def relax(self, fields):
        files = self._write(fields, "r")
        k = next(self._n)
        out = self.m.minimize(files, [f"rel{k}_{i}" for i in range(len(files))], tag=f"relax{k}")
        return [self.read(f) for f in out]

    def neb(self, a, b, n_images=19, initial=None, tag="neb"):
        tag = f"{tag}_{next(self._n)}"
        fa, fb = self._write([a, b], "ends")
        init = None if initial is None else self._write(initial, "init")
        h = self.m.neb(fa, fb, tag, n_images=n_images if initial is None else len(initial), max_evals=self.max_evals,
                       initial=init)
        _, _, path = read_tecplot(self.m.wd / f"{tag}_final.tec")
        return {"profile_kT": h["E_abs_kT"], "path": path,
                "info": {"stop": h["stop"], "evals": h["evals"], "solver": "MERRILL PathMinimize"}}
