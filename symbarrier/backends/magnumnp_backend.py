"""magnum.np (finite differences, PyTorch) behind the array interface.

The grain is a regular grid of N = (nx, ny, nz) cells of size dx with an optional boolean mask; cells outside the
mask carry Ms = A = 0 and a fixed moment, and every field is masked. Energy: demagnetising + exchange + cubic
anisotropy K1 (sum m_i^2 m_j^2) + K2 (m_x^2 m_y^2 m_z^2), crystal axes along the grid axes. The sample points of the
field space are the centres of the active cells (C order), so every O_h element that maps the active-cell set onto
itself acts by an exact permutation of cells (nearest-cell lookup)."""
import math

import numpy as np
import torch
from scipy.spatial import cKDTree

from .base import Backend
from .gneb import geodesic_band, gneb_fire

MU0 = 4e-7 * math.pi
KB = 1.380649e-23
MAGNETITE_FD = {"Ms": 4.80e5, "A": 1.33e-11, "K1": -1.33e4, "K2": 0.0}
MAGNETITE_MERRILL_20C = {"Ms": 482476.91085597320, "A": 1.3444090643665902e-11, "K1": -13039.710203218889,
                         "K2": -3154.0197698514348}


class _Cubic:
    def __init__(self, K1, K2, Ms, mask4, dV):
        self.K1, self.K2, self.mask4, self.dV = K1, K2, mask4, dV
        self.p1, self.p2 = -2 * K1 / (MU0 * Ms), -2 * K2 / (MU0 * Ms)

    def h(self, st):
        m = st.m
        x2, y2, z2 = m[..., 0] ** 2, m[..., 1] ** 2, m[..., 2] ** 2
        h = self.p1 * m * torch.stack([y2 + z2, z2 + x2, x2 + y2], -1)
        if self.K2:
            h = h + self.p2 * m * torch.stack([y2 * z2, z2 * x2, x2 * y2], -1)
        return h * self.mask4

    def E(self, st):
        m = st.m
        x2, y2, z2 = m[..., 0] ** 2, m[..., 1] ** 2, m[..., 2] ** 2
        w = self.mask4[..., 0]
        return self.dV * torch.sum(w * (self.K1 * (x2 * y2 + y2 * z2 + z2 * x2) + self.K2 * x2 * y2 * z2))


class _Masked:
    def __init__(self, t, mask4):
        self.t, self.mask4 = t, mask4

    def h(self, st):
        return self.t.h(st) * self.mask4

    def E(self, st):
        return self.t.E(st)


class MagnumNPBackend(Backend):
    name = "magnum.np"

    def __init__(self, N, dx, mask=None, material=None, temperature_K=300.0, gneb_kw=None, relax_kw=None):
        from magnumnp import DemagField, ExchangeField, Mesh, MinimizerBB, State
        import logging
        logging.getLogger("magnum.np").setLevel(logging.WARNING)
        torch.set_default_dtype(torch.float64)
        mat = dict(MAGNETITE_FD if material is None else material)
        self.N, self.dx = tuple(int(n) for n in N), tuple(float(d) for d in dx)
        self.mask = torch.ones(self.N, dtype=torch.bool) if mask is None else torch.as_tensor(mask, dtype=torch.bool)
        self.idx = torch.nonzero(self.mask, as_tuple=True)
        m4 = self.mask.double()[..., None]
        self._m4 = m4
        self._out = torch.tensor([0.0, 0.0, 1.0]) * (1 - m4)
        self.st = State(Mesh(self.N, self.dx))
        self.st.material = {"Ms": mat["Ms"] * m4, "A": mat["A"] * m4}
        self.terms = [_Masked(DemagField(), m4), _Masked(ExchangeField(), m4),
                      _Cubic(mat["K1"], mat.get("K2", 0.0), mat["Ms"], m4, float(np.prod(self.dx)))]
        self.minimizer = MinimizerBB(self.terms)
        self.kT = KB * temperature_K
        self.gneb_kw = dict(max_iter=3000, tol_f=500.0, ci_gate=5e3, dt_max=6e-5) | (gneb_kw or {})
        self.relax_kw = dict(relax_kw or {})  # passed to MinimizerBB.minimize (maxiter, dm_tol)
        grid = [(np.arange(n) + 0.5) * d - 0.5 * n * d for n, d in zip(self.N, self.dx)]
        X, Y, Z = np.meshgrid(*grid, indexing="ij")
        msk = self.mask.numpy()
        self.coords = np.stack([X[msk], Y[msk], Z[msk]], 1)
        self.weights = np.full(len(self.coords), 1.0 / len(self.coords))
        self.material = mat

    # --- field space ----------------------------------------------------------------------------------------
    def interpolator(self):
        tree = cKDTree(self.coords)
        return lambda pts, vals: vals[tree.query(pts)[1]]

    def to_grid(self, f):
        g = self._out.clone()
        g[self.idx] = torch.as_tensor(np.asarray(f), dtype=torch.float64)
        return g

    def from_grid(self, g):
        return g[self.idx].numpy().copy()

    def _fix(self, g):
        g = g * self._m4 + self._out
        return g / g.norm(dim=-1, keepdim=True)

    def _field(self, g):
        self.st.m = g
        return sum(t.h(self.st) for t in self.terms), float(sum(t.E(self.st) for t in self.terms))

    # --- solver ---------------------------------------------------------------------------------------------
    def energy(self, fields):
        return np.array([self._field(self._fix(self.to_grid(f)))[1] for f in fields]) / self.kT

    def energy_parts(self, f):
        self.st.m = self._fix(self.to_grid(f))
        return {type(t.t if isinstance(t, _Masked) else t).__name__: float(t.E(self.st)) / self.kT for t in self.terms}

    def relax(self, fields):
        out = []
        for f in fields:
            self.st.m = self._fix(self.to_grid(f))
            self.minimizer.minimize(self.st, **self.relax_kw)
            out.append(self.from_grid(self._fix(self.st.m)))
        return out

    def neb(self, a, b, n_images=19, initial=None, tag="neb", verbose=False):
        if initial is None:
            band = geodesic_band(self._fix(self.to_grid(a)), self._fix(self.to_grid(b)), n_images)
        else:
            band = [self._fix(self.to_grid(f)) for f in initial]
        band, Es, info = gneb_fire(self._field, band, mask=self.mask, verbose=verbose, label=tag, **self.gneb_kw)
        return {"profile_kT": [e / self.kT for e in Es], "path": [self.from_grid(self._fix(m)) for m in band],
                "info": info}
