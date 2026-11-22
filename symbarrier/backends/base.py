"""Abstract backend: the interface between the symmetry/network layer and a micromagnetic solver."""
import numpy as np

from .. import group as grp


class Backend:
    coords: np.ndarray
    weights: np.ndarray
    kT: float
    tets = None  # tetrahedra for FE backends; None for grids
    name = "backend"

    # --- geometry -------------------------------------------------------------------------------------------
    def point_group(self, proper_only=False, tol=1e-3):
        return grp.detect_point_group(self.coords, self.tets, tol=tol, proper_only=proper_only)

    def interpolator(self):
        raise NotImplementedError

    def action(self):
        return grp.make_action(self.coords, self.interpolator())

    # --- solver ---------------------------------------------------------------------------------------------
    def energy(self, fields):
        raise NotImplementedError

    def relax(self, fields):
        raise NotImplementedError

    def neb(self, a, b, n_images=19, initial=None, tag="neb"):
        raise NotImplementedError

    # --- helpers --------------------------------------------------------------------------------------------
    @staticmethod
    def unit(m):
        m = np.asarray(m, dtype=float)
        return m / np.linalg.norm(m, axis=1, keepdims=True)
