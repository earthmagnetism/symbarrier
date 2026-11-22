"""Solver backends.

The symmetry and network layer (`group`, `network`, `workflow`) only sees a *field space* and four operations.
A backend provides

  coords      (n, 3) sample points of the field (mesh nodes or cell centres), metres
  weights     (n,)  volume weights, normalised to 1
  kT          energy unit (J)
  point_group(proper_only)       elements R of O_h (crystal frame) that map the sample points onto themselves
  interpolator()                 callable (pts, values) -> values at pts, used by the group action
  energy(fields)                 total energies in units of kT
  relax(fields)                  local minimisation
  neb(a, b, n_images, initial)   minimum-energy path; returns a dict with at least
                                 profile_kT (energies of the images, kT, absolute), path (list of fields), info

Fields are numpy arrays of unit vectors, shape (n, 3), in the backend's sample order. Any micromagnetic code that
can evaluate an energy and run a path method fits behind this interface; two are provided:

  MerrillBackend   finite elements, MERRILL as an external program (files handled internally)
  MagnumNPBackend  finite differences on a regular grid with an optional mask, magnum.np (PyTorch), GNEB-FIRE
"""
from .base import Backend

__all__ = ["Backend", "MerrillBackend", "MagnumNPBackend"]


def __getattr__(name):  # optional dependencies are imported on first use
    if name == "MerrillBackend":
        from .merrill_backend import MerrillBackend
        return MerrillBackend
    if name == "MagnumNPBackend":
        from .magnumnp_backend import MagnumNPBackend
        return MagnumNPBackend
    raise AttributeError(name)
