"""Symmetry group of a grain and its action on magnetisation fields.

The working group is G x {1, T}: G is the set of elements of O_h (the point group of cubic anisotropy in the
crystal frame) that map the mesh node set onto itself, and T is time reversal m -> -m, exact in zero field.
An element g = (R, t) acts on an axial vector field as

    (g . m)(r) = t det(R) R m(R^-1 (r - c) + c),

with c the centroid of the nodes. Fields are evaluated at R^-1 r by P1 interpolation, so the action is exact on
meshes that are symmetric to round-off and approximate otherwise.
"""
import itertools

import numpy as np
from scipy.spatial import cKDTree


def oh_elements(proper_only=False):
    """The 48 signed permutation matrices (O_h); with proper_only the 24 rotations."""
    out = []
    for perm in itertools.permutations(range(3)):
        for sg in itertools.product([1, -1], repeat=3):
            R = np.zeros((3, 3))
            for i, j in enumerate(perm):
                R[i, j] = sg[i]
            if proper_only and np.linalg.det(R) < 0:
                continue
            out.append(R)
    return out


def detect_point_group(coords, tets=None, tol=1e-3, proper_only=False):
    """Elements of O_h that map the node set onto itself, to a tolerance of `tol` times the shortest mesh
    edge (or of the shortest nearest-neighbour distance if no tets are given)."""
    c = coords - coords.mean(0)
    tree = cKDTree(c)
    if tets is not None:
        e = np.concatenate([tets[:, [i, j]] for i in range(4) for j in range(i + 1, 4)])
        hmin = float(np.min(np.linalg.norm(coords[e[:, 1]] - coords[e[:, 0]], axis=1)))
    else:
        hmin = float(tree.query(c, k=2)[0][:, 1].min())
    return [R for R in oh_elements(proper_only) if tree.query(c @ R.T)[0].max() < tol * hmin]


def with_time_reversal(point_group):
    """G x {1, T} as a list of (R, t)."""
    return [(R, t) for R in point_group for t in (1, -1)]


def compose(g, h):
    return (g[0] @ h[0], g[1] * h[1])


def index_of(G, g):
    return next(i for i, (Q, t) in enumerate(G) if t == g[1] and np.allclose(Q, g[0]))


def make_action(coords, interp):
    """Return act(g, m): the field g . m on the mesh nodes (unit vectors)."""
    cen = coords.mean(0)

    def act(g, m):
        R, t = g
        mr = t * np.linalg.det(R) * (R @ interp((coords - cen) @ R + cen, m).T).T
        return mr / np.linalg.norm(mr, axis=1, keepdims=True)

    return act
