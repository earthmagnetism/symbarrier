"""Tests that run without a micromagnetic solver: group detection, orbits and edge classes, network algebra,
the grid backend's exact permutation action (needs magnum.np), and the workflow with a toy solver."""
import numpy as np
from scipy.spatial.transform import Rotation

from symbarrier import group as grp
from symbarrier import network as net
from symbarrier.geometry import P1Interpolator, node_weights, rms_angle_deg
from symbarrier.meshes import analytic_vortex, box, truncated_octahedron


def _orbit(c, t, m0, proper_only=False):
    w = node_weights(c, t)
    G = grp.with_time_reversal(grp.detect_point_group(c, t, proper_only=proper_only))
    act = grp.make_action(c, P1Interpolator(c, t))
    nodes, fam, perm = net.build_nodes([m0], G, act, w)
    neg = int(np.argmin([rms_angle_deg(-m0, s, w) for s in nodes]))
    return G, nodes, perm, neg


def test_point_group_detection():
    c, t = truncated_octahedron(100, 2)
    assert len(grp.detect_point_group(c, t)) == 48
    assert len(grp.detect_point_group(c, t, proper_only=True)) == 24
    cb, tb = box(120, 120, 200, 20)
    assert len(grp.detect_point_group(cb, tb)) == 16  # D4h
    R = Rotation.random(random_state=1).as_matrix()
    Gr = grp.detect_point_group(c @ R.T, t)  # rotated relative to the crystal axes
    assert len(Gr) == 2  # only E and the inversion survive: a centrosymmetric shape keeps C_i for any orientation
    assert any(np.allclose(Q, -np.eye(3)) for Q in Gr)


def test_orbit_and_classes_uniform_111():
    c, t = truncated_octahedron(100, 2)
    m0 = np.tile(np.ones(3) / np.sqrt(3), (len(c), 1))
    G, nodes, perm, neg = _orbit(c, t, m0)
    assert len(nodes) == 8  # the eight <111> directions
    classes = net.stabiliser_classes(len(nodes), 0, neg, perm)
    assert sorted(len(v) for v in classes.values()) == [3, 3]  # 70.5 deg and 109.5 deg neighbours


def test_orbit_and_classes_vortex_111():
    c, t = truncated_octahedron(100, 2)
    m0 = analytic_vortex(c, [1, 1, 1], core_radius=0.012)
    G, nodes, perm, neg = _orbit(c, t, m0)
    assert len(nodes) == 16  # 8 core directions x 2 handedness
    classes = net.stabiliser_classes(len(nodes), 0, neg, perm)
    assert len(classes) == 6
    Gp, nodes_p, perm_p, neg_p = _orbit(c, t, m0, proper_only=True)
    assert len(nodes_p) == 8  # proper rotations (with T) keep the handedness


def test_minimax_and_rates():
    E = np.array([0.0, 0.0, 5.0])
    orbit = {(0, 2): [(0, 2)], (1, 2): [(1, 2)], (0, 1): [(0, 1)]}
    saddle = {(0, 2): 10.0, (1, 2): 12.0, (0, 1): 20.0}
    adj = net.assemble(orbit, saddle)
    best = net.minimax(adj, 0, E)
    assert best[1] == 12.0  # 0 -> 2 -> 1 beats the direct edge at 20
    K = net.rate_matrix(adj, E, f0=1.0)
    assert np.isclose(K[0, 2], np.exp(-10.0)) and np.isclose(K[2, 0], np.exp(-5.0))


def test_grid_backend_symmetry():
    """Grid field space: D4h detected from the active cells, exact permutation action, vortex orbit."""
    import pytest
    pytest.importorskip("magnumnp")
    from symbarrier.backends.magnumnp_backend import MagnumNPBackend
    be = MagnumNPBackend((6, 6, 10), (5e-9,) * 3)
    P = be.point_group()
    assert len(P) == 16
    G = grp.with_time_reversal(P)
    act = be.action()
    m0 = analytic_vortex(be.coords, [1, 0, 0], core_radius=0.3 * np.ptp(be.coords[:, 0]))
    for g in G:  # the action is a permutation of cells: g^-1 . (g . m) = m exactly
        R, t = g
        back = act((R.T, t), act(g, m0))
        assert np.allclose(back, m0)
    nodes, fam, perm = net.build_nodes([m0], G, act, be.weights)
    assert len(nodes) == 8  # transverse vortex in a D4h prism: 4 core directions x 2 handedness
    assert sorted(len(set(p)) for p in perm) == [8] * len(G)


class _ToyBackend:
    """Solver stand-in: flat energy, path maximum = rms angle between the ends (deterministic)."""
    name = "toy"
    tets = None

    def __init__(self):
        c, t = truncated_octahedron(100, 2)
        self.coords, self.tets, self.weights, self.kT = c, t, node_weights(c, t), 1.0
        self.calls = 0

    def point_group(self, proper_only=False, tol=1e-3):
        return grp.detect_point_group(self.coords, self.tets, tol=tol, proper_only=proper_only)

    def action(self):
        return grp.make_action(self.coords, P1Interpolator(self.coords, self.tets))

    def energy(self, fields):
        return np.zeros(len(fields))

    def relax(self, fields):
        return list(fields)

    def neb(self, a, b, n_images=19, initial=None, tag=""):
        self.calls += 1
        s = rms_angle_deg(a, b, self.weights)
        return {"profile_kT": [0.0, s, 0.0], "path": [a, 0.5 * (a + b), b], "info": {}}


def test_workflow_cache_takes_lowest_bound():
    from symbarrier.workflow import run_network
    be = _ToyBackend()
    m0 = analytic_vortex(be.coords, [1, 1, 1], core_radius=0.012)
    cache = []
    r1 = run_network(be, [m0], cache=cache, verbose=False)
    assert be.calls == r1["n_classes"] == r1["n_new_paths"]
    # an extra, lower path on the direct reversal edge (e.g. a repeated band) must lower the minimax
    cache.append({"start": m0, "end": -m0, "saddle_abs_kT": 1.0})
    r2 = run_network(be, [m0], cache=cache, verbose=False)
    assert r2["n_new_paths"] == 0 and be.calls == r1["n_classes"]
    assert r2["minimax_m0_to_minus_m0_kT"] == 1.0 <= r1["minimax_m0_to_minus_m0_kT"]


def test_stored_paths_survive_new_families():
    """Adding families changes the K-nearest graph; stored paths must still enter the edge set."""
    from symbarrier.workflow import run_network
    be = _ToyBackend()
    m0 = analytic_vortex(be.coords, [1, 1, 1], core_radius=0.012)
    cache = []
    r1 = run_network(be, [m0], cache=cache, verbose=False, K=4, K_cross=2)
    extra = [analytic_vortex(be.coords, [1, 0, 0], core_radius=0.012), analytic_vortex(be.coords, [1, 1, 0], core_radius=0.012)]
    r2 = run_network(be, [m0] + extra, cache=cache, verbose=False, K=1, K_cross=1)
    assert r2["minimax_m0_to_minus_m0_kT"] <= r1["minimax_m0_to_minus_m0_kT"] + 1e-9
    e2 = {tuple(e) for e in r2["edges"]}
    assert all(tuple(map(int, k.split("-"))) in e2 for k in r1["hops"])  # family-0 nodes keep their indices
