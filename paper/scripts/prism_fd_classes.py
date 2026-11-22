"""Bookkeeping only: FD prism (120x120x200 nm, 5 nm), B2 ground state, MERRILL constants + K2."""
import json, sys, time
import numpy as np, torch
from _paths import DATA, FIGURES, MERRILL, RUNS  # noqa: F401
from symbarrier.backends.magnumnp_backend import MagnumNPBackend, MAGNETITE_MERRILL_20C
from symbarrier import group as grp, network as net
from symbarrier.geometry import rms_angle_deg
be = MagnumNPBackend((24, 24, 40), (5e-9,) * 3, material=MAGNETITE_MERRILL_20C)
P = be.point_group(); G = grp.with_time_reversal(P)
m0 = be.from_grid(torch.tensor(np.load(f"{RUNS}/benchmarks/fd_c2/B2_ground.npy").astype(np.float64)))
m0 = be.relax([m0])[0]
np.save(f"{RUNS}/prism_m0.npy", m0)
t = time.time()
nodes, fam, perm = net.build_nodes([m0], G, be.action(), be.weights)
D = net.distance_matrix(nodes, be.weights)
neg = int(np.argmin([rms_angle_deg(-nodes[0], s, be.weights) for s in nodes]))
edges = net.short_edge_graph(D, fam, 4, 2, always=[(0, neg)])
cl, orb = net.edge_classes(edges, perm)
print(len(P), len(nodes), len(edges), len(cl), {r: round(D[r], 1) for r in cl}, "s", round(time.time() - t, 2))
print("E kT", be.energy([m0]))
