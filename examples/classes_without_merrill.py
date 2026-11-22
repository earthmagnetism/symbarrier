#!/usr/bin/env python3
"""Symmetry bookkeeping without MERRILL: point group, orbit and edge classes of a seed state on a mesh.

    python examples/classes_without_merrill.py                  # built-in 100 nm truncated octahedron, <111> vortex
    python examples/classes_without_merrill.py grain.pat m0.dat  # your mesh and a state (x y z mx my mz per node)

It prints how many NEB calculations the network needs, before any micromagnetic run.
"""
import sys

import numpy as np

from symbarrier import group as grp
from symbarrier import network as net
from symbarrier.geometry import P1Interpolator, node_weights, read_patran, rms_angle_deg
from symbarrier.meshes import analytic_vortex, truncated_octahedron


def main():
    if len(sys.argv) > 2:
        c, t = read_patran(sys.argv[1])
        a = np.loadtxt(sys.argv[2])
        assert np.allclose(a[:, :3], c, atol=1e-6), "state file must list the mesh nodes in the same order"
        m0 = a[:, 3:6] / np.linalg.norm(a[:, 3:6], axis=1, keepdims=True)
    else:
        c, t = truncated_octahedron(100, 2)
        m0 = analytic_vortex(c, [1, 1, 1], core_radius=0.012)
    P = grp.detect_point_group(c, t)
    G = grp.with_time_reversal(P)
    w = node_weights(c, t)
    act = grp.make_action(c, P1Interpolator(c, t))
    nodes, fam, perm = net.build_nodes([m0], G, act, w)
    neg = int(np.argmin([rms_angle_deg(-m0, s, w) for s in nodes]))
    classes = net.stabiliser_classes(len(nodes), 0, neg, perm)
    print(f"nodes in mesh: {len(c)}; point group |G| = {len(P)}; with time reversal {len(G)}")
    print(f"orbit of m0: {len(nodes)} states; -m0 is state {neg} ({rms_angle_deg(-m0, nodes[neg], w):.2f} deg)")
    print(f"edge classes from m0 (one NEB each): {len(classes)}")
    for k, members in classes.items():
        print(f"  class {k}: {len(members)} equivalent targets, rms angle to m0 {rms_angle_deg(nodes[k], m0, w):.1f} deg")
    print(f"unreduced pairwise network: {len(nodes) * (len(nodes) - 1) // 2} NEBs")


if __name__ == "__main__":
    main()
