"""Edge classes of the Chen, Chang and Williams (2025) state set: 6:5:4 magnetite cuboid, long axis x, short axis z.
States: 2 SD (along x) and 4 SV (core along z, two core signs x two handednesses), as in their Fig. 1.
Group detected from the grid; action exact (cells permute). No energies are computed here."""
import itertools, json, sys
import numpy as np
from scipy.spatial import cKDTree
import os
from _paths import DATA, FIGURES, MERRILL, RUNS  # noqa: F401
os.makedirs(RUNS / "chen", exist_ok=True)
os.chdir(RUNS / "chen")  # inputs and outputs of the cuboid runs live in <RUNS>/chen
from symbarrier import group as grp, network as net
from symbarrier.meshes import analytic_vortex

h = 1.0
L = np.array([12, 10, 8])                      # 6:5:4, cell centres
g = [np.arange(n) - (n - 1) / 2 for n in L]
X = np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3) * h
w = np.full(len(X), 1 / len(X))
tree = cKDTree(X)
interp = lambda pts, m: m[tree.query(pts)[1]]   # exact: group maps cell centres onto cell centres
G0 = grp.detect_point_group(X)
G = grp.with_time_reversal(G0)
act = grp.make_action(X, interp)

sd = np.tile([1.0, 0, 0], (len(X), 1))
sv = analytic_vortex(X, [0, 0, 1], core_radius=1.5, handed=1)
nodes, fam, perm = net.build_nodes([sd, sv], G, act, w)

def label(m, f):
    if f == 0:
        return "SD%+d" % np.sign(m[:, 0].mean())
    rho = np.linalg.norm(X[:, :2], axis=1)
    core = np.sign(m[rho < 1.5, 2].mean())
    circ = np.sign(np.mean(X[:, 0] * m[:, 1] - X[:, 1] * m[:, 0]))
    return "SV(core%+d,circ%+d,hand%+d)" % (core, circ, core * circ)

lab = [label(m, f) for m, f in zip(nodes, fam)]
pairs = list(itertools.combinations(range(len(nodes)), 2))
classes, orbit = net.edge_classes(pairs, perm)

def kind(a, b):
    if fam[a] == fam[b] == 0: return "SD<->SD reversal"
    if fam[a] != fam[b]: return "SD<->SV"
    pa, pb = [tuple(int(s[s.index(k) + len(k):s.index(k) + len(k) + 2]) for k in ("core", "circ", "hand")) for s in (lab[a], lab[b])]
    flips = [n for n, x, y in zip(("core", "circulation", "handedness"), pa, pb) if x != y]
    return "SV<->SV flips " + "+".join(flips)

out = {"G_order": len(G0), "G_x_T": len(G), "nodes": lab,
       "n_pairs": len(pairs), "n_classes": len(classes),
       "classes": [{"rep": [lab[r[0]], lab[r[1]]], "kind": kind(*r), "orbit_size": len(orbit[r]),
                    "pairs": [[lab[a], lab[b]] for a, b in orbit[r]]} for r in sorted(classes)]}
print(json.dumps(out, indent=1))
json.dump(out, open("chen_cuboid_classes.json", "w"), indent=1)
