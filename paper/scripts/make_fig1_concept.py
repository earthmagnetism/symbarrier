"""Fig. 1: (a) the quotient idea on the eight <111> uniform states; (b) algorithm and software structure."""
import itertools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from _paths import DATA, FIGURES, MERRILL, RUNS  # noqa: F401

INK, INK2, EDGE = "#222222", "#5c5c58", "#b5b5ae"
CL = {1: "#2a6fdb", 2: "#e07b39", 3: "#3a9a5b"}  # number of differing signs -> class colour
FILL = {"in": "#eef2f7", "core": "#e3edf9", "ext": "#f6ece4", "cert": "#e6f2ea"}
fig = plt.figure(figsize=(10.5, 4.6), dpi=250)
a = fig.add_axes([0.0, 0.02, 0.33, 0.93]); a.axis("off"); a.set_aspect("equal")
V = [np.array(s) for s in itertools.product([1, -1], repeat=3)]
P = lambda v: np.array([v[0] + 0.42 * v[1], v[2] + 0.30 * v[1]])
for u, w in itertools.combinations(range(8), 2):
    k = int(np.sum(V[u] != V[w]))
    p, q = P(V[u]), P(V[w])
    a.plot(*zip(p, q), color=CL[k], lw=0.9, alpha=0.45, zorder=1, ls="-" if k < 3 else (0, (2, 2)))
m0 = 0
for k, w in ((1, 4), (2, 6), (3, 7)):
    p, q = P(V[m0]), P(V[w])
    a.plot(*zip(p, q), color=CL[k], lw=3.2, zorder=2, solid_capstyle="round", ls="-" if k < 3 else (0, (1.5, 1)))
for i, v in enumerate(V):
    x, y = P(v)
    a.scatter([x], [y], s=150, color="white", edgecolor=INK, zorder=3, lw=1.1)
    lab = "".join("+" if c > 0 else "−" for c in v)
    a.text(x + (0.13 if v[0] > 0 else -0.13), y + 0.16, lab, fontsize=7, ha="left" if v[0] > 0 else "right", color=INK2)
a.text(*(P(V[m0]) + [0.16, -0.24]), r"$\mathbf{m}_0$", fontsize=9, color=INK)
a.text(-1.75, 2.05, "(a) quotient network", fontsize=10, color=INK, ha="left")
a.text(-1.75, 1.75, "8 uniform $\\langle 111\\rangle$ states, 28 pairs, $O_h\\times\\{1,T\\}$", fontsize=7.5, color=INK2)
for j, (k, txt) in enumerate(((1, "70.5°: 12 edges, 1 class"), (2, "109.5°: 12 edges, 1 class"), (3, "180°: 4 edges, 1 class"))):
    y = -1.75 - 0.27 * j
    a.plot([-1.7, -1.35], [y, y], color=CL[k], lw=3, ls="-" if k < 3 else (0, (1.5, 1)))
    a.text(-1.25, y, txt, fontsize=7.5, va="center", color=INK)
a.text(-1.75, -2.65, "bold: the one path computed per class; its upper bound\nholds on every edge of the class orbit (thin)", fontsize=7, color=INK2, va="top")
a.set_xlim(-1.8, 1.9); a.set_ylim(-3.0, 2.2)

b = fig.add_axes([0.35, 0.02, 0.65, 0.93]); b.axis("off"); b.set_xlim(-3, 100); b.set_ylim(2, 66)


def box(x, y, w, h, title, sub, kind, mod=None):
    b.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=FILL[kind], ec=EDGE, lw=0.8))
    b.text(x + w / 2, y + h - 2.4, title, ha="center", va="top", fontsize=7.6, color=INK, weight="bold")
    b.text(x + w / 2, y + h - 6.2, sub, ha="center", va="top", fontsize=6.3, color=INK2, linespacing=1.25)
    if mod:
        b.text(x + w - 0.6, y + 0.8, mod, ha="right", va="bottom", fontsize=5.6, color=INK2, family="monospace")


def arrow(x0, y0, x1, y1, ls="-"):
    b.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, lw=0.9, color=INK2, linestyle=ls, shrinkA=1, shrinkB=1))


b.text(-2, 64.5, "(b) algorithm and software", fontsize=10, color=INK)
box(1, 43, 21, 17, "Field space + seeds", "sample points, weights,\ninterpolation (backend);\nrelaxed LEMs", "in", "backends")
box(27, 43, 21, 17, "Group", "elements of O_h mapping\nthe sample points onto\nthemselves, × T", "core", "group")
box(53, 43, 21, 17, "Orbits + table", "|G| actions per seed;\nπ_g from the group\nmultiplication table", "core", "network")
box(78, 43, 21, 17, "Edge classes", "short-edge graph;\norbits on unordered\npairs", "core", "network")
box(78, 22, 21, 17, "One path per class", "cache hit → reuse;\nelse relax ends + NEB\nvia backend.neb", "ext", "workflow")
box(53, 22, 21, 17, "Network", "bound on every edge of\neach class orbit; minimax,\nescape barriers, rates", "core", "network")
box(27, 22, 21, 17, "Certification", "path maxima = upper\nbounds; lowest of repeats;\npath transfer; index", "cert", "workflow")
box(1, 22, 21, 17, "Outputs", "network.json, rate matrix,\nevery path with its stop\nreason and cost", "in", "workflow")
arrow(22.6, 51.5, 26.4, 51.5); arrow(48.6, 51.5, 52.4, 51.5); arrow(74.6, 51.5, 77.4, 51.5); arrow(88.5, 42.4, 88.5, 39.6)
arrow(77.4, 30.5, 74.6, 30.5); arrow(52.4, 30.5, 48.6, 30.5); arrow(26.4, 30.5, 22.6, 30.5)
b.plot([88.5, 88.5, -1.8, -1.8], [21.4, 15.0, 15.0, 51.5], color=INK2, lw=0.9, linestyle=(0, (3, 2)))
arrow(-1.8, 51.5, 0.4, 51.5)
b.text(43, 16.0, "Algorithm 2: interior band minima relaxed; new states become new families; computed paths reused",
       ha="center", va="bottom", fontsize=6.3, color=INK2)
b.add_patch(FancyBboxPatch((52, 3.5), 47, 8, boxstyle="round,pad=0.3,rounding_size=1", fc="white", ec=EDGE, lw=0.8, ls=(0, (2, 2))))
b.text(75.5, 9.6, "solver interface: energy · relax · neb", ha="center", fontsize=6.4, color=INK, family="monospace")
b.text(75.5, 5.4, "MerrillBackend (FE, external)   MagnumNPBackend (FD, GNEB-FIRE)", ha="center", fontsize=5.9, color=INK2)
arrow(88.5, 21.4, 88.5, 12.2, ls=(0, (2, 2)))
b.text(24, 7.5, "bookkeeping: seconds; cost = one path per class", ha="center", fontsize=6.4, color=INK2, style="italic")
fig.savefig(f"{FIGURES}/fig1_concept_workflow.png", bbox_inches="tight")
