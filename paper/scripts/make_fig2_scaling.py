"""Fig. 2: bookkeeping cost against mesh size, and path counts per grain (from scaling.json)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from _paths import DATA, FIGURES, MERRILL, RUNS  # noqa: F401
rows = json.load(open(f"{RUNS}/scaling.json"))
fe = [r for r in rows if r["mesh"].startswith("truncated octahedron") and r["mesh"].endswith("nm")]
fd = [r for r in rows if "grid" in r["mesh"]]
C1, C2, INK, MUTED = "#2a6fdb", "#e07b39", "#222222", "#888888"
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
a = ax[0]
for grp, c, lab in ((fe, C1, "FE, truncated octahedron, |G|=48"), (fd, C2, "FD grid, prism, |G|=16")):
    x = [r["n_mesh_nodes"] for r in grp]; y = [r["time_s"]["total"] for r in grp]
    a.plot(x, y, "-o", color=c, lw=2, ms=6, label=lab)
xs = np.array([500, 1e5]); a.plot(xs, 1e-3 * xs, ":", color=MUTED, lw=1); a.text(4.5e4, 25, "∝ N", color=MUTED, fontsize=8)
a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("sample points N (nodes or cells)"); a.set_ylabel("bookkeeping time (s, one core)")
a.legend(frameon=False, fontsize=8, loc="upper left"); a.set_title("(a) cost of the symmetry layer", fontsize=10, loc="left")
for s in ("top", "right"): a.spines[s].set_visible(False)
a.grid(alpha=0.25, lw=0.5)
b = ax[1]
sel = [("truncated octahedron 150 nm", "TO, $O_h$"), ("truncated octahedron 150 nm, proper only", "TO, $O$"),
       ("prism 120x120x200 nm (FE)", "prism, $D_{4h}$"), ("truncated octahedron 150 nm, rotated", "TO rotated, $C_i$")]
R = {r["mesh"]: r for r in rows}
x = np.arange(len(sel)); w = 0.26
for k, (key, col, lab) in enumerate((("all_pairs", "#c9c9c9", "all pairs"), ("short_edges", "#8fb4ea", "short-edge graph"), ("classes", C1, "classes of the short-edge graph\n(paths computed)"))):
    v = [R[s][key] for s, _ in sel]
    bars = b.bar(x + (k - 1) * w, v, w * 0.92, color=col, label=lab)
    for xi, vi in zip(x + (k - 1) * w, v):
        b.text(xi, vi + 1.5, str(vi), ha="center", fontsize=7, color=INK)
b.set_xticks(x, [l for _, l in sel], fontsize=8); b.set_ylabel("number of transitions")
b.legend(frameon=False, fontsize=8); b.set_title("(b) transitions and paths, ⟨111⟩ vortex seed", fontsize=10, loc="left")
for s in ("top", "right"): b.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(f"{FIGURES}/fig2_scaling.png", dpi=200)
