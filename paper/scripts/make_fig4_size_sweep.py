#!/usr/bin/env python3
"""Fig. 4 (best bound = min over the network and the single band): reversal barrier of the lowest state vs grain size (truncated octahedron, MERRILL v2.2.1, 20 C).
Reads <RUNS>/sweep/S*/sweep.json; writes sweep.json and fig4_size_sweep.png."""
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _paths import DATA, FIGURES, MERRILL, RUNS  # noqa: F401

BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#e8693a", "#222222", "#5c5c58", "#e6e6e3"
rows = []
for f in sorted(glob.glob(f"{RUNS}/sweep/S*/sweep.json")):
    r = json.load(open(f))
    sd, d = r["symdecomp"], r["default"]
    dirn = [abs(x) for x in r["m0"]["mean_m_dir"]]
    kind = "⟨100⟩" if max(dirn) > 0.95 else ("⟨111⟩" if min(dirn) > 0.5 else "⟨110⟩")
    rows.append({"size_nm": r["size_nm"], "n_nodes": r["n_nodes"], "m0_from": r["m0_from"], "m0_core_axis": kind,
                 "m0_mean_m": r["m0"]["mean_m"], "n_orbit_proper": sd["n_orbit_states"], "n_classes": sd["n_classes"],
                 "minus_m0_match_rms_deg": sd["minus_m0_match_rms_deg"],
                 "sym_barrier_kBT": sd["minimax_barrier_m0_to_minus_m0_kBT"], "sym_escape_kBT": sd["lowest_escape_barrier_kBT"],
                 "sym_wall_s": sd["wall_s"], "sym_evals": sd["total_path_evals"],
                 "hops": {h["class"]: h["barrier_kBT"] for h in sd["hops"].values()},
                 "default_barrier_kBT": d["barrier_kBT"], "default_stop": d["stop"], "default_evals": d["evals"],
                 "default_wall_s": d["wall_s"], "gs_search_s": r["gs_search_s"],
                 "ratio_default_over_sym": round(d["barrier_kBT"] / sd["minimax_barrier_m0_to_minus_m0_kBT"], 2)})
rows.sort(key=lambda x: x["size_nm"])


plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#b5b5ae", "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=300)
S = [r["size_nm"] for r in rows]
ax.plot(S, [r["default_barrier_kBT"] for r in rows], "-o", color=ORANGE, lw=2, ms=5, label="single band $\\mathbf{m}_0\\to-\\mathbf{m}_0$ (MERRILL default)")
ax.plot(S, [min(r["sym_barrier_kBT"], r["default_barrier_kBT"]) for r in rows], "-o", color=BLUE, lw=2, ms=5, label="best upper bound (all computed paths)")
ax.set_yscale("log")
ax.set_xlabel("grain size (nm, square-face separation)")
ax.set_ylabel("reversal barrier of lowest state ($k_\\mathrm{B}T_{300}$)")
ax.grid(True, which="major", color=GRID, lw=0.6)
ax.spines[["top", "right"]].set_visible(False)
for r in rows:
    ax.annotate(f"{r['m0_core_axis']}", (r["size_nm"], min(r["sym_barrier_kBT"], r["default_barrier_kBT"])), textcoords="offset points", xytext=(0, -13),
                ha="center", fontsize=7, color=INK2)
ax.axhline(58.7, color=INK2, lw=0.8, ls="--")
ax.text(252, 64, "1 Gyr ($\\tau_0=10^{-9}$ s)", ha="right", va="bottom", fontsize=7, color=INK2)
ax.set_ylim(25, 1.2e4)
ax.legend(frameon=False, loc="upper left", fontsize=7.5)
fig.tight_layout()
fig.savefig(f"{FIGURES}/fig4_size_sweep.png")
for r in rows:
    print({k: v for k, v in r.items() if k != "hops"})
