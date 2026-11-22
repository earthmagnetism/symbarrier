#!/usr/bin/env python3
"""Markdown Table 3 from sweep.json (+ full-O_h results if present)."""
import json
import os
from _paths import DATA, FIGURES, MERRILL, RUNS  # noqa: F401

rows = json.load(open(f"{RUNS}/sweep.json"))
H = {80: 10.0, 100: 8.33, 120: 10.0, 150: 9.38, 180: 9.0, 200: 10.0, 250: 8.93}
full = {}
for r in rows:
    f = f"{RUNS}/sweep/S{int(r['size_nm']):03d}/symfull/symdecomp.json"
    if os.path.exists(f):
        d = json.load(open(f))
        full[r["size_nm"]] = d
out = ["| Size (nm) | Nodes, $h$ (nm) | Lowest state | Symmetry-resolved barrier | Single band (default) | Ratio | Wall time: resolved / single (s) |",
       "|---|---|---|---|---|---|---|"]
for r in rows:
    s = r["size_nm"]
    out.append(f"| {s:.0f} | {r['n_nodes']}, {H[int(s)]:.2f} | {r['m0_core_axis']} vortex, $\\lVert\\langle\\mathbf{{m}}\\rangle\\rVert$ = {r['m0_mean_m']:.2f} | "
               f"**{r['sym_barrier_kBT']:.1f}** | {r['default_barrier_kBT']:.1f} | {r['ratio_default_over_sym']:.2f} | {r['sym_wall_s']:.0f} / {r['default_wall_s']:.0f} |")
print("\n".join(out))
if full:
    print("\nfull O_h:")
    for s, d in full.items():
        hops = sorted((h["class"], h["barrier_kBT"]) for h in d["hops"].values())
        print(s, d["n_orbit_states"], len(d["hops"]), d["minimax_barrier_m0_to_minus_m0_kBT"], d["timing"]["total_s"], hops)
