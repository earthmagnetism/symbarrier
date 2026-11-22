#!/usr/bin/env python3
"""Table 4: tomography-derived grains, all orientations, against Gergov et al. (2025) VesuviusGrainMetrics.csv."""
import csv
import json
import os
from _paths import DATA, FIGURES, MERRILL, RUNS  # noqa: F401

KBT = 1.380649e-23 * 300.0
G = {r["Mesh ID"]: r for r in csv.DictReader(open(f"{DATA}/VesuviusGrainMetrics.csv"))}
rows = []
for g in ["V120", "V116", "V091", "V092"]:
    for run, lab in [("run", "crystal = mesh axes"), ("run_rot1", "random 1"), ("run_rot2", "random 2")]:
        p = f"{RUNS}/natural/{g}/{run}/natural.json"
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        n = d["network"]
        fams = d["distinct_LEMs_mod_T"]
        gs = fams[0]
        kind = "near-uniform" if gs["mean_m"] > 0.95 else "non-uniform"
        mm = n["minimax_m0_to_minus_m0_kBT"]
        sb = d["single_band_default"]["barrier_kBT"]
        best = min(x for x in (mm, sb) if x is not None)
        rows.append({"grain": g, "esvd": d["esvd_nm"], "orient": lab, "gs": f"{kind} ({gs['mean_m']:.2f})",
                     "others": ", ".join(f"{f['E_kBT']:.0f}" for f in fams[1:4]), "classes": n["n_edge_classes"],
                     "minimax": mm, "single": sb, "best": best, "esc": n["escape_barrier_by_family_kBT"]["0"],
                     "new_states": sum(1 for s in n["split_states"] if s["new"]),
                     "gergov_gs": G[g]["Ground State"], "gergov_kbt": round(float(G[g]["Energy Gap (J)"]) / KBT, 0)})
json.dump(rows, open(f"{RUNS}/natural.json", "w"), indent=1)
print("| Grain (ESVD) | Crystal axes | Lowest state ($\\lVert\\langle\\mathbf{m}\\rangle\\rVert$) | Other states found | Classes (NEBs) | Network minimax | Single band | Escape of lowest state | Gergov et al. [20]: ground state, barrier |")
print("|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['grain']} ({r['esvd']:.0f} nm) | {r['orient']} | {r['gs']} | {r['others']} | {r['classes']} | {r['minimax']:.1f} | {r['single']:.1f} | {r['esc']:.1f} | {r['gergov_gs']}, {r['gergov_kbt']:.0f} |")
