#!/usr/bin/env python3
"""Collect the benchmark numbers (Tables 1-2, Fig. 3, Supplementary S1-S3) into numbers.json, all from run outputs."""
import json
import re
from pathlib import Path

import numpy as np
from _paths import DATA, FIGURES, MERRILL, RUNS  # noqa: F401

R = RUNS / "benchmarks"
out = {}


def wall(fname):
    d = {}
    for l in open(R / fname):
        p = l.split()
        if len(p) == 2:
            d[p[0]] = round(float(p[1]), 1)
    return d


w1, w3 = wall("bm_wall.txt"), wall("bm3_wall.txt")


def merrill_run(d):
    s = json.load(open(R / d / "summary.json"))
    t = (R / d / "run.log").read_text()
    stop = ("failed" if "FAILED TO CONVERGE" in t else "max_evals" if "MAX Path" in t else "stagnation" if "DeltaS negligible" in t else "?")
    n = sum(int(x) for x in re.findall(r"CalcPathAction Calls:\s*(\d+)", t))
    return {"barrier_kBT": s["barrier_kBT"], "stop": stop, "path_evals": n, "wall_s": w1[d], "profile_kBT": s["profile_kBT"]}


for g in ["TO", "CO"]:
    blk = {m: merrill_run(f"BM_{g}_{m}") for m in ["cov", "default", "staged"]}
    sd_p = json.load(open(R / f"BM_{g}_symdecomp" / "symdecomp.json"))
    blk["symdecomp_proper"] = {"barrier_kBT": sd_p["minimax_barrier_m0_to_minus_m0_kBT"], "path_evals": sd_p["total_path_evals"],
                               "wall_s": w1.get(f"BM_{g}_symdecomp", sd_p["timing"]["total_s"]), "n_classes": len(sd_p["hops"]),
                               "n_orbit": sd_p["n_orbit_states"]}
    sd = json.load(open(R / f"BM3_{g}_symdecomp" / "symdecomp.json"))
    blk["symdecomp_full"] = {"barrier_kBT": sd["minimax_barrier_m0_to_minus_m0_kBT"], "path_evals": sd["total_path_evals"],
                             "wall_s": w3[f"BM3_{g}_symdecomp"], "timing": sd["timing"], "n_classes": len(sd["hops"]),
                             "n_orbit": sd["n_orbit_states"], "lowest_escape_kBT": sd["lowest_escape_barrier_kBT"],
                             "hops": {k: {kk: h[kk] for kk in ["class", "barrier_kBT", "evals", "wall_s"]} for k, h in sd["hops"].items()},
                             "mean_hop_wall_s": round(float(np.mean([h["wall_s"] for h in sd["hops"].values()])), 1)}
    rl = json.load(open(R / f"RAND_{g}" / "random_lem.json"))
    blk["random_lem"] = {k: rl[k] for k in ["n_seeds", "wall_s", "wall_per_seed_s", "n_distinct_LEMs"]}
    fo = json.load(open(R / f"RAND_{g}" / "random_lem_fullOh.json"))
    blk["random_lem"]["vs_full_Oh_orbit"] = {k: v for k, v in fo.items() if k != "rms_each_deg"}
    n_or = sd["n_orbit_states"]
    pairs = n_or * (n_or - 1) // 2
    blk["network_estimate"] = {"n_pairs": pairs, "est_pairwise_wall_s": round(pairs * blk["symdecomp_full"]["mean_hop_wall_s"]),
                               "est_total_with_lem_search_s": round(pairs * blk["symdecomp_full"]["mean_hop_wall_s"] + rl["wall_s"]),
                               "speedup_vs_symdecomp_full": round((pairs * blk["symdecomp_full"]["mean_hop_wall_s"] + rl["wall_s"]) / w3[f"BM3_{g}_symdecomp"], 1)}
    out[f"merrill_{g}"] = blk
out["merrill_TO"]["staged_earlier_run_v221_kBT"] = 1465.93
# FD benchmark
fd = {}
for name, d in [("TO_band112", "FD_TO_band112"), ("PR_band001", "FD_PR_band001"), ("PR_band100", "FD_PR_band100")]:
    b = json.load(open(R / d / "band.json"))
    fd[name] = {k: b[k] for k in ["conv", "iters", "barrier_kBT", "wall_s", "image_iterations"]}
for name, d in [("TO_symdecomp", "FD_TO_symdecomp"), ("PR_symdecomp", "FD_PR_symdecomp")]:
    s = json.load(open(R / d / "symdecomp.json"))
    fd[name] = {"barrier_kBT": s["minimax_barrier_m0_to_minus_m0_kBT"], "image_iterations": s["total_image_iterations"],
                "wall_s": s["timing"]["total_s"], "n_classes": len(s["hops"])}
p3 = R / "FD3_CU_symdecomp" / "symdecomp.json"
if p3.exists():
    s = json.load(open(p3))
    fd["CU_from_C_nearest3"] = {"hops": {k: {kk: h[kk] for kk in ["class", "rms_to_m0_deg", "conv", "iters", "barrier_from_m0_kBT", "E_end_rel_m0_kBT", "wall_s"]} for k, h in s["hops"].items()},
                                "n_orbit": s["n_orbit_states"], "wall_s": s["timing"]["total_s"]}
out["fd"] = fd
# Hessian and mode push (cube)
hs = {}
for f in ["hess/A_min", "hess/cube_AtoAp", "hess/cube_m0toA", "hess/prism_B2hop90", "hess/truncoct_band112", "hess/truncoct_hop70",
          "hess/cube_AAp_symbroken", "hess/cube_m0A_symbroken", "modepush/h_AAp_push0.3", "modepush/h_m0A_push0.3",
          "modepush/h_AAp_push-0.3", "modepush/h_m0A_push-0.3"]:
    d = json.load(open(R / (f + ".json")))
    hs[f] = {"E_rel_first_kBT": d["E_rel_first_kBT"], "eigs": d["lowest_eigenvalues_kBT_per_rad2"], "n_negative": d["n_negative"],
             "overlap": d.get("overlap_lowest_mode_with_band_tangent")}
out["hessian"] = hs
mp = {}
for f in ["AAp_push0.3", "AAp_push-0.3", "m0A_push0.3", "m0A_push-0.3"]:
    r = json.load(open(R / "modepush" / f"result_{f}.json"))
    mp[f] = {"conv": r["conv"], "iters": r["iters"], "barrier_kBT": r["barrier_kBT"], "max_rel_m0_kBT": max(r["profile_rel_m0_kBT"]), "profile_rel_m0_kBT": r["profile_rel_m0_kBT"]}
for f in ["AAp_s1", "AAp_s2", "m0A_s1", "m0A_s2"]:
    r = json.load(open(R / "symbreak" / f"result_{f}.json"))
    mp["noise_" + f] = {"barrier_kBT": r["barrier_kBT"], "max_rel_m0_kBT": max(r["profile_rel_m0_kBT"])}
out["cube_modepush"] = mp
out["cube_C_state"] = {"E_rel_m0_kBT": -50.12, "core_tilt_deg": 14.2, "core_axis": [0.174, -0.174, 0.969], "A_E_rel_m0_kBT": -12.32}
# values from the earlier benchmark runs (numbers_base.json)
v2 = json.load(open(f"{RUNS}/numbers_base.json"))
out["earlier_runs"] = {"meshconv_hop70": {t: v2["trunc_oct_150_meshconv_hop70"][t]["hop_default"]["barrier_kBT"] for t in ["h12.5", "h9.38", "h7.5"]},
                       "meshconv_extrap": v2["trunc_oct_150_meshconv_hop70"]["extrapolation"],
                       "band_meshconv_v186p": v2["trunc_oct_150_v186p"]["mesh_conv_barrier"],
                       "fd_trunc_oct": {k: v2["fd_trunc_oct_150"][k]["barrier_kBT"] for k in ["band112", "hop70"]}}
(RUNS / "numbers.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: (v if k not in ("hessian", "cube_modepush") else "...") for k, v in out.items()}, indent=1)[:4000])
