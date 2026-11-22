"""Re-assemble every tomography-grain network from all stored MERRILL paths (network bands, the repeated single band
and, for V116/V091, the adaptive runs of Table 5), taking the lowest bound per edge; no new NEB is computed. Writes
network.json and rate_matrix_per_s.csv per run, and a summary table."""
import glob, json, sys
import numpy as np
from _paths import DATA, FIGURES, MERRILL, RUNS  # noqa: F401
from symbarrier.backends.merrill_backend import MerrillBackend
from symbarrier.workflow import run_network
from cache_from_merrill import build

N = f"{RUNS}/natural"
RUNS = [("V092", "run", "V092_mesh.pat", [], []), ("V092_rot1", "run_rot1", "V092_rot1.pat", [], []),
        ("V092_rot2", "run_rot2", "V092_rot2.pat", [], []), ("V117", "run", "V117_mesh.pat", [], []),
        ("V117_rot1", "run_rot1", "V117_rot1.pat", [], []),
        ("V116", "run", "V116_mesh.pat", [f"{RUNS}/V116_explore/out/families.npy:3"], [f"{RUNS}/V116_explore"]),
        ("V091", "run", "V091_mesh.pat", [f"{N}/V091/run/net/split_0_1_10_r.dat"], [f"{RUNS}/V091_explore", f"{RUNS}/V091_warm"])]
summary = []
for name, rd, mesh, extra_fams, extra_dirs in RUNS:
    g = name.split("_")[0]
    net_dir = f"{N}/{g}/{rd}/net"
    be = MerrillBackend(MERRILL, f"{net_dir}/{mesh}", f"{RUNS}/regen/{name}")
    seeds = []
    for f in sorted(glob.glob(f"{net_dir}/seed[0-9].dat")):
        a = np.loadtxt(f); assert np.allclose(a[:, :3], be.coords, atol=1e-6); seeds.append(a[:, 3:6])
    for x in extra_fams:
        if x.endswith(".dat"):
            seeds.append(np.loadtxt(x)[:, 3:6])
        else:
            f, i = x.split(":"); seeds.append(np.load(f)[int(i)])
    seeds = [s / np.linalg.norm(s, axis=1, keepdims=True) for s in seeds]
    pairs = [(f"{d}/finalpath.tec", f"{d}/energies.txt") for d in sorted(glob.glob(f"{net_dir}/hop_*")) if not d.endswith(".out")]
    pairs.append((f"{N}/{g}/{rd}/default/finalpath.tec", f"{N}/{g}/{rd}/default/energies.txt"))
    for d in extra_dirs:
        pairs += [(t, t.replace("_final.tec", "_E.txt")) for t in sorted(glob.glob(f"{d}/hop_*_final.tec"))]
    cache = build(pairs, be.coords)
    r = run_network(be, seeds, outdir=f"{RUNS}/regen/{name}/out", cache=cache, cache_only=True, verbose=False)
    row = {"run": name, "families": len(seeds), "paths": len(cache), "classes": r["n_classes"], "missing": r["missing_classes"],
           "family_energy_kT": r["family_energy_kT"], "reversal_kT": r["minimax_m0_to_minus_m0_kT"], "escape_kT": r["escape_barrier_by_family_kT"]}
    print(json.dumps(row, default=float), flush=True)
    summary.append(row)
json.dump(summary, open(f"{RUNS}/regen/summary.json", "w"), indent=1, default=float)
