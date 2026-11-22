"""V091 warm start: the three LEM families plus the state recovered by band splitting in the first run, with every
earlier path (first run and the cold explore run) in the cache; only classes involving the new family are computed."""
import json, sys, time
import numpy as np
from _paths import DATA, FIGURES, MERRILL, RUNS  # noqa: F401
from symbarrier.backends.merrill_backend import MerrillBackend
from symbarrier.workflow import explore
from cache_from_merrill import build, v091_pairs
src = f"{RUNS}/natural/V091/run/net"
be = MerrillBackend(MERRILL, f"{src}/V091_mesh.pat", f"{RUNS}/V091_warm")
seeds = []
for f in [f"{src}/seed0.dat", f"{src}/seed1.dat", f"{src}/seed2.dat", f"{src}/split_0_1_10_r.dat"]:
    a = np.loadtxt(f)
    assert np.allclose(a[:, :3], be.coords, atol=1e-6)
    seeds.append(a[:, 3:6] / np.linalg.norm(a[:, 3:6], axis=1, keepdims=True))
cache = build(v091_pairs(), be.coords)
print("cached paths", len(cache), flush=True)
t = time.time()
r = explore(be, seeds, outdir=f"{RUNS}/V091_warm/out", cache=cache)
json.dump({"hops": r["hops"], "history": r["history"], "wall_s": time.time() - t}, open(f"{RUNS}/V091_warm/out/summary.json", "w"), indent=1, default=float)
print("wall", time.time() - t)
