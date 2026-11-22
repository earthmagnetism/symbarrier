"""Rebuild a path cache (end fields + highest image energy) from MERRILL path outputs of earlier runs."""
import glob, sys
import numpy as np
from _paths import DATA, FIGURES, MERRILL, RUNS  # noqa: F401
from symbarrier.geometry import read_tecplot

KT = 1.380649e-23 * 300.0


def build(pairs, coords):
    cache = []
    for tec, etxt in pairs:
        c, _, path = read_tecplot(tec)
        assert np.allclose(c, coords, atol=1e-6), tec
        e = np.array([float(l.split()[-1]) for l in open(etxt) if l.strip()]) / KT
        if np.all(np.isfinite(e)):
            cache.append({"start": path[0], "end": path[-1], "saddle_abs_kT": float(e.max()), "src": tec})
    return cache


def v091_pairs():
    old = [(f"{d}/finalpath.tec", f"{d}/energies.txt") for d in sorted(glob.glob(f"{RUNS}/natural/V091/run/net/hop_*")) if not d.endswith(".out")]
    new = [(t, t.replace("_final.tec", "_E.txt")) for t in sorted(glob.glob(f"{RUNS}/V091_explore/hop_*_final.tec"))]
    return old + new
