#!/usr/bin/env python3
"""
Post-hoc: compare the random-seed LEMs of merrill_random_lem.py with the FULL O_h orbit of m0
(proper + improper elements; improper act on the axial vector m with det(R)).

Usage: random_lem_fullOh.py <workdir-of-merrill_random_lem> [n_seeds]
Writes <workdir>/random_lem_fullOh.json
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from merrill_diag import read_tec, ang_deg, tet_geometry, node_weights  # noqa: E402
from merrill_runner import P1Interpolator  # noqa: E402


def main():
    wd = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    c, t, _ = read_tec(wd / "rand000_mult.tec")
    vol, _ = tet_geometry(c, t)
    w = node_weights(c, t, vol)
    d = lambda x, y: float(np.sqrt(np.sum(w * ang_deg(x, y) ** 2)))
    L = lambda f: (lambda a: a[:, 3:6] / np.linalg.norm(a[:, 3:6], axis=1, keepdims=True))(np.loadtxt(f))
    m0 = L(wd / "m0.dat")
    P = P1Interpolator(c, t)
    cen = c.mean(0)
    orbit, proper = [], []
    for perm in itertools.permutations(range(3)):
        for sg in itertools.product([1, -1], repeat=3):
            R = np.zeros((3, 3))
            for i, j in enumerate(perm):
                R[i, j] = sg[i]
            det = np.linalg.det(R)
            mr = det * (R @ P((c - cen) @ R + cen, m0).T).T
            mr /= np.linalg.norm(mr, axis=1, keepdims=True)
            if all(d(mr, o) > 5 for o in orbit):
                orbit.append(mr)
                proper.append(bool(det > 0))
    distinct = []
    for k in range(n):
        m = L(wd / f"rand{k:03d}.dat")
        j = next((i for i, s in enumerate(distinct) if d(m, s) < 5.0), None)
        if j is None:
            distinct.append(m)
    rms, hit = [], set()
    for m in distinct:
        dd = [d(m, o) for o in orbit]
        i = int(np.argmin(dd))
        rms.append(round(dd[i], 3))
        hit.add(i)
    res = {"orbit_size_Oh": len(orbit), "orbit_size_proper": int(sum(proper)), "n_distinct_LEMs": len(distinct),
           "all_in_orbit_5deg": bool(max(rms) < 5.0), "max_rms_to_orbit_deg": max(rms),
           "orbit_states_found": len(hit), "orbit_states_missed": len(orbit) - len(hit),
           "found_only_via_improper": int(sum(1 for i in hit if not proper[i])), "rms_each_deg": rms}
    (wd / "random_lem_fullOh.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "rms_each_deg"}))


if __name__ == "__main__":
    main()
