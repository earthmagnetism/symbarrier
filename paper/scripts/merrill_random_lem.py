#!/usr/bin/env python3
"""
Community-style baseline: enumerate LEMs by repeated energy minimisation from random initial states.

One MERRILL run performs n cycles of `Randomize: type=all` + `Minimize` + write. Relaxed states are
de-duplicated (rms < 5 deg), their energies evaluated, and each is compared with the cubic orbit of m0.
Reports wall time, distinct LEMs found, and how many seeds were needed to cover the orbit of m0.

Usage: merrill_random_lem.py <merrill_bin> <mesh.pat> <m0.dat> <workdir> <n_seeds>
"""
import itertools
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from merrill_diag import read_tec, ang_deg, tet_geometry, node_weights  # noqa: E402
from merrill_runner import P1Interpolator  # noqa: E402

KBT = 1.380649e-23 * 300.0


def main():
    mbin, mesh, m0file, wd, n = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4]), int(sys.argv[5])
    wd.mkdir(parents=True, exist_ok=True)
    shutil.copy(mesh, wd / Path(mesh).name)
    shutil.copy(m0file, wd / "m0.dat")
    head = ["Set: MaxMeshNumber = 1", f"ReadMesh: mesh=1, filename={Path(mesh).name}",
            "Set: MaxEnergyEvaluations = 20000", "Material: Type=Magnetite, TempC=20", "ConjugateGradient"]
    lines = list(head)
    for k in range(n):
        lines += ["Randomize: type=all", "Minimize", f"WriteMagnetization: filename=rand{k:03d}"]
    (wd / "rand.mscr").write_text("\n".join(lines + ["End"]) + "\n")
    t0 = time.time()
    with open(wd / "rand.log", "w") as lf:
        subprocess.run([str(Path(mbin).resolve()), "rand.mscr"], cwd=wd, stdout=lf, stderr=subprocess.STDOUT, check=True)
    wall = time.time() - t0
    # energies of m0 and all relaxed states (path of n+1 structures)
    el = list(head) + [f"Set: PathN = {n + 1}", "ReadMagnetization: filename=m0.dat", "MagnetizationToPath: 1"]
    for k in range(n):
        el += [f"ReadMagnetization: filename=rand{k:03d}.dat", f"MagnetizationToPath: {k + 2}"]
    (wd / "E.mscr").write_text("\n".join(el + ["RenewPath", "PathStructureEnergies: filename=E.txt", "End"]) + "\n")
    subprocess.run([str(Path(mbin).resolve()), "E.mscr"], cwd=wd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True)
    e = np.array([float(l.split()[-1]) for l in open(wd / "E.txt") if l.strip()])
    erel = (e - e[0]) / KBT
    c, t, _ = read_tec(wd / "rand000_mult.tec")
    vol, G = tet_geometry(c, t)
    w = node_weights(c, t, vol)
    d = lambda x, y: float(np.sqrt(np.sum(w * ang_deg(x, y) ** 2)))
    L = lambda f: (lambda a: a[:, 3:6] / np.linalg.norm(a[:, 3:6], axis=1, keepdims=True))(np.loadtxt(f))
    m0 = L(wd / "m0.dat")
    P = P1Interpolator(c, t)
    cen = c.mean(0)
    orbit = []
    for perm in itertools.permutations(range(3)):
        for sg in itertools.product([1, -1], repeat=3):
            R = np.zeros((3, 3))
            for i, j in enumerate(perm):
                R[i, j] = sg[i]
            if np.linalg.det(R) < 0:
                continue
            mr = (R @ P((c - cen) @ R + cen, m0).T).T
            mr /= np.linalg.norm(mr, axis=1, keepdims=True)
            if all(d(mr, o) > 5 for o in orbit):
                orbit.append(mr)
    distinct, label, orbit_hit = [], [], {}
    for k in range(n):
        m = L(wd / f"rand{k:03d}.dat")
        j = next((i for i, s in enumerate(distinct) if d(m, s["m"]) < 5.0), None)
        if j is None:
            oi = next((i for i, o in enumerate(orbit) if d(m, o) < 5.0), None)
            distinct.append({"m": m, "E_rel_m0_kBT": round(float(erel[k + 1]), 2), "orbit_index": oi, "first_seed": k, "count": 1})
            j = len(distinct) - 1
        else:
            distinct[j]["count"] += 1
        label.append(j)
        oi = distinct[j]["orbit_index"]
        if oi is not None and oi not in orbit_hit:
            orbit_hit[oi] = k
    res = {"n_seeds": n, "wall_s": round(wall, 1), "wall_per_seed_s": round(wall / n, 2), "orbit_size": len(orbit),
           "n_distinct_LEMs": len(distinct), "orbit_states_found": len(orbit_hit),
           "seeds_to_cover_orbit": (max(orbit_hit.values()) + 1) if len(orbit_hit) == len(orbit) else None,
           "distinct": [{k: v for k, v in s.items() if k != "m"} for s in sorted(distinct, key=lambda s: s["E_rel_m0_kBT"])]}
    (wd / "random_lem.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "distinct"}, indent=1))
    for s in res["distinct"]:
        print(s)


if __name__ == "__main__":
    main()
