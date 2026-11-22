#!/usr/bin/env python3
"""
Size sweep, one grain size per call (MERRILL v2.2.1, magnetite 20 C).

Protocol (fixed before any result was seen; identical for every size):
  1. ground-state search: minimise a fixed candidate set in one MERRILL run
       - the 150 nm reference vortex (core along [111]) mapped onto this mesh by coordinate scaling (P1)
       - the same vortex rotated so that its core lies along [001] and along [110]
       - uniform magnetisation along [111], [110], [001]
       - N_RANDOM random states (default 3)
     energies of all relaxed candidates are evaluated; the lowest is m0.
  2. symmetry-resolved barrier m0 -> -m0: merrill_symdecomp.py (SUBGROUP from the environment).
  3. single-path baseline: MERRILL default (MakeInitialPath) m0 -> -m0, 19 images, same evaluation limit.

Usage: merrill_sizesweep.py <merrill_bin> <mesh.pat> <size_nm> <ref_m0.dat> <ref_mesh.tec> <workdir> [max_evals]
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from merrill_diag import read_tec, rodrigues, tet_geometry, helicity  # noqa: E402
from merrill_runner import P1Interpolator  # noqa: E402

KBT = 1.380649e-23 * 300.0
HEAD = lambda mesh: ["Set: MaxMeshNumber = 1", f"ReadMesh: mesh=1, filename={mesh}",
                     "Set: MaxEnergyEvaluations = 20000", "Material: Type=Magnetite, TempC=20", "ConjugateGradient"]


def rot_a_to_b(a, b):
    a, b = np.asarray(a, float) / np.linalg.norm(a), np.asarray(b, float) / np.linalg.norm(b)
    v = np.cross(a, b)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3)
    return rodrigues(v, float(np.arccos(np.clip(a @ b, -1, 1))))


def run(mbin, wd, name, lines):
    (wd / f"{name}.mscr").write_text("\n".join(lines + ["End"]) + "\n")
    t0 = time.time()
    with open(wd / f"{name}.log", "w") as lf:
        subprocess.run([mbin, f"{name}.mscr"], cwd=wd, stdout=lf, stderr=subprocess.STDOUT, check=True)
    return time.time() - t0


def main():
    mbin, mesh, size, ref_m0, ref_tec, wd = sys.argv[1:7]
    max_evals = sys.argv[7] if len(sys.argv) > 7 else "1000"
    mbin, size, wd = str(Path(mbin).resolve()), float(size), Path(wd)
    wd.mkdir(parents=True, exist_ok=True)
    mname = Path(mesh).name
    shutil.copy(mesh, wd / mname)
    res = {"size_nm": size, "mesh": mname}
    rng = np.random.default_rng(int(size))
    n_rand = int(os.environ.get("N_RANDOM", "3"))

    # mesh topology of this mesh
    run(mbin, wd, "topo", HEAD(mname) + ["Randomize: type=all", "WriteMagnetization: filename=topo"])
    c, t, _ = read_tec(wd / "topo_mult.tec")
    res["n_nodes"], res["n_tets"] = int(len(c)), int(len(t))
    cen = c.mean(0)

    # reference vortex on the reference mesh
    rc, rt, _ = read_tec(ref_tec)
    ra = np.loadtxt(ref_m0)
    assert np.allclose(ra[:, :3], rc, atol=1e-6)
    rm = ra[:, 3:6] / np.linalg.norm(ra[:, 3:6], axis=1, keepdims=True)
    Pref = P1Interpolator(rc, rt)
    rcen = rc.mean(0)
    scale = 150.0 / size
    cands = {}
    for tag, target in [("vortex111", [1, 1, 1]), ("vortex001", [0, 0, 1]), ("vortex110", [1, 1, 0])]:
        R = rot_a_to_b([1, 1, 1], target)
        src = ((c - cen) @ R) * scale + rcen  # R^T applied to positions
        m = (R @ Pref(src, rm).T).T
        cands[tag] = m / np.linalg.norm(m, axis=1, keepdims=True)
    for tag, v in [("uniform111", [1, 1, 1]), ("uniform110", [1, 1, 0]), ("uniform001", [0, 0, 1])]:
        cands[tag] = np.tile(np.asarray(v, float) / np.linalg.norm(v), (len(c), 1))
    for k in range(n_rand):
        m = rng.normal(size=(len(c), 3))
        cands[f"random{k}"] = m / np.linalg.norm(m, axis=1, keepdims=True)
    lines = HEAD(mname)
    for tag, m in cands.items():
        np.savetxt(wd / f"cand_{tag}.dat", np.hstack([c, m]), fmt="%.10e")
        lines += [f"ReadMagnetization: filename=cand_{tag}.dat", "Minimize", f"WriteMagnetization: filename=gs_{tag}"]
    res["gs_search_s"] = round(run(mbin, wd, "gs", lines), 1)
    tags = list(cands)
    el = HEAD(mname) + [f"Set: PathN = {len(tags)}"]
    for i, tag in enumerate(tags, 1):
        el += [f"ReadMagnetization: filename=gs_{tag}.dat", f"MagnetizationToPath: {i}"]
    run(mbin, wd, "gsE", el + ["RenewPath", "PathStructureEnergies: filename=gsE.txt"])
    E = np.array([float(l.split()[-1]) for l in open(wd / "gsE.txt") if l.strip()])
    vol, G = tet_geometry(c, t)
    summ = {}
    for i, tag in enumerate(tags):
        m = np.loadtxt(wd / f"gs_{tag}.dat")[:, 3:6]
        m /= np.linalg.norm(m, axis=1, keepdims=True)
        mv = (vol[:, None] * m[t].mean(1)).sum(0) / vol.sum()
        summ[tag] = {"E_rel_kBT": round(float((E[i] - E.min()) / KBT), 2), "mean_m": round(float(np.linalg.norm(mv)), 3),
                     "mean_m_dir": np.round(mv / max(np.linalg.norm(mv), 1e-12), 3).tolist(),
                     "helicity": round(float(helicity(m, t, vol, G)), 3)}
    res["gs_candidates"] = summ
    best = tags[int(np.argmin(E))]
    res["m0_from"] = best
    res["m0"] = summ[best]
    shutil.copy(wd / f"gs_{best}.dat", wd / "m0.dat")
    print(json.dumps(res, indent=1), flush=True)

    # symmetry-resolved
    t0 = time.time()
    with open(wd / "symdecomp.out", "w") as f:
        subprocess.run([sys.executable, "-I", str(HERE / "merrill_symdecomp.py"), mbin, str(wd / mname), str(wd / "m0.dat"),
                        str(wd / "sym"), max_evals], stdout=f, stderr=subprocess.STDOUT, check=True, env=dict(os.environ))
    sd = json.load(open(wd / "sym" / "symdecomp.json"))
    res["symdecomp"] = {k: sd[k] for k in ["subgroup", "n_orbit_states", "minus_m0_match_rms_deg", "minimax_barrier_m0_to_minus_m0_kBT",
                                           "lowest_escape_barrier_kBT", "total_path_evals", "timing"]}
    res["symdecomp"]["n_classes"] = len(sd["hops"])
    res["symdecomp"]["hops"] = {k: {kk: h[kk] for kk in ["class", "barrier_kBT", "evals", "wall_s"]} for k, h in sd["hops"].items()}
    res["symdecomp"]["wall_s"] = round(time.time() - t0, 1)
    print(json.dumps(res["symdecomp"], indent=1), flush=True)

    # single-path baseline
    env = {k: v for k, v in os.environ.items() if k != "END_DAT"}
    with open(wd / "default.out", "w") as f:
        subprocess.run([sys.executable, "-I", str(HERE / "merrill_runner.py"), mbin, str(wd / mname), str(wd / "m0.dat"),
                        str(wd / "default"), "default", "-", max_evals], stdout=f, stderr=subprocess.STDOUT, check=True, env=env)
    s = json.load(open(wd / "default" / "summary.json"))
    txt = (wd / "default" / "run.log").read_text()
    n = sum(int(x) for x in re.findall(r"CalcPathAction Calls:\s*(\d+)", txt))
    stop = ("failed" if "FAILED TO CONVERGE" in txt else "max_evals" if "MAX Path" in txt
            else "stagnation" if "DeltaS negligible" in txt else "?")
    res["default"] = {"barrier_kBT": s["barrier_kBT"], "evals": n, "wall_s": s.get("wall_s"),
                      "stop": stop, "profile_kBT": s["profile_kBT"]}
    (wd / "sweep.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("gs_candidates",)}, indent=1))


if __name__ == "__main__":
    main()
