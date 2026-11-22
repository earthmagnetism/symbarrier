#!/usr/bin/env python3
"""
Full workflow on a grain without point symmetry (e.g. a tomography-derived mesh), MERRILL v2.2.1, 20 C.

Protocol (fixed in advance, identical for every grain):
  1. LEM search: uniform states along the four <111> and three <100> axes plus N_RANDOM random states,
     each minimised; relaxed states de-duplicated modulo time reversal (rms < 5 deg).
  2. Families: the MAX_FAM lowest distinct states (default 4) become seed families; the lowest is m0.
  3. merrill_network.py on these seeds: automatic group detection (expected G = {e}, so only time
     reversal T remains), short-edge graph, one NEB per edge class, minimax m0 -> -m0, escape barriers,
     rate matrix.
  4. Single-band baseline: MERRILL default NEB m0 -> -m0 with the same evaluation limit.

Usage: merrill_natural.py <merrill_bin> <mesh.pat> <workdir>
Env:   N_RANDOM (default 8), MAX_FAM (default 4), plus merrill_network.py variables
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
from merrill_diag import read_tec, ang_deg, tet_geometry, node_weights  # noqa: E402

KBT = 1.380649e-23 * 300.0


def head(mesh):
    return ["Set: MaxMeshNumber = 1", f"ReadMesh: mesh=1, filename={mesh}", "Set: MaxEnergyEvaluations = 20000",
            "Material: Type=Magnetite, TempC=20", "ConjugateGradient"]


def merrill(mbin, wd, name, lines):
    (wd / f"{name}.mscr").write_text("\n".join(lines + ["End"]) + "\n")
    t0 = time.time()
    with open(wd / f"{name}.log", "w") as lf:
        subprocess.run([mbin, f"{name}.mscr"], cwd=wd, stdout=lf, stderr=subprocess.STDOUT, check=True)
    return time.time() - t0


def main():
    mbin, mesh, wd = str(Path(sys.argv[1]).resolve()), sys.argv[2], Path(sys.argv[3])
    n_rand, max_fam = int(os.environ.get("N_RANDOM", "8")), int(os.environ.get("MAX_FAM", "4"))
    wd.mkdir(parents=True, exist_ok=True)
    mname = Path(mesh).name
    shutil.copy(mesh, wd / mname)
    res = {"mesh": mname}
    merrill(mbin, wd, "topo", head(mname) + ["Randomize: type=all", "WriteMagnetization: filename=topo"])
    c, t, _ = read_tec(wd / "topo_mult.tec")
    vol, _ = tet_geometry(c, t)
    w = node_weights(c, t, vol)
    d = lambda x, y: float(np.sqrt(np.sum(w * ang_deg(x, y) ** 2)))
    res.update(n_nodes=int(len(c)), n_tets=int(len(t)), esvd_nm=round(float((6 * vol.sum() / np.pi) ** (1 / 3) * 1e3), 1))
    rng = np.random.default_rng(12345)
    cands = {}
    for v in ([1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]):
        cands["u" + "".join("p" if x > 0 else "m" if x < 0 else "0" for x in v)] = np.tile(np.array(v, float) / np.linalg.norm(v), (len(c), 1))
    for k in range(n_rand):
        m = rng.normal(size=(len(c), 3))
        cands[f"random{k}"] = m / np.linalg.norm(m, axis=1, keepdims=True)
    lines = head(mname)
    for tag, m in cands.items():
        np.savetxt(wd / f"cand_{tag}.dat", np.hstack([c, m]), fmt="%.10e")
        lines += [f"ReadMagnetization: filename=cand_{tag}.dat", "Minimize", f"WriteMagnetization: filename=lem_{tag}"]
    res["lem_search_s"] = round(merrill(mbin, wd, "lems", lines), 1)
    tags = list(cands)
    el = head(mname) + [f"Set: PathN = {len(tags)}"]
    for i, tag in enumerate(tags, 1):
        el += [f"ReadMagnetization: filename=lem_{tag}.dat", f"MagnetizationToPath: {i}"]
    merrill(mbin, wd, "lemE", el + ["RenewPath", "PathStructureEnergies: filename=lemE.txt"])
    E = np.array([float(l.split()[-1]) for l in open(wd / "lemE.txt") if l.strip()])
    E = (E - E.min()) / KBT
    L = lambda f: (lambda a: a[:, 3:6] / np.linalg.norm(a[:, 3:6], axis=1, keepdims=True))(np.loadtxt(f))
    fams = []
    for i in np.argsort(E):
        m = L(wd / f"lem_{tags[i]}.dat")
        j = next((k for k, f in enumerate(fams) if min(d(m, f["m"]), d(m, -f["m"])) < 5.0), None)
        if j is None:
            mv = (vol[:, None] * m[t].mean(1)).sum(0) / vol.sum()
            fams.append({"m": m, "tag": tags[i], "E_kBT": round(float(E[i]), 2), "count": 1,
                         "mean_m": round(float(np.linalg.norm(mv)), 3), "mean_m_dir": np.round(mv / max(np.linalg.norm(mv), 1e-12), 3).tolist()})
        else:
            fams[j]["count"] += 1
    res["distinct_LEMs_mod_T"] = [{k: v for k, v in f.items() if k != "m"} for f in fams]
    seeds = [str(wd / f"lem_{f['tag']}.dat") for f in fams[:max_fam]]
    print(json.dumps(res, indent=1), flush=True)
    if os.environ.get("LEM_ONLY") == "1":  # stop after the LEM search (protocol comparisons)
        (wd / "lems.json").write_text(json.dumps(res, indent=1, default=float))
        return

    t0 = time.time()
    with open(wd / "network.out", "w") as f:
        subprocess.run([sys.executable, "-I", str(HERE / "merrill_network.py"), mbin, str(wd / mname), str(wd / "net")] + seeds,
                       stdout=f, stderr=subprocess.STDOUT, check=True)
    net = json.load(open(wd / "net" / "network.json"))
    res["network"] = {k: v for k, v in net.items() if k not in ("hops",)}
    res["network"]["hops"] = net["hops"]
    res["network"]["wall_s"] = round(time.time() - t0, 1)

    env = {k: v for k, v in os.environ.items() if k != "END_DAT"}
    with open(wd / "default.out", "w") as f:
        subprocess.run([sys.executable, "-I", str(HERE / "merrill_runner.py"), mbin, str(wd / mname), seeds[0],
                        str(wd / "default"), "default", "-", "1000"], stdout=f, stderr=subprocess.STDOUT, check=True, env=env)
    s = json.load(open(wd / "default" / "summary.json"))
    txt = (wd / "default" / "run.log").read_text()
    res["single_band_default"] = {"barrier_kBT": s["barrier_kBT"], "wall_s": s.get("wall_s"),
                                  "evals": sum(int(x) for x in re.findall(r"CalcPathAction Calls:\s*(\d+)", txt)),
                                  "stop": ("failed" if "FAILED TO CONVERGE" in txt else "max_evals" if "MAX Path" in txt
                                           else "stagnation" if "DeltaS negligible" in txt else "?"),
                                  "profile_kBT": s["profile_kBT"]}
    (wd / "natural.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k not in ("network",)}, indent=1, default=float))
    print(json.dumps({k: v for k, v in res["network"].items() if k != "hops"}, indent=1, default=float))


if __name__ == "__main__":
    main()
