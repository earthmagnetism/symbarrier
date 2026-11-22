#!/usr/bin/env python3
"""
MERRILL side of the MERRILL/FD prism alignment: the rate-limiting (~97 deg) hop only.

  prism_align_merrill.py <merrill_bin> <mesh.pat> <ref_m0.dat> <ref_mesh.tec> <workdir> [EXTRA_MATERIAL lines ';'-separated]

1. map the reference m0 (h = 10 nm prism) onto <mesh.pat> by P1 interpolation (identity if same mesh);
2. relax it with the given material (Magnetite 20 C + extra lines, e.g. 'Set: K2 = 0');
3. merrill_symdecomp.py with ONLY_ANGLE=97 and the same material.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from merrill_diag import read_tec  # noqa: E402
from merrill_runner import P1Interpolator  # noqa: E402


def main():
    mbin, mesh, ref_m0, ref_tec, wd = sys.argv[1:6]
    extra = sys.argv[6] if len(sys.argv) > 6 else ""
    mbin, wd = str(Path(mbin).resolve()), Path(wd)
    wd.mkdir(parents=True, exist_ok=True)
    mname = Path(mesh).name
    shutil.copy(mesh, wd / mname)
    head = ["Set: MaxMeshNumber = 1", f"ReadMesh: mesh=1, filename={mname}", "Set: MaxEnergyEvaluations = 20000",
            "Material: Type=Magnetite, TempC=20"] + [l for l in extra.split(";") if l] + ["ConjugateGradient"]
    run = lambda name, lines: subprocess.run([mbin, f"{name}.mscr"], cwd=wd, check=True, stdout=open(wd / f"{name}.log", "w"),
                                            stderr=subprocess.STDOUT) if (wd / f"{name}.mscr").write_text("\n".join(lines + ["End"]) + "\n") else None
    run("topo", head + ["Randomize: type=all", "WriteMagnetization: filename=topo"])
    c, t, _ = read_tec(wd / "topo_mult.tec")
    rc, rt, _ = read_tec(ref_tec)
    ra = np.loadtxt(ref_m0)
    rm = ra[:, 3:6] / np.linalg.norm(ra[:, 3:6], axis=1, keepdims=True)
    m = P1Interpolator(rc, rt)(c, rm)
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    np.savetxt(wd / "m0_init.dat", np.hstack([c, m]), fmt="%.10e")
    run("relax_m0", head + ["ReadMagnetization: filename=m0_init.dat", "Minimize", "WriteMagnetization: filename=m0"])
    env = dict(os.environ, ONLY_ANGLE="97", EXTRA_MATERIAL=extra, SUBGROUP="full")
    with open(wd / "symdecomp.out", "w") as f:
        subprocess.run([sys.executable, "-I", str(HERE / "merrill_symdecomp.py"), mbin, str(wd / mname), str(wd / "m0.dat"), str(wd / "sym")],
                       env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
    d = json.load(open(wd / "sym" / "symdecomp.json"))
    res = {"mesh": mname, "n_nodes": int(len(c)), "extra_material": extra, "n_shape_group": d["n_shape_group"],
           "classes": {k: v["rms_to_m0_deg"] for k, v in d["classes"].items()},
           "hops": {k: {kk: h[kk] for kk in ["class", "barrier_kBT", "evals", "wall_s"]} for k, h in d["hops"].items()}}
    (wd / "align_merrill.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
