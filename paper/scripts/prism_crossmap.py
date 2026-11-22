#!/usr/bin/env python3
"""
Cross-map a converged FD band into MERRILL (prism alignment).

  prism_crossmap.py <merrill_bin> <mesh.pat> <band.npy> <cell_nm> <workdir> [EXTRA_MATERIAL]

1. trilinear interpolation of every FD image (cell centres, box centred at 0) onto the MERRILL nodes;
2. relax the two mapped endpoints in MERRILL (they become MERRILL minima);
3. MERRILL energies of the path [relaxed end A, mapped interior images, relaxed end B]: its maximum is an
   upper bound on the MERRILL barrier between those two minima (mountain-pass property, discrete path);
4. MERRILL PathMinimize from that path (19 images, 1000 evaluations).
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from merrill_diag import read_tec, ang_deg, tet_geometry, node_weights  # noqa: E402

KBT = 1.380649e-23 * 300.0


def main():
    mbin, mesh, bandf, cell, wd = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]) * 1e-3, Path(sys.argv[5])
    extra = sys.argv[6] if len(sys.argv) > 6 else ""
    mbin = str(Path(mbin).resolve())
    wd.mkdir(parents=True, exist_ok=True)
    mname = Path(mesh).name
    shutil.copy(mesh, wd / mname)
    head = ["Set: MaxMeshNumber = 1", f"ReadMesh: mesh=1, filename={mname}", "Set: MaxEnergyEvaluations = 20000",
            "Material: Type=Magnetite, TempC=20"] + [l for l in extra.split(";") if l] + ["ConjugateGradient"]

    def run(name, lines):
        (wd / f"{name}.mscr").write_text("\n".join(lines + ["End"]) + "\n")
        with open(wd / f"{name}.log", "w") as lf:
            subprocess.run([mbin, f"{name}.mscr"], cwd=wd, stdout=lf, stderr=subprocess.STDOUT, check=True)

    run("topo", head + ["Randomize: type=all", "WriteMagnetization: filename=topo"])
    c, t, _ = read_tec(wd / "topo_mult.tec")
    vol, _ = tet_geometry(c, t)
    w = node_weights(c, t, vol)
    d = lambda x, y: float(np.sqrt(np.sum(w * ang_deg(x, y) ** 2)))
    band = np.load(bandf)  # (n_img, Nx, Ny, Nz, 3)
    nimg, N = band.shape[0], band.shape[1:4]
    axes = [(np.arange(n) + 0.5) * cell - 0.5 * n * cell for n in N]  # um, centred
    q = np.clip(c - c.mean(0), [a[0] for a in axes], [a[-1] for a in axes])
    imgs = []
    for k in range(nimg):
        comp = [RegularGridInterpolator(axes, band[k][..., j])(q) for j in range(3)]
        m = np.stack(comp, 1)
        imgs.append(m / np.linalg.norm(m, axis=1, keepdims=True))
    for k, m in enumerate(imgs):
        np.savetxt(wd / f"img_{k + 1:02d}.dat", np.hstack([c, m]), fmt="%.10e")
    run("relax_ends", head + ["ReadMagnetization: filename=img_01.dat", "Minimize", "WriteMagnetization: filename=endA",
                              f"ReadMagnetization: filename=img_{nimg:02d}.dat", "Minimize", "WriteMagnetization: filename=endB"])
    L = lambda f: (lambda a: a[:, 3:6] / np.linalg.norm(a[:, 3:6], axis=1, keepdims=True))(np.loadtxt(f))
    shiftA, shiftB = d(L(wd / "endA.dat"), imgs[0]), d(L(wd / "endB.dat"), imgs[-1])
    files = ["endA.dat"] + [f"img_{k + 1:02d}.dat" for k in range(1, nimg - 1)] + ["endB.dat"]
    load = []
    for i, f in enumerate(files, 1):
        load += [f"ReadMagnetization: filename={f}", f"MagnetizationToPath: {i}"]
    run("energies", head + [f"Set: PathN = {nimg}"] + load + ["RenewPath", "PathStructureEnergies: filename=mapped_E.txt"])
    E = np.array([float(l.split()[-1]) for l in open(wd / "mapped_E.txt") if l.strip()]) / KBT
    prof_mapped = E - min(E[0], E[-1])
    run("pathmin", head + ["Set: MaxPathEvaluations = 1000", f"Set: PathN = {nimg}"] + load +
        ["RenewPath", "Set: ExchangeCalculator = 1", "WriteTecPlotPath: filename=initpath.tec", "PathLogfile: filename=neb", "PathMinimize",
         "PathStructureEnergies: filename=final_E.txt"])
    E2 = np.array([float(l.split()[-1]) for l in open(wd / "final_E.txt") if l.strip()]) / KBT
    prof_final = E2 - min(E2[0], E2[-1])
    txt = (wd / "pathmin.log").read_text()
    res = {"band": bandf, "mesh": mname, "n_nodes": int(len(c)), "extra_material": extra,
           "end_relax_shift_deg": [round(shiftA, 2), round(shiftB, 2)], "ends_dE_kBT": round(float(E[-1] - E[0]), 2),
           "mapped_path_max_kBT": round(float(prof_mapped.max()), 2), "mapped_profile_kBT": [round(float(v), 1) for v in prof_mapped],
           "pathmin_max_kBT": round(float(prof_final.max()), 2), "pathmin_profile_kBT": [round(float(v), 1) for v in prof_final],
           "pathmin_evals": sum(int(x) for x in re.findall(r"CalcPathAction Calls:\s*(\d+)", txt)),
           "pathmin_stop": ("failed" if "FAILED TO CONVERGE" in txt else "max_evals" if "MAX Path" in txt
                            else "stagnation" if "DeltaS negligible" in txt else "?")}
    (wd / "crossmap.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
