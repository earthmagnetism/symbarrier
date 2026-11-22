"""Thin driver for MERRILL (tested with v2.2.1, key=value script syntax). MERRILL is called as an external
program; nothing here links to it."""
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

KB = 1.380649e-23


class Merrill:
    def __init__(self, binary, mesh, workdir, material="Type=Magnetite, TempC=20", extra=(), temperature_K=300.0):
        self.bin = str(Path(binary).resolve())
        self.wd = Path(workdir)
        self.wd.mkdir(parents=True, exist_ok=True)
        self.mesh = Path(mesh).name
        if not (self.wd / self.mesh).exists():
            shutil.copy(mesh, self.wd / self.mesh)
        self.material = material
        self.extra = list(extra)
        self.kT = KB * temperature_K

    def head(self):
        return (["Set: MaxMeshNumber = 1", f"ReadMesh: mesh=1, filename={self.mesh}", "Set: MaxEnergyEvaluations = 20000",
                 f"Material: {self.material}"] + self.extra + ["ConjugateGradient"])

    def run(self, name, lines):
        (self.wd / f"{name}.mscr").write_text("\n".join(self.head() + lines + ["End"]) + "\n")
        with open(self.wd / f"{name}.log", "w") as lf:
            subprocess.run([self.bin, f"{name}.mscr"], cwd=self.wd, stdout=lf, stderr=subprocess.STDOUT, check=True)
        return (self.wd / f"{name}.log").read_text()

    def write_field(self, name, coords, m):
        np.savetxt(self.wd / f"{name}.dat", np.hstack([coords, m]), fmt="%.10e")
        return f"{name}.dat"

    def read_field(self, name):
        a = np.loadtxt(self.wd / name)
        return a[:, :3], a[:, 3:6] / np.linalg.norm(a[:, 3:6], axis=1, keepdims=True)

    def topology(self):
        """Node coordinates in MERRILL order and tets, via one random state written to TecPlot."""
        from .geometry import read_tecplot
        self.run("topo", ["Randomize: type=all", "WriteMagnetization: filename=topo"])
        c, t, _ = read_tecplot(self.wd / "topo_mult.tec")
        return c, t

    def minimize(self, files_in, names_out, tag="relax"):
        lines = []
        for f, o in zip(files_in, names_out):
            lines += [f"ReadMagnetization: filename={f}", "Minimize", f"WriteMagnetization: filename={o}"]
        self.run(tag, lines)
        return [f"{o}.dat" for o in names_out]

    def energies(self, files, tag="E"):
        """Total energies (units of kT) of the listed states."""
        lines = [f"Set: PathN = {len(files)}"]
        for i, f in enumerate(files, 1):
            lines += [f"ReadMagnetization: filename={f}", f"MagnetizationToPath: {i}"]
        self.run(tag, lines + ["RenewPath", f"PathStructureEnergies: filename={tag}.txt"])
        return np.array([float(l.split()[-1]) for l in open(self.wd / f"{tag}.txt") if l.strip()]) / self.kT

    def neb(self, start, end, tag, n_images=19, max_evals=1000, initial=None):
        """MERRILL NEB between two states. initial=None: MakeInitialPath (default); otherwise a list of
        n_images files to use as the initial path. Returns dict with profile (kT, relative to the lower end),
        max, evaluations and stop reason."""
        if initial is None:
            lines = ["Set: PathN = 2", f"ReadMagnetization: filename={start}", "MagnetizationToPath: 1",
                     f"ReadMagnetization: filename={end}", "MagnetizationToPath: 2", f"RefinePathTo: {n_images}"]
            lines += ["Set: ExchangeCalculator = 1", "MakeInitialPath"]
        else:
            lines = [f"Set: PathN = {len(initial)}"]
            for i, f in enumerate(initial, 1):
                lines += [f"ReadMagnetization: filename={f}", f"MagnetizationToPath: {i}"]
            lines += ["RenewPath", "Set: ExchangeCalculator = 1"]
        lines = [f"Set: MaxPathEvaluations = {max_evals}"] + lines
        lines += [f"WriteTecPlotPath: filename={tag}_init.tec", f"PathLogfile: filename={tag}", "PathMinimize",
                  f"PathStructureEnergies: filename={tag}_E.txt", f"WriteTecPlotPath: filename={tag}_final.tec"]
        log = self.run(tag, lines)
        e = np.array([float(l.split()[-1]) for l in open(self.wd / f"{tag}_E.txt") if l.strip()]) / self.kT
        prof = e - min(e[0], e[-1])
        stop = ("failed" if "FAILED TO CONVERGE" in log else "max_evals" if "MAX Path" in log
                else "stagnation" if "DeltaS negligible" in log else "unknown")
        return {"profile_kT": prof.round(2).tolist(), "barrier_kT": round(float(prof.max()), 2), "E_abs_kT": e.tolist(),
                "evals": sum(int(x) for x in re.findall(r"CalcPathAction Calls:\s*(\d+)", log)), "stop": stop}
