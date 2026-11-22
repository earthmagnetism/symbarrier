#!/usr/bin/env python3
"""
FD (magnum.np) side of the MERRILL/FD alignment for the 120 x 120 x 200 nm prism.

Material = MERRILL v2.2.1 'Type=Magnetite, TempC=20' constants exactly (Ms, A, K1, K2), K2 optional.
State   = FD transverse ground state B2 (5 nm), resampled to the target cell size and relaxed.
Hop     = B2 -> its D4h image at ~97 deg (the rate-limiting class), straight geodesic initial band,
          19 images, GNEB-FIRE with climbing image, force tolerance TOL (A/m).

Usage: prism_align_fd.py <cell_nm> <with_K2: 0|1> <tol_A_per_m> <outdir> [max_iter]
"""
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).resolve().parent))
torch.set_default_dtype(torch.float64)
ns = {}
exec(open(Path(__file__).resolve().parent / "fd_defs.py").read(), ns)
from magnumnp import Mesh, State, DemagField, ExchangeField, MinimizerBB  # noqa: E402

MU0, KB = ns["MU0"], ns["KBT_300"]
MS, AEX, K1, K2 = 482476.91085597320, 1.3444090643665902e-11, -13039.710203218889, -3154.0197698514348
import os
from _paths import DATA, FIGURES, MERRILL, RUNS  # noqa: F401
if os.environ.get("FDCONST") == "1":  # original FD constants of Table 2 (no K2)
    MS, AEX, K1 = 4.80e5, 1.33e-11, -1.33e4


class CubicK2:
    def __init__(self, K2, Ms):
        self.K2, self.Ms = float(K2), float(Ms)

    def h(self, state):
        m = state.m
        mx, my, mz = m[..., 0], m[..., 1], m[..., 2]
        pref = -2.0 * self.K2 / (MU0 * self.Ms)
        return torch.stack([pref * mx * my * my * mz * mz, pref * my * mz * mz * mx * mx, pref * mz * mx * mx * my * my], dim=-1)

    def E(self, state):
        m = state.m
        dV = float(state.mesh.dx[0] * state.mesh.dx[1] * state.mesh.dx[2])
        return self.K2 * dV * torch.sum(m[..., 0] ** 2 * m[..., 1] ** 2 * m[..., 2] ** 2)


def rot_grid(m, R):
    perm = [int(np.nonzero(R[a])[0][0]) for a in range(3)]
    sgn = [int(R[a, perm[a]]) for a in range(3)]
    arr = m.permute(*perm, 3)
    for a in range(3):
        if sgn[a] < 0:
            arr = torch.flip(arr, dims=[a])
    return float(np.linalg.det(R)) * torch.einsum("ij,...j->...i", torch.tensor(R, dtype=torch.float64), arr)


def main():
    cell, withk2, tol, out = float(sys.argv[1]) * 1e-9, int(sys.argv[2]), float(sys.argv[3]), Path(sys.argv[4])
    max_iter = int(sys.argv[5]) if len(sys.argv) > 5 else 8000
    out.mkdir(parents=True, exist_ok=True)
    N = tuple(int(round(L / cell)) for L in (120e-9, 120e-9, 200e-9))
    dx = (cell,) * 3
    mesh = Mesh(N, dx)
    st = State(mesh)
    st.material = {"Ms": MS, "A": AEX}
    terms = [DemagField(), ExchangeField(), ns["ExactCubicAnisotropy"](K1=K1, Ms=MS)] + ([CubicK2(K2, MS)] if withk2 else [])
    mn = MinimizerBB(terms)
    fix = lambda m: m / m.norm(dim=-1, keepdim=True)
    b2 = torch.tensor(np.load(f"{RUNS}/benchmarks/fd_c2/B2_ground.npy").astype(np.float64))  # (24,24,40,3) at 5 nm
    m = Fn.interpolate(b2.permute(3, 0, 1, 2)[None], size=N, mode="trilinear", align_corners=False)[0].permute(1, 2, 3, 0)
    t0 = time.time()
    st.m = fix(m.contiguous())
    mn.minimize(st)
    m0 = fix(st.m.clone())

    def E(mm):
        st.m = mm
        return float(sum(t.E(st) for t in terms))

    def Eparts(mm):
        st.m = mm
        return {type(t).__name__: float(t.E(st)) / KB for t in terms}

    def d(a, b):
        x = torch.acos((a * b).sum(-1).clamp(-1, 1))
        return float(torch.rad2deg(torch.sqrt((x ** 2).mean())))

    # D4h images of m0; pick the ~97 deg class representative (not -m0)
    Rs = []
    for perm in itertools.permutations(range(3)):
        if tuple(N[p] for p in perm) != N:
            continue
        for sg in itertools.product([1, -1], repeat=3):
            R = np.zeros((3, 3))
            for i, j in enumerate(perm):
                R[i, j] = sg[i]
            Rs.append(R)
    imgs = [fix(rot_grid(m0, R)) for R in Rs]
    cands = [(d(mi, m0), d(mi, -m0), k) for k, mi in enumerate(imgs)]
    tgt = min((c for c in cands if 85 < c[0] < 110 and c[1] > 20), key=lambda c: abs(c[0] - 97))
    st.m = imgs[tgt[2]].clone()
    mn.minimize(st)
    m1 = fix(st.m.clone())
    relax_s = time.time() - t0
    band = [fix(x) for x in ns["build_slerp_random_band"](m0, m1, 19, sigma=0.0, seed=0)]
    t1 = time.time()
    r = ns["run_gneb_fire"](st, terms, band, f"prism_{sys.argv[1]}nm", max_iter=max_iter, tol_f=tol, ci_gate=5e3, dt_max=6e-5)
    E0 = E(m0)
    prof = [(E(fix(x)) - E0) / KB for x in band]
    i_s = int(np.argmax(prof))
    res = {"constants": "FD original" if os.environ.get("FDCONST") == "1" else "MERRILL 20C", "cell_nm": float(sys.argv[1]), "N": N, "with_K2": bool(withk2), "tol_A_per_m": tol, "conv": r["conv"], "iters": r["iters"],
           "final_maxF": r["final_maxF"], "barrier_kBT": round(max(prof), 2), "E_end_rel_kBT": round(prof[-1], 2),
           "rms_m0_to_end_deg": round(tgt[0], 1), "relax_s": round(relax_s, 1), "gneb_s": round(time.time() - t1, 1),
           "E_parts_m0_kBT": Eparts(m0), "E_parts_saddle_kBT": Eparts(fix(band[i_s])), "profile_kBT": [round(p, 2) for p in prof]}
    np.save(out / "m0.npy", m0.numpy())
    np.save(out / "saddle.npy", fix(band[i_s]).numpy())
    np.save(out / "band.npy", torch.stack([fix(x) for x in band]).numpy())
    (out / "align_fd.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "profile_kBT"}, indent=1))


if __name__ == "__main__":
    main()
