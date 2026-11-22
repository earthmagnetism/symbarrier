#!/usr/bin/env python3
"""
FD cross-check of the truncated-octahedron result (magnum.np, GNEB-FIRE solver).

Geometry: regular truncated octahedron |x|,|y|,|z| <= 2s, |x|+|y|+|z| <= 3s, s = 37.5 nm
(same definition as symbarrier.meshes.truncated_octahedron), voxelised on a 30^3 grid of 5 nm cells.
Outside cells: Ms = A = 0, all fields masked, m fixed to +z in every image.

Modes:
  relax  <outdir>                        : relax a [111] vortex seed -> m0.npy ; symmetry image I = R_[110](pi) m0, relax -> I.npy
  band   <outdir> <ax,ay,az> <label> [overrides] : covariant band m0 -> -m0
  hop    <outdir> <start.npy> <end.npy> <n_img> <label> [overrides] : geodesic-initialised NEB
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
exec(open(Path(__file__).resolve().parent / "fd_defs.py").read())
from magnumnp import Mesh, State, DemagField, ExchangeField, MinimizerBB  # noqa: E402

torch.set_default_dtype(torch.float64)
N, DX = (30, 30, 30), (5e-9, 5e-9, 5e-9)
MS, AEX, K1 = 4.80e5, 1.33e-11, -1.33e4


def make_mask():
    x = (torch.arange(N[0]) + 0.5) * DX[0] - 0.5 * N[0] * DX[0]
    X, Y, Z = torch.meshgrid(x, x, x, indexing="ij")
    s = 37.5e-9
    inside = (X.abs() <= 2 * s) & (Y.abs() <= 2 * s) & (Z.abs() <= 2 * s) & (X.abs() + Y.abs() + Z.abs() <= 3 * s + 1e-15)
    return inside.double()


MASK = make_mask()
M4 = MASK[..., None]
OUT = torch.tensor([0.0, 0.0, 1.0]) * (1 - M4)


class Masked:
    def __init__(self, term):
        self.t = term

    def h(self, state):
        return self.t.h(state) * M4

    def E(self, state):
        return self.t.E(state)


class MaskedCubic(ExactCubicAnisotropy):
    def h(self, state):
        return super().h(state) * M4

    def E(self, state):
        m = state.m
        dV = float(state.mesh.dx[0] * state.mesh.dx[1] * state.mesh.dx[2])
        return self.K1 * dV * torch.sum(MASK * (m[..., 0] ** 2 * m[..., 1] ** 2 + m[..., 1] ** 2 * m[..., 2] ** 2 + m[..., 2] ** 2 * m[..., 0] ** 2))


def fix(m):
    m = m * M4 + OUT
    return m / m.norm(dim=-1, keepdim=True)


def setup():
    mesh = Mesh(N, DX)
    st = State(mesh)
    st.material = {"Ms": MS * MASK[..., None], "A": AEX * MASK[..., None]}
    st.m = st.Constant([0.0, 0.0, 1.0])
    terms = [Masked(DemagField()), Masked(ExchangeField()), MaskedCubic(K1=K1, Ms=MS)]
    return st, terms, MinimizerBB(terms)


def energy(st, terms, m):
    st.m = m
    return float(sum(t.E(st) for t in terms))


def relax(st, mn, m):
    st.m = fix(m)
    mn.minimize(st)
    return fix(st.m.clone())


def rms_in(a, b):
    d = torch.acos((a * b).sum(-1).clamp(-1, 1))[MASK > 0]
    return float(torch.rad2deg(torch.sqrt((d ** 2).mean())))


def rot_grid(m, R):
    perm = [int(np.nonzero(R[a])[0][0]) for a in range(3)]
    sgn = [int(R[a, perm[a]]) for a in range(3)]
    arr = m.permute(*perm, 3)
    for a in range(3):
        if sgn[a] < 0:
            arr = torch.flip(arr, dims=[a])
    return torch.einsum("ij,...j->...i", torch.tensor(R, dtype=torch.float64), arr)


def c2_110():
    # R_[110](pi): (x,y,z) -> (y,x,-z)
    return np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], float)


def overrides(args):
    return {k: float(v) for k, v in (a.split("=") for a in args)}


def main():
    mode, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    st, terms, mn = setup()
    if mode == "relax":
        seed = make_seed_vortex(N, DX, [1, 1, 1], st.m.device, st.m.dtype)
        m0 = relax(st, mn, seed)
        E0 = energy(st, terms, m0)
        mavg = (m0 * M4).sum((0, 1, 2)) / MASK.sum()
        I0 = fix(rot_grid(m0, c2_110()))
        I = relax(st, mn, I0)
        res = {"n_inside": int(MASK.sum()), "mean_m0": np.round(mavg.numpy(), 4).tolist(),
               "E_I_minus_E_m0_kBT": round((energy(st, terms, I) - E0) / KBT_300, 3),
               "I_vs_gm0_rms_deg": round(rms_in(I, I0), 3),
               "minus_m0_vs_C2_10m1_m0_rms_deg": round(rms_in(fix(-m0 * M4 + OUT * 0 + OUT), fix(rot_grid(m0, np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]], float)))), 3)}
        torch.save(m0, out / "m0.pt")
        torch.save(I, out / "I.pt")
        (out / "relax.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res))
        return
    m0 = torch.load(out / "m0.pt")
    E0 = energy(st, terms, m0)
    if mode == "band":
        axis, label = [float(v) for v in sys.argv[3].split(",")], sys.argv[4]
        kw = overrides(sys.argv[5:])
        band = [fix(m) for m in build_so3_covariant_band(m0, fix(-m0 * M4 + OUT), DX, 19, axis)]
        band[-1] = fix(-m0 * M4 + OUT)
    else:
        a, b, nimg, label = sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6]
        kw = overrides(sys.argv[7:])
        ma = m0 if a == "m0" else torch.load(a)
        mb = torch.load(b)
        band = [fix(m) for m in build_slerp_random_band(ma, mb, nimg, sigma=0.0, seed=0)]
    init_prof = [round((energy(st, terms, m) - E0) / KBT_300, 1) for m in band]
    r = run_gneb_fire(st, terms, band, label, max_iter=1500, verbose=True, **kw)
    band = [fix(m) for m in band]
    r["profile_rel_m0_kBT"] = [round((energy(st, terms, m) - E0) / KBT_300, 2) for m in band]
    r["init_profile_rel_m0_kBT"] = init_prof
    r["solver_overrides"] = kw
    r["mean_m_inside"] = [np.round(((m * M4).sum((0, 1, 2)) / MASK.sum()).numpy(), 3).tolist() for m in band]
    torch.save(torch.stack(band), out / f"band_{label}.pt")
    (out / f"result_{label}.json").write_text(json.dumps(r, indent=1))
    print(json.dumps({k: r[k] for k in ["label", "conv", "iters", "barrier_kBT", "final_maxF"]}))
    print("profile:", r["profile_rel_m0_kBT"])


if __name__ == "__main__":
    main()
