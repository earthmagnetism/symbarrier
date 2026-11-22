#!/usr/bin/env python3
"""
Lowest eigenvalues of the Riemannian Hessian of the FD micromagnetic energy on (S^2)^N.

For a state m (unit vectors), tangent basis {e1, e2} per cell, v = a e1 + b e2.
Riemannian Hessian:  Hess[v] = P_m( D^2 E_ext . v ) - (m . grad E_ext) v
with E_ext the natural Euclidean extension of the energy (demag/exchange quadratic, cubic
anisotropy quartic), grad E_ext = -mu0 Ms h dV, and D^2 E_ext . v by central differences.
Eigenvalues (in k_B T_300 per rad^2) from scipy eigsh, k lowest algebraic.

Usage: fd_hessian.py <system> <state-file> <image index or -1 for argmax-energy> <label> <out.json>
  system: cube | prism | truncoct
  state-file: .npz with 'band' (n_img, nx, ny, nz, 3) or .pt tensor (band) or .npy (single state)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, eigsh

sys.path.insert(0, str(Path(__file__).resolve().parent))
torch.set_default_dtype(torch.float64)
MU0 = 4e-7 * np.pi
KBT = 1.380649e-23 * 300.0


def load_system(name):
    if name in ("cube", "prism"):
        exec(open(Path(__file__).resolve().parent / "fd_defs.py").read(), globals())
        N = (30, 30, 30) if name == "cube" else (24, 24, 40)
        dx = (5e-9,) * 3
        st, tm, _ = setup_magnetite(N, dx)
        mask = torch.ones(N)
        return st, tm, mask, dx
    import fd_truncoct as T
    st, tm, _ = T.setup()
    return st, tm, T.MASK, T.DX


def main():
    system, sfile, idx, label, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
    st, tm, mask, dx = load_system(system)
    dV = dx[0] * dx[1] * dx[2]
    Ms = st.material["Ms"]  # [nx,ny,nz,1]

    def E(m):
        st.m = m
        return float(sum(t.E(st) for t in tm))

    def grad(m):
        st.m = m
        h = sum(t.h(st) for t in tm)
        return -MU0 * Ms * h * dV  # dE/dm (J per unit m)

    if sfile.endswith(".npz"):
        band = torch.tensor(np.load(sfile)["band"].astype(np.float64))
    elif sfile.endswith(".pt"):
        band = torch.load(sfile).double()
    else:
        band = torch.tensor(np.load(sfile).astype(np.float64))[None]
    if idx < 0:
        Es = [E(b) for b in band]
        idx = int(np.argmax(Es[1:-1])) + 1 if len(band) > 2 else 0
    m = band[idx].clone()
    m = m / m.norm(dim=-1, keepdim=True)
    inside = (mask > 0)
    # tangent basis
    ref = torch.tensor([1.0, 0.0, 0.0]).expand_as(m).clone()
    alt = torch.tensor([0.0, 1.0, 0.0]).expand_as(m)
    use_alt = (m * ref).sum(-1).abs() > 0.9
    ref[use_alt] = alt[use_alt]
    e1 = ref - (ref * m).sum(-1, keepdim=True) * m
    e1 = e1 / e1.norm(dim=-1, keepdim=True)
    e2 = torch.cross(m, e1, dim=-1)
    g0 = grad(m)
    lam = (m * g0).sum(-1, keepdim=True)  # m . grad E
    n_in = int(inside.sum())
    eps = 1e-5

    def unpack(x):
        x = torch.tensor(x, dtype=torch.float64)
        a = torch.zeros(m.shape[:3])
        b = torch.zeros(m.shape[:3])
        a[inside] = x[:n_in]
        b[inside] = x[n_in:]
        return a[..., None] * e1 + b[..., None] * e2

    def pack(v):
        return np.concatenate([(v * e1).sum(-1)[inside].numpy(), (v * e2).sum(-1)[inside].numpy()])

    nmv = [0]

    def matvec(x):
        nmv[0] += 1
        v = unpack(np.asarray(x).ravel())
        d2 = (grad(m + eps * v) - grad(m - eps * v)) / (2 * eps)
        hv = d2 - (d2 * m).sum(-1, keepdim=True) * m - lam * v
        return pack(hv) / KBT

    n = 2 * n_in
    op = LinearOperator((n, n), matvec=matvec, dtype=np.float64)
    # shift so that the lowest eigenvalues are the largest-magnitude ones of (sigma*I - H)
    vals, vecs = eigsh(op, k=3, which="SA", tol=1e-6, maxiter=5000)
    order = np.argsort(vals)
    vals = vals[order]
    v0 = unpack(vecs[:, order[0]])
    res = {"label": label, "system": system, "image": idx, "E_rel_first_kBT": round((E(m) - E(band[0])) / KBT, 2),
           "lowest_eigenvalues_kBT_per_rad2": [float(f"{v:.4g}") for v in vals], "n_negative": int((vals < 0).sum()),
           "n_matvec": nmv[0], "dim": n,
           "max_tangent_force_kBT_per_rad": float(((g0 - lam * m).norm(dim=-1)).max() / KBT)}
    if len(band) > 2:
        tau = band[min(idx + 1, len(band) - 1)] - band[max(idx - 1, 0)]
        tau = tau - (tau * m).sum(-1, keepdim=True) * m
        res["overlap_lowest_mode_with_band_tangent"] = round(float(abs((v0 * tau).sum()) / (v0.norm() * tau.norm())), 3)
    import os
    if os.environ.get("SAVE_MODES"):
        np.save(os.environ["SAVE_MODES"], np.stack([unpack(vecs[:, order[j]]).numpy() for j in range(len(order))]))
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
