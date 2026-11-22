#!/usr/bin/env python3
"""
Symmetry-breaking continuation of a converged FD band (150 nm cube).

Adds small random noise (amplitude sigma) to all interior images of a saved band, renormalises,
and re-converges with the GNEB-FIRE solver (damped parameters). If the original saddle was a
higher-order saddle confined to a symmetric subspace, the band relaxes to a lower first-order saddle.

Usage: fd_symbreak.py <band.npz> <sigma> <seed> <label> <outdir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
exec(open(Path(__file__).resolve().parent / "fd_defs.py").read())
torch.set_default_dtype(torch.float64)


def main():
    bfile, sigma, seed, label, out = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4], Path(sys.argv[5])
    out.mkdir(parents=True, exist_ok=True)
    N, dx = (30, 30, 30), (5e-9,) * 3
    st, tm, mn = setup_magnetite(N, dx)
    st.m = make_seed_vortex(N, dx, [0, 0, 1], st.m.device, st.m.dtype)
    mn.minimize(st)
    m0 = st.m.clone()

    def E(m):
        st.m = m
        return float(sum(t.E(st) for t in tm))

    E0 = E(m0)
    band = [torch.tensor(b) for b in np.load(bfile)["band"].astype(np.float64)]
    g = torch.Generator().manual_seed(seed)
    for i in range(1, len(band) - 1):
        m = band[i] + sigma * torch.randn(band[i].shape, generator=g)
        band[i] = m / m.norm(dim=-1, keepdim=True)
    before = max(E(b) for b in band[1:-1])
    r = run_gneb_fire(st, tm, band, label, max_iter=3000, verbose=True, ci_gate=5e3, dt_max=6e-5)
    r["sigma"], r["seed"] = sigma, seed
    r["profile_rel_m0_kBT"] = [round((E(m) - E0) / KBT_300, 2) for m in band]
    r["perturbed_band_max_rel_m0_kBT"] = round((before - E0) / KBT_300, 1)
    np.savez_compressed(out / f"band_{label}.npz", band=np.stack([m.numpy().astype(np.float32) for m in band]))
    (out / f"result_{label}.json").write_text(json.dumps(r, indent=1))
    print(json.dumps({k: r[k] for k in ["label", "conv", "iters", "barrier_kBT", "final_maxF"]}))
    print("profile rel m0:", r["profile_rel_m0_kBT"])


if __name__ == "__main__":
    main()
