#!/usr/bin/env python3
"""
Structured escape from a higher-order saddle (FD cube): push the saddle image of a converged band
along a chosen Hessian mode (Gaussian taper over neighbouring images), re-converge with GNEB.
Usage: fd_modepush.py <band.npz> <modes.npy> <mode_index> <amplitude_rad> <label> <outdir>
"""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
exec(open(Path(__file__).resolve().parent / "fd_defs.py").read())
torch.set_default_dtype(torch.float64)
bfile, mfile, j, amp, label, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5], Path(sys.argv[6])
out.mkdir(parents=True, exist_ok=True)
N, dx = (30, 30, 30), (5e-9,) * 3
st, tm, mn = setup_magnetite(N, dx)
st.m = make_seed_vortex(N, dx, [0, 0, 1], st.m.device, st.m.dtype); mn.minimize(st); m0 = st.m.clone()
def E(m):
    st.m = m; return float(sum(t.E(st) for t in tm))
E0 = E(m0)
band = [torch.tensor(b) for b in np.load(bfile)["band"].astype(np.float64)]
Es = [E(b) for b in band]; idx = int(np.argmax(Es[1:-1])) + 1
v = torch.tensor(np.load(mfile)[j])
v = v / torch.sqrt((v ** 2).sum(-1).mean())  # rms 1 per cell
for i in range(1, len(band) - 1):
    w = np.exp(-0.5 * ((i - idx) / 1.5) ** 2)
    vi = v - (v * band[i]).sum(-1, keepdim=True) * band[i]
    m = band[i] + amp * w * vi
    band[i] = m / m.norm(dim=-1, keepdim=True)
pushed = (max(E(b) for b in band[1:-1]) - E0) / KBT_300
r = run_gneb_fire(st, tm, band, label, max_iter=3000, verbose=True, ci_gate=5e3, dt_max=6e-5)
r.update({"mode_index": j, "amplitude": amp, "pushed_band_max_rel_m0": round(pushed, 1),
          "profile_rel_m0_kBT": [round((E(m) - E0) / KBT_300, 2) for m in band]})
np.savez_compressed(out / f"band_{label}.npz", band=np.stack([m.numpy().astype(np.float32) for m in band]))
(out / f"result_{label}.json").write_text(json.dumps(r, indent=1))
print(json.dumps({k: r[k] for k in ["label", "conv", "iters", "barrier_kBT", "final_maxF", "pushed_band_max_rel_m0"]}))
print("profile rel m0:", r["profile_rel_m0_kBT"])
