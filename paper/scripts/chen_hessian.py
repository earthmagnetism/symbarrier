"""Riemannian Hessian index check for the Chen cuboid (105 nm, 2.92 nm grid): saddle of the short-axis -> long-axis
vortex hop (the 572.0 class), and the short-axis vortex, long-axis vortex and tilted flower.
Same operator as scripts/fd_hessian.py (central differences of the effective field, Lanczos, k_B T_300 / rad^2)."""
import json, sys, time
import numpy as np, torch
from scipy.sparse.linalg import LinearOperator, eigsh
import os
from _paths import DATA, FIGURES, MERRILL, RUNS  # noqa: F401
os.makedirs(RUNS / "chen", exist_ok=True)
os.chdir(RUNS / "chen")  # inputs and outputs of the cuboid runs live in <RUNS>/chen
from symbarrier.backends.magnumnp_backend import MagnumNPBackend, MAGNETITE_MERRILL_20C, MU0
from symbarrier import network as net, group as grp
torch.set_default_dtype(torch.float64)
nx = 36; N = (36, 30, 24)
be = MagnumNPBackend(N, (105e-9 / nx,) * 3, material=MAGNETITE_MERRILL_20C, gneb_kw=dict(max_iter=3000))
S = np.load(f"chen_states_{nx}.npz")
G = grp.with_time_reversal(be.point_group())
nodes, fam, perm = net.build_nodes([S["vortex"], S["perturbed_flower"], S["tilted_x"]], G, be.action(), be.weights)
assert fam[4] == 1 and fam[8] == 2
dV = float(np.prod(be.dx)); Ms = be.material["Ms"]

def hessian(f, k=3):
    m = be._fix(be.to_grid(f))
    grad = lambda x: -MU0 * Ms * be._field(x)[0] * dV
    ref = torch.tensor([1.0, 0, 0]).expand_as(m).clone(); alt = torch.tensor([0, 1.0, 0]).expand_as(m)
    use = (m * ref).sum(-1).abs() > 0.9; ref[use] = alt[use]
    e1 = ref - (ref * m).sum(-1, keepdim=True) * m; e1 = e1 / e1.norm(dim=-1, keepdim=True); e2 = torch.cross(m, e1, dim=-1)
    g0 = grad(m); lam = (m * g0).sum(-1, keepdim=True); n = m[..., 0].numel(); eps = 1e-5; nmv = [0]
    def unpack(x):
        x = torch.tensor(np.asarray(x).ravel()); return x[:n].reshape(N)[..., None] * e1 + x[n:].reshape(N)[..., None] * e2
    def matvec(x):
        nmv[0] += 1; v = unpack(x)
        d2 = (grad(m + eps * v) - grad(m - eps * v)) / (2 * eps)
        hv = d2 - (d2 * m).sum(-1, keepdim=True) * m - lam * v
        return np.concatenate([(hv * e1).sum(-1).ravel().numpy(), (hv * e2).sum(-1).ravel().numpy()]) / be.kT
    vals = np.sort(eigsh(LinearOperator((2 * n, 2 * n), matvec=matvec), k=k, which="SA", tol=1e-6, maxiter=5000)[0])
    return {"lowest_eigenvalues_kT_per_rad2": [float(f"{v:.4g}") for v in vals], "n_negative": int((vals < -1e-3).sum()),
            "max_tangent_force_kT_per_rad": round(float((g0 - lam * m).norm(dim=-1).max() / be.kT), 3), "n_matvec": nmv[0]}

out = {}
t = time.time()
h = be.neb(nodes[0], nodes[4], n_images=19, tag="hop_0_4")
prof = np.asarray(h["profile_kT"]); i = int(np.argmax(prof))
np.save("chen_hop04_band_36.npy", np.stack(h["path"]).astype(np.float32))
out["hop_0_4"] = {"saddle_above_m0": round(float(prof[i] - prof[0]), 2), "image": i, "info": h["info"], "neb_s": round(time.time() - t, 1)}
for name, f in [("saddle_0_4", h["path"][i]), ("short_axis_vortex", nodes[0]), ("long_axis_vortex", nodes[4]), ("tilted_flower", nodes[8])]:
    t1 = time.time(); out[name] = hessian(f) | {"s": round(time.time() - t1, 1)}
    print(name, out[name], flush=True)
json.dump(out, open("chen_hessian_36.json", "w"), indent=1, default=str)
print(json.dumps(out["hop_0_4"], default=str))
