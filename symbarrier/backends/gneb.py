"""Geodesic nudged elastic band with FIRE on (S^2)^N, climbing image switched on below a force gate.

Solver-neutral: it needs only `field(m) -> (h_eff, E)` for one image. Fields are torch tensors (..., 3) with an
optional boolean mask of active cells. The algorithm is the one used for all finite-difference results in the paper
(upwind tangent with energy-weighted bisection at extrema, spring force on rms geodesic spacing, FIRE velocity
mixing, step clipped to 0.12 rad per cell)."""
import math
import time

import torch


def _rms_geo(a, b, mask):
    x = torch.acos((a * b).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7))
    return torch.sqrt((x[mask] ** 2).mean()).item()


def gneb_fire(field, band, mask=None, max_iter=3000, k_spring=2e5, tol_f=500.0, ci_gate=5e3, dt_max=6e-5,
              verbose=False, label="gneb"):
    n = len(band)
    band = [b.clone() for b in band]
    if mask is None:
        mask = torch.ones(band[0].shape[:-1], dtype=torch.bool)
    n_cells = int(mask.sum())
    dt, dt_min = min(1.0e-4, dt_max), 2.0e-5
    alpha_start = alpha = 0.15
    f_inc, f_dec, f_alpha, n_min = 1.05, 0.5, 0.99, 5
    n_pos, ci = 0, False
    vel = [torch.zeros_like(m) for m in band]
    t0 = time.perf_counter()
    conv = False
    for it in range(1, max_iter + 1):
        hs, Es = zip(*(field(m) for m in band))
        i_max = max(range(1, n - 1), key=lambda i: Es[i])
        d = [_rms_geo(band[i], band[i + 1], mask) for i in range(n - 1)]
        F = [None] * n
        fmax = 0.0
        for i in range(1, n - 1):
            m = band[i]
            if Es[i + 1] > Es[i] > Es[i - 1]:
                tau = band[i + 1] - m
            elif Es[i - 1] > Es[i] > Es[i + 1]:
                tau = m - band[i - 1]
            else:
                dmax = max(abs(Es[i + 1] - Es[i]), abs(Es[i - 1] - Es[i]))
                dmin = min(abs(Es[i + 1] - Es[i]), abs(Es[i - 1] - Es[i]))
                tp, tm = band[i + 1] - m, m - band[i - 1]
                tau = tp * dmax + tm * dmin if Es[i + 1] > Es[i - 1] else tp * dmin + tm * dmax
            tau = tau - (tau * m).sum(-1, keepdim=True) * m
            tau = tau / (tau.norm() + 1e-30)
            f = hs[i] - (hs[i] * m).sum(-1, keepdim=True) * m
            fpar = (f * tau).sum() * tau
            if ci and i == i_max:
                ft = f - 2.0 * fpar
            else:
                ft = f - fpar + k_spring * (d[i] - d[i - 1]) * (math.sqrt(n_cells) * tau)
            F[i] = ft
            fmax = max(fmax, ft.abs().max().item())
        if not ci and it > 30 and fmax < ci_gate:
            ci = True
            for v in vel:
                v.zero_()
            dt = min(dt, 1.0e-4)
        if verbose and (it == 1 or it % 100 == 0):
            print(f"  [{label}] it={it} max|F|={fmax:.1f} CI={ci} t={time.perf_counter() - t0:.0f}s", flush=True)
        if fmax < tol_f:
            conv = True
            break
        if not math.isfinite(fmax) or not all(math.isfinite(e) for e in Es):
            break  # diverged: the band is returned as is and flagged
        P = sum((F[i] * vel[i]).sum().item() for i in range(1, n - 1))
        if P > 0:
            vn = math.sqrt(sum((vel[i] ** 2).sum().item() for i in range(1, n - 1)))
            fn = math.sqrt(sum((F[i] ** 2).sum().item() for i in range(1, n - 1))) + 1e-30
            for i in range(1, n - 1):
                vel[i] = (1 - alpha) * vel[i] + alpha * (vn / fn) * F[i]
            n_pos += 1
            if n_pos > n_min:
                dt, alpha = min(dt * f_inc, dt_max), alpha * f_alpha
        else:
            for i in range(1, n - 1):
                vel[i].zero_()
            dt, alpha, n_pos = max(dt * f_dec, dt_min), alpha_start, 0
        for i in range(1, n - 1):
            vel[i] = vel[i] + dt * F[i]
            step = dt * vel[i]
            sn = step.norm(dim=-1, keepdim=True)
            step = torch.where(sn > 0.12, step * (0.12 / (sn + 1e-30)), step)
            mn = band[i] + step
            mn = mn / (mn.norm(dim=-1, keepdim=True) + 1e-30)
            band[i] = mn
            vel[i] = vel[i] - (vel[i] * mn).sum(-1, keepdim=True) * mn
    Es = [field(m)[1] for m in band]
    diverged = not all(math.isfinite(e) for e in Es) or not math.isfinite(fmax)
    return band, Es, {"converged": conv, "diverged": diverged, "iters": it, "final_maxF_A_per_m": round(fmax, 1),
                      "climbing_image": ci, "time_s": round(time.perf_counter() - t0, 1), "solver": "GNEB-FIRE"}


def geodesic_band(m0, m1, n_images):
    """Images on the per-cell great circle from m0 to m1. Antipodal cells (m1 = -m0, e.g. under a group element
    that reverses them) have no unique geodesic; they are rotated by pi about an axis perpendicular to m0 built from a
    fixed reference direction (a second reference where m0 is parallel to the first, since by the hairy-ball theorem no
    single continuous rule exists)."""
    dot = (m0 * m1).sum(-1, keepdim=True).clamp(-1.0, 1.0)
    anti = dot < -1 + 1e-6
    ex = torch.tensor([1.0, 0.0, 0.0], dtype=m0.dtype).expand_as(m0)
    ey = torch.tensor([0.0, 1.0, 0.0], dtype=m0.dtype).expand_as(m0)
    ax = torch.cross(m0, ex, dim=-1)
    ax = torch.where(ax.norm(dim=-1, keepdim=True) < 0.5, torch.cross(m0, ey, dim=-1), ax)
    ax = ax / ax.norm(dim=-1, keepdim=True)
    perp = torch.cross(ax, m0, dim=-1)  # unit, perpendicular to m0
    om = torch.acos(dot.clamp(-1 + 1e-12, 1 - 1e-12))
    so = torch.sin(om)
    band = []
    for k in range(n_images):
        s = k / (n_images - 1)
        w0 = torch.where(so < 1e-6, torch.full_like(so, 1 - s), torch.sin((1 - s) * om) / so.clamp_min(1e-12))
        w1 = torch.where(so < 1e-6, torch.full_like(so, s), torch.sin(s * om) / so.clamp_min(1e-12))
        m = w0 * m0 + w1 * m1
        m = torch.where(anti, math.cos(math.pi * s) * m0 + math.sin(math.pi * s) * perp, m)
        band.append(m / m.norm(dim=-1, keepdim=True))
    band[0], band[-1] = m0.clone(), m1.clone()
    return band
