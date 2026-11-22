"""Finite-difference (magnum.np) definitions shared by the fd_* scripts: magnetite material, cubic anisotropy,
geodesic band construction and the GNEB-FIRE solver (run_gneb_fire)."""
import time, math, json, logging
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from magnumnp import Mesh, State, DemagField, ExchangeField, MinimizerBB

logging.getLogger('magnum.np').setLevel(logging.WARNING)
MU0 = 4.0 * math.pi * 1e-7
KBT_300 = 1.380649e-23 * 300.0

class ExactCubicAnisotropy:
    """Cubic magnetocrystalline anisotropy field (room-temperature magnetite, K1 = -1.33e4 J/m^3, easy axes <111>)."""
    def __init__(self, K1=-1.33e4, Ms=4.80e5):
        self.K1 = float(K1)
        self.Ms = float(Ms)

    def h(self, state):
        m = state.m
        mx, my, mz = m[..., 0], m[..., 1], m[..., 2]
        mx2, my2, mz2 = mx * mx, my * my, mz * mz
        pref = -2.0 * self.K1 / (MU0 * self.Ms)
        return torch.stack([
            pref * mx * (my2 + mz2),
            pref * my * (mz2 + mx2),
            pref * mz * (mx2 + my2),
        ], dim=-1)

    def E(self, state):
        m = state.m
        mx2, my2, mz2 = m[..., 0] ** 2, m[..., 1] ** 2, m[..., 2] ** 2
        dV = float(state.mesh.dx[0] * state.mesh.dx[1] * state.mesh.dx[2])
        return self.K1 * dV * torch.sum(mx2 * my2 + my2 * mz2 + mz2 * mx2)

def compute_helicity(m, dx):
    dmx_dy = (m[1:-1, 2:, 1:-1, 0] - m[1:-1, :-2, 1:-1, 0]) / (2.0 * dx[1])
    dmx_dz = (m[1:-1, 1:-1, 2:, 0] - m[1:-1, 1:-1, :-2, 0]) / (2.0 * dx[2])
    dmy_dx = (m[2:, 1:-1, 1:-1, 1] - m[:-2, 1:-1, 1:-1, 1]) / (2.0 * dx[0])
    dmy_dz = (m[1:-1, 1:-1, 2:, 1] - m[1:-1, 1:-1, :-2, 1]) / (2.0 * dx[2])
    dmz_dx = (m[2:, 1:-1, 1:-1, 2] - m[:-2, 1:-1, 1:-1, 2]) / (2.0 * dx[0])
    dmz_dy = (m[1:-1, 2:, 1:-1, 2] - m[1:-1, :-2, 1:-1, 2]) / (2.0 * dx[1])
    curl_x = dmz_dy - dmy_dz
    curl_y = dmx_dz - dmz_dx
    curl_z = dmy_dx - dmx_dy
    mc = m[1:-1, 1:-1, 1:-1]
    h_dens = mc[..., 0] * curl_x + mc[..., 1] * curl_y + mc[..., 2] * curl_z
    return (h_dens.sum() * dx[0] * dx[1] * dx[2]).item()

def rodrigues_matrix(axis, angle, device, dtype):
    a = torch.tensor(axis, device=device, dtype=dtype)
    a = a / (a.norm() + 1e-30)
    K = torch.tensor([
        [0.0, -a[2], a[1]],
        [a[2], 0.0, -a[0]],
        [-a[1], a[0], 0.0]
    ], device=device, dtype=dtype)
    I = torch.eye(3, device=device, dtype=dtype)
    return I + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)

def build_so3_covariant_band(m0, m1, dx, n_images, rot_axis):
    Nx, Ny, Nz, _ = m0.shape
    device, dtype = m0.device, m0.dtype
    hx, hy, hz = 0.5 * (Nx - 1) * dx[0], 0.5 * (Ny - 1) * dx[1], 0.5 * (Nz - 1) * dx[2]
    xs = torch.linspace(-hx, hx, Nx, device=device, dtype=dtype)
    ys = torch.linspace(-hy, hy, Ny, device=device, dtype=dtype)
    zs = torch.linspace(-hz, hz, Nz, device=device, dtype=dtype)
    Z, Y, X = torch.meshgrid(zs, ys, xs, indexing='ij')
    r_grid = torch.stack([X, Y, Z], dim=-1)
    inp = m0.permute(3, 2, 1, 0).unsqueeze(0)

    band = [m0.clone()]
    for k in range(1, n_images - 1):
        s = k / (n_images - 1)
        R = rodrigues_matrix(rot_axis, math.pi * s, device, dtype)
        r_src = torch.einsum('ij,...j->...i', R.T, r_grid)
        grid_norm = torch.stack([
            r_src[..., 0] / (hx + 1e-30),
            r_src[..., 1] / (hy + 1e-30),
            r_src[..., 2] / (hz + 1e-30),
        ], dim=-1).unsqueeze(0)
        sampled = F.grid_sample(inp, grid_norm, mode='bilinear', padding_mode='border', align_corners=True)
        m_samp = sampled.squeeze(0).permute(3, 2, 1, 0)
        m_rot = torch.einsum('ij,...j->...i', R, m_samp)
        band.append(m_rot / (m_rot.norm(dim=-1, keepdim=True) + 1e-30))
    band.append(m1.clone())
    return band

def build_slerp_random_band(m0, m1, n_images, sigma=0.2, seed=0):
    g = torch.Generator(device=m0.device).manual_seed(seed)
    band = [m0.clone()]
    for k in range(1, n_images - 1):
        s = k / (n_images - 1)
        dot = (m0 * m1).sum(dim=-1, keepdim=True).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        omega = torch.acos(dot)
        sin_om = torch.sin(omega)
        small = sin_om < 1e-4
        w0 = torch.where(small, 1.0 - s, torch.sin((1.0 - s) * omega) / (sin_om + 1e-12))
        w1 = torch.where(small, s, torch.sin(s * omega) / (sin_om + 1e-12))
        m = w0 * m0 + w1 * m1
        noise = torch.randn(m.shape, generator=g, device=m.device, dtype=m.dtype)
        m = m + (sigma * math.sin(math.pi * s)) * noise
        band.append(m / (m.norm(dim=-1, keepdim=True) + 1e-30))
    band.append(m1.clone())
    return band

def rms_geodesic_dist(m_a, m_b):
    dot = (m_a * m_b).sum(dim=-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.sqrt(torch.mean(torch.acos(dot) ** 2)).item()

def max_neighbor_angle_deg(m):
    dx_dot = (m[:-1, :, :] * m[1:, :, :]).sum(dim=-1).clamp(-1.0, 1.0)
    dy_dot = (m[:, :-1, :] * m[:, 1:, :]).sum(dim=-1).clamp(-1.0, 1.0)
    dz_dot = (m[:, :, :-1] * m[:, :, 1:]).sum(dim=-1).clamp(-1.0, 1.0)
    min_dot = min(dx_dot.min().item(), dy_dot.min().item(), dz_dot.min().item())
    return math.degrees(math.acos(max(-1.0, min(1.0, min_dot))))

def run_gneb_fire(state, terms, band, label, max_iter=600, k_spring=2e5, tol_f=2e3, ci_gate=1.5e4, dt_max=1.8e-4, verbose=False):
    n_img = len(band)
    n_cells = band[0].shape[0] * band[0].shape[1] * band[0].shape[2]
    dt, dt_min = min(1.0e-4, dt_max), 2.0e-5
    alpha_start, alpha = 0.15, 0.15
    f_inc, f_dec, f_alpha, n_min = 1.05, 0.5, 0.99, 5
    n_pos = 0
    ci_active = False
    vel = [torch.zeros_like(m) for m in band]
    t0 = time.perf_counter()
    converged = False
    hist_it, hist_f, hist_b = [], [], []

    for it in range(1, max_iter + 1):
        fields, energies = [], []
        for m in band:
            state.m = m
            fields.append(sum(t.h(state) for t in terms))
            energies.append(float(sum(t.E(state) for t in terms)))
        i_max = max(range(1, n_img - 1), key=lambda i: energies[i])
        dists = [rms_geodesic_dist(band[i], band[i + 1]) for i in range(n_img - 1)]
        forces = [None] * n_img
        max_f_norm = 0.0

        for i in range(1, n_img - 1):
            m = band[i]
            if energies[i + 1] > energies[i] > energies[i - 1]:
                tau = band[i + 1] - m
            elif energies[i - 1] > energies[i] > energies[i + 1]:
                tau = m - band[i - 1]
            else:
                dE_max = max(abs(energies[i + 1] - energies[i]), abs(energies[i - 1] - energies[i]))
                dE_min = min(abs(energies[i + 1] - energies[i]), abs(energies[i - 1] - energies[i]))
                t_plus, t_minus = band[i + 1] - m, m - band[i - 1]
                tau = (t_plus * dE_max + t_minus * dE_min) if energies[i + 1] > energies[i - 1] else (t_plus * dE_min + t_minus * dE_max)
            tau = tau - (tau * m).sum(dim=-1, keepdim=True) * m
            tau = tau / (tau.norm() + 1e-30)
            f_eff = fields[i] - (fields[i] * m).sum(dim=-1, keepdim=True) * m
            f_par = (f_eff * tau).sum() * tau
            f_perp = f_eff - f_par
            if ci_active and i == i_max:
                f_tot = f_eff - 2.0 * f_par
            else:
                f_spring = k_spring * (dists[i] - dists[i - 1]) * (math.sqrt(n_cells) * tau)
                f_tot = f_perp + f_spring
            forces[i] = f_tot
            max_f_norm = max(max_f_norm, f_tot.abs().max().item())

        if (not ci_active) and it > 30 and max_f_norm < ci_gate:
            ci_active = True
            for v in vel:
                v.zero_()
            dt = min(dt, 1.0e-4)

        dE_kBT = (max(energies) - min(energies[0], energies[-1])) / KBT_300
        if it == 1 or it % 5 == 0 or max_f_norm < tol_f:
            hist_it.append(it)
            hist_f.append(max_f_norm)
            hist_b.append(dE_kBT)

        if (verbose and (it == 1 or it % 100 == 0)) or max_f_norm < tol_f:
            if verbose or max_f_norm < tol_f:
                print(f'  [{label}] iter={it:4d} | max|F|={max_f_norm:8.1f} | barrier={dE_kBT:7.1f} kBT | CI={ci_active} | t={time.perf_counter()-t0:.1f}s')
            if max_f_norm < tol_f:
                converged = True
                break

        P = sum((forces[i] * vel[i]).sum().item() for i in range(1, n_img - 1))
        if P > 0.0:
            v_norm = math.sqrt(sum((vel[i] ** 2).sum().item() for i in range(1, n_img - 1)))
            f_norm_l2 = math.sqrt(sum((forces[i] ** 2).sum().item() for i in range(1, n_img - 1))) + 1e-30
            for i in range(1, n_img - 1):
                vel[i] = (1.0 - alpha) * vel[i] + alpha * (v_norm / f_norm_l2) * forces[i]
            n_pos += 1
            if n_pos > n_min:
                dt = min(dt * f_inc, dt_max)
                alpha *= f_alpha
        else:
            for i in range(1, n_img - 1):
                vel[i].zero_()
            dt = max(dt * f_dec, dt_min)
            alpha = alpha_start
            n_pos = 0

        for i in range(1, n_img - 1):
            vel[i] = vel[i] + dt * forces[i]
            step = dt * vel[i]
            sn = step.norm(dim=-1, keepdim=True)
            step = torch.where(sn > 0.12, step * (0.12 / (sn + 1e-30)), step)
            m_next = band[i] + step
            m_next = m_next / (m_next.norm(dim=-1, keepdim=True) + 1e-30)
            band[i] = m_next
            vel[i] = vel[i] - (vel[i] * m_next).sum(dim=-1, keepdim=True) * m_next

    elapsed = time.perf_counter() - t0
    e_ref = min(energies[0], energies[-1])
    profile_kBT = [(e - e_ref) / KBT_300 for e in energies]
    max_ang = max(max_neighbor_angle_deg(m) for m in band[1:-1])
    return {
        'label': label, 'conv': converged, 'iters': it,
        'time_s': round(elapsed, 2), 'barrier_kBT': round(max(profile_kBT), 2),
        'min_mid_kBT': round(min(profile_kBT[1:-1]), 2),
        'final_maxF': round(max_f_norm, 1), 'max_ang': round(max_ang, 1),
        'profile_kBT': [round(v, 2) for v in profile_kBT],
        'hist_it': hist_it, 'hist_f': hist_f, 'hist_b': hist_b
    }

def make_seed_vortex(N, dx, u_axis, device, dtype, Rc=20e-9):
    xs = (torch.arange(N[0], device=device, dtype=dtype) + 0.5) * dx[0] - 0.5 * N[0] * dx[0]
    ys = (torch.arange(N[1], device=device, dtype=dtype) + 0.5) * dx[1] - 0.5 * N[1] * dx[1]
    zs = (torch.arange(N[2], device=device, dtype=dtype) + 0.5) * dx[2] - 0.5 * N[2] * dx[2]
    X, Y, Z = torch.meshgrid(xs, ys, zs, indexing='ij')
    dr = torch.stack([X, Y, Z], dim=-1)
    u = torch.tensor(u_axis, device=device, dtype=dtype)
    u = u / u.norm()
    rho_vec = dr - (dr * u).sum(dim=-1, keepdim=True) * u
    rho = rho_vec.norm(dim=-1, keepdim=True)
    phi_hat = torch.cross(u.expand_as(rho_vec), rho_vec, dim=-1)
    phi_hat = phi_hat / (phi_hat.norm(dim=-1, keepdim=True) + 1e-30)
    w = torch.exp(-0.5 * (rho / Rc) ** 2)
    m = w * u + torch.sqrt((1.0 - w ** 2).clamp(min=0.0)) * phi_hat
    return m / (m.norm(dim=-1, keepdim=True) + 1e-30)

def setup_magnetite(N, dx):
    mesh = Mesh(N, dx)
    state = State(mesh)
    state.material = {'Ms': 4.80e5, 'A': 1.33e-11}
    state.m = state.Constant([1.0, 0.0, 0.0])
    terms = [DemagField(), ExchangeField(), ExactCubicAnisotropy(K1=-1.33e4, Ms=4.80e5)]
    minimizer = MinimizerBB(terms)
    return state, terms, minimizer

