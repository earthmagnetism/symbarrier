#!/usr/bin/env python3
"""Fig. 3: energy profiles."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from _paths import DATA, FIGURES, MERRILL, RUNS  # noqa: F401

n = json.load(open(f"{RUNS}/numbers.json"))
R = f"{RUNS}/benchmarks"
BLUE, ORANGE, AQUA, MAGENTA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#8a8a85"
INK, INK2 = "#1f1f1e", "#5c5c57"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": "#b5b4ad", "axes.linewidth": 0.8, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.labelcolor": INK, "text.color": INK, "font.family": "DejaVu Sans"})
lin = lambda p: np.linspace(0, 1, len(p))


def chain(segs):
    ys = []
    for i, seg in enumerate(segs):
        ys += list(seg if i == 0 else seg[1:])
    return np.linspace(0, 1, len(ys)), np.array(ys)


def style(ax, title, ylab=True):
    ax.set_title(title, loc="left", fontsize=9, color=INK, pad=6)
    ax.grid(axis="y", color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_xlim(0, 1)
    ax.set_xlabel("normalised path coordinate")
    if ylab:
        ax.set_ylabel(r"$E-E_{\rm ref}$ ($k_{\rm B}T_{300}$)")


def leg(ax, loc):
    ax.legend(frameon=False, fontsize=7, loc=loc, handlelength=2.2)


fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.2), dpi=300)
# (a) truncated octahedron, MERRILL; reference m0
ax = axes[0, 0]
band = np.array(n["merrill_TO"]["cov"]["profile_kBT"])
hop = json.load(open(f"{R}/BM3_TO_symdecomp/symdecomp.json"))["hops"]["9"]["profile_kBT"]
fdb = np.array(json.load(open(f"{R}/tofd/result_band112.json"))["profile_rel_m0_kBT"])
s, y = chain([hop, hop, hop])
ax.plot(lin(band), band, "-o", color=ORANGE, lw=2, ms=3.3, label=f"covariant single band, MERRILL ({band.max():.1f})")
ax.plot(s, y, "-", color=BLUE, lw=2, label=f"quotient-network chain, MERRILL ({max(hop):.1f})")
ax.plot(lin(fdb), fdb, "--s", color=AQUA, lw=1.4, ms=2.8, label=f"covariant single band, FD ({fdb.max():.1f})")
style(ax, r"(a) 150 nm truncated octahedron; ref. $\mathbf{m}_0$")
ax.set_ylim(-60, 1700)
leg(ax, "upper right")
# (b) cuboctahedron, MERRILL; reference m0
ax = axes[0, 1]
band = np.array(n["merrill_CO"]["cov"]["profile_kBT"])
hop = json.load(open(f"{R}/BM3_CO_symdecomp/symdecomp.json"))["hops"]["5"]["profile_kBT"]
dflt = np.array(n["merrill_CO"]["default"]["profile_kBT"])
s, y = chain([hop, hop[::-1]])
ax.plot(lin(band), band, "-o", color=ORANGE, lw=2, ms=3.3, label=f"covariant single band ({band.max():.1f})")
ax.plot(lin(dflt), dflt, ":", color=GREY, lw=1.6, label=f"MERRILL default single band ({dflt.max():.1f}, 'failed')")
ax.plot(s, y, "-", color=BLUE, lw=2, label=f"quotient-network chain ({max(hop):.1f})")
style(ax, r"(b) 100 nm cuboctahedron (MERRILL); ref. $\mathbf{m}_0$", ylab=False)
ax.set_ylim(-25, 600)
leg(ax, "upper right")
# (c) cube, FD; reference: state C
ax = axes[1, 0]
EC = 50.12
band = np.array(json.load(open(f"{R}/fd/result_so3.json"))["profile_kBT"]) + EC
symhop = np.array(json.load(open(f"{R}/fd_hops/result_m0_to_A.json"))["profile_rel_m0_kBT"]) + EC
esc = np.array(json.load(open(f"{R}/modepush/result_m0A_push0.3.json"))["profile_rel_m0_kBT"]) + EC
cc = json.load(open(f"{R}/FD3_CU_symdecomp/symdecomp.json"))["hops"]["6"]["profile_rel_m0_kBT"]
ax.plot(lin(band), band, "-o", color=ORANGE, lw=1.6, ms=2.8, label=r"covariant single band $\mathbf{m}_0\to-\mathbf{m}_0$")
ax.plot(np.linspace(0, 0.45, len(symhop)), symhop, "--", color=GREY, lw=1.6, label=r"hop $\mathbf{m}_0\to$A, index-2 saddle")
ax.plot(np.linspace(0, 0.45, len(esc)), esc, "-", color=MAGENTA, lw=1.8, label="same hop after mode push, index 1")
ax.plot(np.linspace(0.55, 1, len(cc)), cc, "-", color=BLUE, lw=2, label=f"quotient-network hop from C ({max(cc):.1f})")
ax.axhline(EC, color=GREY, lw=0.8, ls=(0, (3, 3)))
ax.text(0.02, EC - 62, r"level of $\mathbf{m}_0$", ha="left", color=INK2, fontsize=7)
style(ax, "(c) 150 nm cube (FD); ref. lowest state found, C")
ax.set_ylim(-25, 800)
leg(ax, "upper left")
# (d) prism, FD with the original FD constants; reference B2 (lowest state)
ax = axes[1, 1]
g = 1278.07
bx = np.array(json.load(open(f"{R}/fd_c2g_damped/result_band.json"))["profile_rel_m0_kBT"]) + g
bz = np.array(json.load(open(f"{R}/fd_c2g_z/result_band.json"))["profile_rel_m0_kBT"]) + g
hj = json.load(open(f"{R}/FD_PR_symdecomp/symdecomp.json"))
hp = np.array(hj["hops"][list(hj["hops"])[0]]["profile_rel_m0_kBT"])
s, y = chain([hp, hp[::-1]])
ax.plot(lin(bx), bx, "-o", color=ORANGE, lw=2, ms=3.3, label=f"covariant band about [100] ({bx.max():.1f})")
ax.plot(lin(bz), bz, "--s", color=AQUA, lw=1.4, ms=2.8, label=f"covariant band about [001] ({bz.max():.1f})")
ax.plot(s, y, "-", color=BLUE, lw=2, label=f"quotient-network chain, two 97° hops ({max(y):.1f})")
style(ax, "(d) 120×120×200 nm prism (FD); ref. B$_2$", ylab=False)
ax.set_ylim(-40, 1900)
leg(ax, "upper left")
fig.tight_layout()
fig.savefig(f"{FIGURES}/fig3_profiles.png", dpi=300)
print("saved")
