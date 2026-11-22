"""One vortex class of the 6:5:4 cuboid (pair 0-b of the vortex orbit): recompute the band and test for a Bloch point
(solid-angle charge of each cube of eight neighbouring cells, largest neighbour angle). Usage: chen_bloch_point.py nx b"""
import sys, json, numpy as np
import os
from _paths import DATA, FIGURES, MERRILL, RUNS  # noqa: F401
os.makedirs(RUNS / "chen", exist_ok=True)
os.chdir(RUNS / "chen")  # inputs and outputs of the cuboid runs live in <RUNS>/chen
from symbarrier.backends.magnumnp_backend import MagnumNPBackend, MAGNETITE_MERRILL_20C
from symbarrier import network as net, group as grp
nx = int(sys.argv[1]); N = (nx, nx * 5 // 6, nx * 4 // 6)
be = MagnumNPBackend(N, (105e-9 / nx,) * 3, material=MAGNETITE_MERRILL_20C)
S = np.load(f"chen_states_{nx}.npz")
G = grp.with_time_reversal(be.point_group()); nodes, fam, perm = net.build_nodes([S["vortex"]], G, be.action(), be.weights)
D = net.distance_matrix(nodes, be.weights); b = int(sys.argv[2])
h = be.neb(nodes[0], nodes[b], n_images=19, tag=f"bp{b}")
prof = np.asarray(h["profile_kT"]); k = int(np.argmax(prof))
def neighbour_max_angle(m):
    M = m.reshape(*N, 3); best = 0
    for ax in range(3):
        d = np.clip((M.take(range(1, N[ax]), ax) * M.take(range(0, N[ax] - 1), ax)).sum(-1), -1, 1)
        best = max(best, np.degrees(np.arccos(d.min())))
    return best
def solid_angle_charge(M):
    # topological charge of each unit cube of 8 cell centres (12 triangles on 6 faces), Berg-Luscher
    def omega(a, b, c):
        num = np.einsum('...i,...i', a, np.cross(b, c)); den = 1 + (a*b).sum(-1) + (b*c).sum(-1) + (c*a).sum(-1)
        return 2 * np.arctan2(num, den)
    c = {(i, j, l): M[i:N[0]-1+i, j:N[1]-1+j, l:N[2]-1+l] for i in (0, 1) for j in (0, 1) for l in (0, 1)}
    faces = [((0,0,0),(0,1,0),(0,1,1),(0,0,1)), ((1,0,0),(1,0,1),(1,1,1),(1,1,0)),
             ((0,0,0),(0,0,1),(1,0,1),(1,0,0)), ((0,1,0),(1,1,0),(1,1,1),(0,1,1)),
             ((0,0,0),(1,0,0),(1,1,0),(0,1,0)), ((0,0,1),(0,1,1),(1,1,1),(1,0,1))]
    q = 0
    for f in faces:
        a, b_, c_, d = (c[v] for v in f)
        q = q + omega(a, b_, c_) + omega(a, c_, d)
    return q / (4 * np.pi)
qs = [np.abs(solid_angle_charge(np.asarray(m).reshape(*N, 3))).max() for m in h["path"]]
out = {"nx": nx, "pair": [0, b], "rms_deg": round(D[0, b], 1), "saddle_kT": round(prof.max() - prof[0], 2), "i_max": k,
       "max_neighbour_angle_at_saddle_deg": round(neighbour_max_angle(np.asarray(h["path"][k])), 1),
       "max_neighbour_angle_ends_deg": round(neighbour_max_angle(np.asarray(h["path"][0])), 1),
       "max_abs_cube_charge_per_image": [round(float(x), 2) for x in qs], "info": h["info"]}
print(json.dumps(out)); json.dump(out, open(f"chen_bp_{nx}_0-{b}.json", "w"), indent=1, default=str)
