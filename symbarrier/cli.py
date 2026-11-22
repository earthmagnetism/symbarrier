"""Command line.

  symbarrier-network merrill <merrill_bin> <mesh.pat> <workdir> <m0.dat> [seed2.dat ...]
  symbarrier-network grid    <N_x,N_y,N_z> <dx_nm> <workdir> <m0.npy> [seed2.npy ...] [--mask mask.npy]

MERRILL seeds are .dat files (x y z mx my mz per node, MERRILL node order). Grid seeds are .npy arrays of shape
(N_x, N_y, N_z, 3). The first seed is m0. --explore adds intermediate minima found on bands as new families until
none is new.

Assumptions: cubic anisotropy (the group is searched in O_h), crystal axes along the x, y, z axes of the mesh or grid
(rotate the mesh for another orientation), zero applied field. Default material: magnetite at 20 C; MERRILL accepts any
material line (--material, --extra); the grid backend takes the presets 'fd' or 'merrill20C', or any dict of Ms, A, K1,
K2 through the Python API.
"""
import argparse
import json

import numpy as np

from .workflow import explore, run_network


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="backend", required=True)
    pm = sub.add_parser("merrill")
    pm.add_argument("merrill_bin")
    pm.add_argument("mesh")
    pg = sub.add_parser("grid")
    pg.add_argument("N")
    pg.add_argument("dx_nm", type=float)
    pg.add_argument("--mask")
    pg.add_argument("--material", choices=["fd", "merrill20C"], default="fd")
    for q in (pm, pg):
        q.add_argument("workdir")
        q.add_argument("seeds", nargs="+")
        q.add_argument("--K", type=int, default=4, help="nearest nodes joined in the short-edge graph")
        q.add_argument("--K-cross", type=int, default=2, help="nearest nodes of every other family")
        q.add_argument("--proper-only", action="store_true", help="restrict G to proper rotations")
        q.add_argument("--explore", action="store_true", help="adaptive loop over intermediate minima")
    pm.add_argument("--max-evals", type=int, default=1000)
    pm.add_argument("--material", default="Type=Magnetite, TempC=20")
    pm.add_argument("--extra", action="append", default=[], help="extra MERRILL lines, e.g. 'Set: K2 = 0'")
    a = p.parse_args(argv)
    if a.backend == "merrill":
        from .backends.merrill_backend import MerrillBackend
        be = MerrillBackend(a.merrill_bin, a.mesh, a.workdir, material=a.material, extra=a.extra, max_evals=a.max_evals)
        seeds = [np.loadtxt(s)[:, 3:6] for s in a.seeds]
    else:
        from .backends.magnumnp_backend import MAGNETITE_FD, MAGNETITE_MERRILL_20C, MagnumNPBackend
        N = tuple(int(x) for x in a.N.split(","))
        mask = None if a.mask is None else np.load(a.mask).astype(bool)
        be = MagnumNPBackend(N, (a.dx_nm * 1e-9,) * 3, mask=mask,
                             material=MAGNETITE_FD if a.material == "fd" else MAGNETITE_MERRILL_20C)
        seeds = [be.from_grid(__import__("torch").tensor(np.load(s), dtype=__import__("torch").float64)) for s in a.seeds]
    seeds = [s / np.linalg.norm(s, axis=1, keepdims=True) for s in seeds]
    kw = dict(K=a.K, K_cross=a.K_cross, proper_only=a.proper_only)
    res = explore(be, seeds, outdir=a.workdir, **kw) if a.explore else run_network(be, seeds, outdir=a.workdir, **kw)
    drop = {"hops", "paths", "split_candidates", "nodes", "node_family", "families"}
    print(json.dumps({k: v for k, v in res.items() if k not in drop}, indent=1, default=float))


if __name__ == "__main__":
    main()
