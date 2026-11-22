#!/usr/bin/env python3
"""PATRAN meshes for MERRILL: BCC lattice clipped to the shape, Delaunay tetrahedralisation, sliver filter.

  make_meshes.py box <lx_nm> <ly_nm> <lz_nm> <h_nm> <out.pat>     rectangular prism (half-lengths multiples of h)
  make_meshes.py to  <size_nm> <n_sub> <out.pat>                    truncated octahedron, h = size / (4 n_sub)
"""
import sys

from _paths import RUNS  # noqa: F401  (puts the package on the path)
from symbarrier.geometry import write_patran
from symbarrier.meshes import box, truncated_octahedron


def main():
    kind = sys.argv[1]
    if kind == "box":
        lx, ly, lz, h = (float(v) for v in sys.argv[2:6])
        out = sys.argv[6]
        c, t = box(lx, ly, lz, h)
        title = f"box {sys.argv[2]}x{sys.argv[3]}x{sys.argv[4]} nm (h={h:.2f} nm)"
    elif kind == "to":
        size, n_sub = float(sys.argv[2]), int(sys.argv[3])
        out = sys.argv[4]
        c, t = truncated_octahedron(size, n_sub)
        title = f"truncated octahedron {size:g} nm (h={size / (4 * n_sub):.2f} nm)"
    else:
        raise SystemExit(__doc__)
    write_patran(out, c, t, title=title)
    print(f"{out}: nodes={len(c)} tets={len(t)}")


if __name__ == "__main__":
    main()
