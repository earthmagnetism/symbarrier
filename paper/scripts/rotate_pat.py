#!/usr/bin/env python3
"""
Rotate a PATRAN mesh rigidly about its centroid (equivalent to rotating the cubic crystal axes,
which MERRILL keeps along x, y, z). Usage: rotate_pat.py <in.pat> <out.pat> <seed>
Writes the rotation matrix to <out.pat>.rot.json.
"""
import json
import sys

import numpy as np
from scipy.spatial.transform import Rotation


def main():
    src, dst, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    R = Rotation.random(random_state=seed).as_matrix()
    L = open(src).read().split("\n")
    # pass 1: centroid of nodes
    X = []
    i = 0
    while i < len(L):
        l = L[i]
        if len(l) >= 2 and l[:2].strip().isdigit() and int(l[:2]) == 1:
            X.append([float(L[i + 1][j:j + 16]) for j in (0, 16, 32)])
            i += 3
            continue
        i += 1
    cen = np.mean(X, axis=0)
    out, i = [], 0
    while i < len(L):
        l = L[i]
        if len(l) >= 2 and l[:2].strip().isdigit() and int(l[:2]) == 1:
            x = np.array([float(L[i + 1][j:j + 16]) for j in (0, 16, 32)])
            y = R @ (x - cen) + cen
            out += [l, "".join(f"{v:16.9e}" for v in y), L[i + 2]]
            i += 3
            continue
        out.append(l)
        i += 1
    open(dst, "w").write("\n".join(out))
    json.dump({"seed": seed, "R": R.round(6).tolist(), "centroid": cen.tolist()}, open(dst + ".rot.json", "w"), indent=1)
    print(dst, np.round(R, 3).tolist())


if __name__ == "__main__":
    main()
