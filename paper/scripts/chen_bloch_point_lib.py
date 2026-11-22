"""Topological charge of the field on the cubes of eight neighbouring grid cells (Bloch-point detection)."""
import numpy as np
def max_cube_charge(M):
    """Largest |topological charge| over unit cubes of 8 neighbouring cell centres (Berg-Luscher triangles)."""
    N = M.shape[:3]
    def omega(a, b, c):
        num = np.einsum('...i,...i', a, np.cross(b, c)); den = 1 + (a*b).sum(-1) + (b*c).sum(-1) + (c*a).sum(-1)
        return 2 * np.arctan2(num, den)
    c = {(i, j, l): M[i:N[0]-1+i, j:N[1]-1+j, l:N[2]-1+l] for i in (0, 1) for j in (0, 1) for l in (0, 1)}
    faces = [((0,0,0),(0,1,0),(0,1,1),(0,0,1)), ((1,0,0),(1,0,1),(1,1,1),(1,1,0)),
             ((0,0,0),(0,0,1),(1,0,1),(1,0,0)), ((0,1,0),(1,1,0),(1,1,1),(0,1,1)),
             ((0,0,0),(1,0,0),(1,1,0),(0,1,0)), ((0,0,1),(0,1,1),(1,1,1),(1,0,1))]
    q = 0
    for f in faces:
        a, b, cc, d = (c[v] for v in f)
        q = q + omega(a, b, cc) + omega(a, cc, d)
    return np.abs(q / (4 * np.pi)).max()
