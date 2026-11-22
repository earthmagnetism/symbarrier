"""Quotient barrier network.

Nodes are the G x {1,T} orbits of one or more seed states; candidate edges form a short-edge graph; edge
classes are the group orbits on unordered node pairs; one NEB per class populates every edge of its orbit.
The minimax over the network is an upper bound on the barrier between two nodes (mountain-pass property).
"""
import heapq

import numpy as np

from .geometry import rms_angle_deg


def _key(g):
    return (tuple(np.rint(g[0]).astype(int).ravel()), int(g[1]))


def multiplication_table(G):
    """table[i][j] = index of G[i] G[j]."""
    idx = {_key(g): k for k, g in enumerate(G)}
    return [[idx[_key((gi[0] @ gj[0], gi[1] * gj[1]))] for gj in G] for gi in G]


def build_nodes(seeds, G, act, w, tol_deg=5.0):
    """Orbits of every seed under G (list of (R,t)). Returns nodes (list of fields), family index per node,
    and the permutation table perm[g][a] = index of g . node_a.

    Only |G| interpolations per seed are needed: node a is h_a . seed_f, so g . node_a = (g h_a) . seed_f and the
    permutation table follows from the group multiplication table."""
    nodes, fam, rep, node_of = [], [], [], []
    for i, m in enumerate(seeds):
        row = []
        for gi, g in enumerate(G):
            mi = act(g, m)
            k = next((j for j, s in enumerate(nodes) if rms_angle_deg(mi, s, w) < tol_deg), None)
            if k is None:
                nodes.append(mi)
                fam.append(i)
                rep.append(gi)
                k = len(nodes) - 1
            row.append(k)
        node_of.append(row)
    mult = multiplication_table(G)
    perm = [[node_of[fam[a]][mult[gi][rep[a]]] for a in range(len(nodes))] for gi in range(len(G))]
    return nodes, fam, perm


def distance_matrix(nodes, w):
    n = len(nodes)
    D = np.zeros((n, n))
    for a in range(n):
        for b in range(a + 1, n):
            D[a, b] = D[b, a] = rms_angle_deg(nodes[a], nodes[b], w)
    return D


def short_edge_graph(D, fam, K=4, K_cross=2, always=()):
    """Join every node to its K nearest nodes and to its K_cross nearest nodes of every other family,
    plus any pairs listed in `always` (e.g. the direct reversal edge)."""
    fam = np.asarray(fam)
    edges = set(tuple(sorted(e)) for e in always)
    for a in range(len(D)):
        for b in np.argsort(D[a])[1:K + 1]:
            edges.add(tuple(sorted((a, int(b)))))
        for f in set(fam.tolist()) - {int(fam[a])}:
            idx = np.where(fam == f)[0]
            for b in idx[np.argsort(D[a, idx])[:K_cross]]:
                edges.add(tuple(sorted((a, int(b)))))
    return sorted(edges)


def edge_classes(edges, perm):
    """Group orbits on unordered pairs. Returns {representative: [edges of the short graph in the class]} and
    {representative: full orbit of the representative (all images, also outside the short graph)}."""
    canon = lambda e: min(tuple(sorted((p[e[0]], p[e[1]]))) for p in perm)
    classes = {}
    for e in edges:
        classes.setdefault(canon(e), []).append(e)
    orbit = {r: sorted({tuple(sorted((p[r[0]], p[r[1]]))) for p in perm}) for r in classes}
    return classes, orbit


def stabiliser_classes(n_nodes, m0_node, neg_node, perm):
    """Single-family special case: classes of edges (m0, a) = orbits of Stab(m0) on the other nodes."""
    stab = [p for p in perm if p[m0_node] == m0_node]
    classes, seen = {}, set()
    for k in range(n_nodes):
        if k in (m0_node, neg_node) or k in seen:
            continue
        members = sorted({p[k] for p in stab})
        seen.update(members)
        classes[min(members)] = members
    return classes


def minimax(adj, src, E):
    """Minimax path cost from src: best[v] = min over paths of the max saddle energy (absolute scale)."""
    best = {src: E[src]}
    pq = [(E[src], src)]
    while pq:
        cost, u = heapq.heappop(pq)
        if cost > best.get(u, np.inf):
            continue
        for v, s in adj.get(u, {}).items():
            nc = max(cost, s)
            if nc < best.get(v, np.inf):
                best[v] = nc
                heapq.heappush(pq, (nc, v))
    return best


def assemble(orbit, saddle_E):
    """Weighted adjacency from class orbits and the saddle energy of each class representative."""
    adj = {}
    for r, imgs in orbit.items():
        for a, b in imgs:
            for x, y in ((a, b), (b, a)):
                adj.setdefault(x, {})[y] = min(adj.get(x, {}).get(y, np.inf), saddle_E[r])
    return adj


def rate_matrix(adj, E, f0=1e9):
    """Arrhenius rates k_ab = f0 exp(-(E_s - E_a)), energies in kT units. Returns (n, n) array."""
    n = len(E)
    K = np.zeros((n, n))
    for a, nb in adj.items():
        for b, s in nb.items():
            K[a, b] = f0 * np.exp(-(s - E[a]))
    return K
