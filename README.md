# symbarrier

Symmetry quotient networks and verifiable upper bounds for energy barriers in micromagnetic models of single grains, on any solver.

Accompanies: Xiang Zhao and Xiangyu Zhao, *Symmetry quotient networks: automatic reduction and verifiable upper bounds for energy barriers in micromagnetic models of magnetic grains* (submitted to *Computers & Geosciences*). Questions and bug reports: please open an issue.

## Overview

The thermal stability of a magnetic grain is set by the energy barriers between its local energy minima. A grain with many minima needs a barrier for every pair of states, and each barrier normally costs one nudged-elastic-band (NEB) calculation. `symbarrier` detects the symmetry group of the grain from its mesh, combines it with time reversal, and computes the barrier network on the quotient: states and transitions related by symmetry share one path, so one NEB is run per class of state pairs. Every path maximum is an upper bound on its barrier, and every reported transition, reversal and escape barrier is the lowest bound over all computed paths, stored together with the path that realises it. The output is a barrier network with escape barriers and a rate matrix that can feed Markov or Boltzmann models of remanence acquisition.

## Quick start

```
pip install .
python examples/classes_without_merrill.py
```

This needs no micromagnetic solver. For a 100 nm truncated octahedron with a <111> vortex it prints:

```
nodes in mesh: 597; point group |G| = 48; with time reversal 96
orbit of m0: 16 states; -m0 is state 1 (0.00 deg)
edge classes from m0 (one NEB each): 6
...
unreduced pairwise network: 120 NEBs
```

The 16 symmetry-related states form 120 pairs; these fall into 7 classes, so 7 paths replace 120. With a solver installed, `symbarrier-network` runs the paths and assembles the network (see Use).

## What it does

Given a discretised grain (tetrahedral mesh or regular grid with a mask) and one or more relaxed magnetisation states, `symbarrier`

1. detects the point group *G* of the grain in the crystal frame (the elements of O_h that map the mesh onto itself) and adds time reversal *T* (m → −m);
2. generates the orbit of every seed state under *G* × {1, *T*}, with improper elements acting on the axial vector m with det *R*; the permutation action on all states follows from the group multiplication table (|G| field interpolations per seed);
3. builds a short-edge graph between the states and classifies its edges into orbits of the group on unordered pairs (for a single seed: orbits of the stabiliser of m0);
4. runs one minimum-energy-path calculation per edge class through a solver backend (MERRILL or magnum.np);
5. assembles the full network, and returns the minimax barrier m0 → −m0 (an upper bound by the mountain-pass property), the escape barrier of every state, and the Arrhenius rate matrix;
6. relaxes interior minima of every band and, in `explore`, adds new states as new families and repeats until none is new, reusing every computed path (endpoint cache).

The reduction is up to 96-fold for O_h, 32-fold for D4h, 4-fold for any centrosymmetric shape (inversion survives every crystal orientation) and 2-fold for a grain without point symmetry.

## Install

```
pip install .            # numpy, scipy
pip install .[test]      # + pytest
```

A solver is needed only for relaxation and paths:

- `MerrillBackend` (finite elements): MERRILL (https://bitbucket.org/wynwilliams/merrill), called as an external program, not bundled; tested with 2.2.1 (key=value script syntax).
- `MagnumNPBackend` (finite differences, optional mask, cubic K1 and K2): `pip install magnumnp torch`; tested with magnum.np 2.2.0. Paths use the bundled geodesic NEB with FIRE and a climbing image (`backends/gneb.py`).

Another solver fits behind the four-operation interface in `backends/base.py` (`energy`, `relax`, `neb`, sample points and weights).

## Use

```
symbarrier-network merrill /path/to/merrill grain.pat work/ m0.dat [seed2.dat ...] [--explore] [--K 4] [--K-cross 2] [--extra "Set: K2 = 0"]
symbarrier-network grid 24,24,40 5 work/ m0.npy [seed2.npy ...] [--mask mask.npy] [--material merrill20C] [--explore]
```

Outputs in `work/`: `network.json` (orbits, classes, every path with stop reason and cost, minimax, escape barriers, timing) and `rate_matrix_per_s.csv`; with `--explore`, one folder per round and `explore.json`.

From Python, with arrays of unit vectors (n, 3) in the backend's sample order:

```python
from symbarrier.backends import MerrillBackend, MagnumNPBackend
from symbarrier.workflow import run_network, explore
be = MerrillBackend("merrill", "grain.pat", "work")          # or MagnumNPBackend((24, 24, 40), (5e-9,) * 3, mask=None)
res = explore(be, [m0, m1], outdir="work/out")              # run_network(...) for a single pass
```

## Without a solver

The symmetry bookkeeping runs on its own and tells you, before any micromagnetic run, how many NEB calculations a grain needs:

```
python examples/classes_without_merrill.py                    # built-in 100 nm truncated octahedron, <111> vortex
python examples/classes_without_merrill.py grain.pat m0.dat   # your mesh and any saved state (x y z mx my mz per node)
```

For the built-in case: |G| = 48 (96 with time reversal), 16 states in the orbit of m0, 6 classes of edges from m0 besides the direct reversal; all 120 pairs of the orbit fall into 7 classes.

## Tests

`pytest tests` runs without a solver (the grid-backend test is skipped if magnum.np is absent): point-group detection on a truncated octahedron (48), a square prism (16) and a rotated truncated octahedron (2, inversion only); orbit sizes and class counts for a uniform ⟨111⟩ state (8 states, 2 classes) and a ⟨111⟩ vortex (16 states, 6 classes; 8 with proper rotations only); minimax and rate-matrix algebra; exact cell-permutation action and orbit on a masked grid.

## Reproducing the paper

`paper/` holds the scripts behind every table, figure and supplementary section and the JSON/CSV outputs of all runs; the four figures can be regenerated from these outputs without a solver. See `paper/README.md`.

## Hardware

Any machine that runs the chosen solver; all paper timings are single-core.

## How to cite

Xiang Zhao and Xiangyu Zhao. Symmetry quotient networks: automatic reduction and verifiable upper bounds for energy barriers in micromagnetic models of magnetic grains. Submitted to *Computers & Geosciences*.

The software metadata are in `CITATION.cff`; GitHub shows them under "Cite this repository".

## License

MIT (see `LICENSE`). MERRILL itself is AGPL-3.0 and is only invoked as a separate program; magnum.np is used as an optional Python dependency.
