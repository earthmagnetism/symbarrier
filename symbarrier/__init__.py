"""symbarrier: symmetry quotient networks and verifiable upper bounds for energy barriers in micromagnetic grain models,
on any solver behind a four-operation interface.

Modules
  geometry  mesh I/O (PATRAN, TecPlot), P1 interpolation, volume-weighted rms angle
  group     point-group detection from the mesh, action of G x {1,T} on axial fields
  network   orbits, short-edge graph, edge classes, minimax, rate matrix
  merrill   external-program driver for MERRILL (v2.2.1 syntax)
  backends  solver interface; MerrillBackend (FE) and MagnumNPBackend (FD, GNEB-FIRE)
  workflow  run_network (Algorithm 1) and explore (Algorithm 2, adaptive closure with path reuse)
"""
__version__ = "1.0.1"
