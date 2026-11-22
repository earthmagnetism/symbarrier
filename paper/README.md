# Reproducing the paper

Scripts and stored results for *Symmetry quotient networks: automatic reduction and verifiable upper bounds for energy barriers in micromagnetic models of magnetic grains* (Computers & Geosciences, submitted).

## Layout

```
paper/
  scripts/   drivers for every table, figure and supplementary section
  results/   JSON/CSV outputs of all runs (the run tree without meshes, states and bands)
  figures/   created by the figure scripts
  data/      external input data, not included (see below)
```

`scripts/_paths.py` defines four locations, each of which can be changed with an environment variable:

| Name | Default | Variable | Content |
|---|---|---|---|
| `RUNS` | `paper/results` | `SYMBARRIER_RUNS` | run tree read and written by the scripts |
| `DATA` | `paper/data` | `SYMBARRIER_DATA` | grain meshes and grain table of Gergov et al. |
| `FIGURES` | `paper/figures` | | figure output |
| `MERRILL` | `merrill` | `MERRILL` | MERRILL executable |

The figure and table scripts need only `results/`. Scripts that call a solver also need the meshes, relaxed states and bands of the full run tree; point `SYMBARRIER_RUNS` to the archived release, or to an empty working directory to recompute from scratch.

## Without a solver

```
cd paper/scripts
python make_fig1_concept.py      # Fig. 1
python make_fig2_scaling.py      # Fig. 2
python make_fig3_profiles.py     # Fig. 3
python make_fig4_size_sweep.py   # Fig. 4 and results/sweep.json
python sweep_table.py            # Table 3
```

The four figures are reproduced exactly from `results/`. `scaling.py` (Table 1) needs no solver either, but reads the truncated-octahedron meshes of the size sweep (`make_meshes.py to <size> <n_sub>`).

## Scripts by section

| Section | Scripts |
|---|---|
| Meshes | `make_meshes.py` (truncated octahedra and prisms), `rotate_pat.py` (crystal orientations of the tomography-derived grains) |
| 5.1 Backends and cost (Table 1, Fig. 2) | `scaling.py`, `merrill80_backend.py`, `prism_fd_classes.py`, `prism_fd_network.py`, `prism_fd_class966.py` |
| 5.2 Idealised grains (Table 2, Fig. 3) | `merrill_runner.py`, `merrill_symdecomp.py`, `fd_symdecomp.py`, `fd_truncoct.py`, `collect_numbers.py` |
| 5.3 Size dependence (Table 3, Fig. 4) | `merrill_sizesweep.py`, `make_fig4_size_sweep.py`, `sweep_table.py`, `ranking_meshconv.py` |
| 5.4 Square prism | `prism_align_fd.py`, `prism_align_merrill.py`, `prism_crossmap.py`, `prism_fd_split.py`, `prism_rerelax.py`, `prism_close.py` |
| 5.5 Tomography-derived grains (Tables 4 and 5) | `merrill_natural.py`, `merrill_network.py`, `natural_table.py`, `regen_networks.py`, `cache_from_merrill.py`, `explore_V116.py`, `explore_V091.py`, `explore_V091_warm.py` |
| 5.6 Symmetry traps | `fd_hessian.py`, `fd_modepush.py`, `fd_symbreak.py` |
| Supplementary S1–S7 | `make_supplement.py` (needs the full run tree), `merrill_random_lem.py`, `random_lem_fullOh.py`, `merrill_diag.py` |
| Supplementary S8 (6:5:4 cuboid) | `chen_cuboid_classes.py`, `chen_fd_relax.py`, `chen_fd_network.py`, `chen_fd_flower.py`, `chen_grid_states.py`, `chen_stability.py`, `chen_fd_network_grid.py`, `chen_bloch_point.py`, `chen_full_network.py`, `chen_hessian.py` |
| Shared code | `_paths.py`, `fd_defs.py` (finite-difference definitions), `path_fixes.py`, `chen_bloch_point_lib.py` |

Each script states its arguments in its docstring.

## External data

The meshes and the grain table (`VesuviusGrainMetrics.csv`) of the Vesuvius basalt grains are published by Gergov et al. (2024), https://doi.org/10.5281/zenodo.11369780. Download them into `paper/data` (or set `SYMBARRIER_DATA`); they are not redistributed here.

## Software versions

MERRILL 2.2.1 (magnetite, 20 °C); magnum.np 2.2.0 with PyTorch; Python 3.9 or later with NumPy, SciPy and Matplotlib.
