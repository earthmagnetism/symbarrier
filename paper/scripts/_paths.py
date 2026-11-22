"""Locations used by the reproduction scripts.

RUNS     run tree: solver outputs and the JSON/CSV summaries read by the table, figure and supplement scripts.
         Defaults to paper/results, which holds the JSON/CSV part of the run tree; set SYMBARRIER_RUNS to a
         directory with the full archive (meshes, states, bands) to rerun the solver scripts.
DATA     external input data (the Vesuvius grain meshes and grain table of Gergov et al.); set SYMBARRIER_DATA.
FIGURES  output folder for figures (paper/figures).
MERRILL  MERRILL executable; set MERRILL or keep 'merrill' on the PATH.
"""
import os
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("SYMBARRIER_RUNS", PAPER / "results"))
DATA = Path(os.environ.get("SYMBARRIER_DATA", PAPER / "data"))
FIGURES = PAPER / "figures"
MERRILL = os.environ.get("MERRILL", "merrill")

FIGURES.mkdir(exist_ok=True)
if str(PAPER.parent) not in sys.path:  # use the package from this repository without installing it
    sys.path.insert(0, str(PAPER.parent))
