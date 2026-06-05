"""Regenerate every figure and the reported numbers in the paper.

Runs each experiment script in turn and writes its figure(s) into paper/figs/.
Console output reproduces the quantitative claims made in the text (thresholds,
order parameters, distances). Run from the repository root:

    python reproduce_all.py

Each experiment is also runnable on its own, e.g. `python experiments/exp_birth.py`.
All experiments use fixed random seeds, so results are deterministic.
"""

from __future__ import annotations

import importlib
import sys
import time

# (module, figure(s) produced, paper reference)
EXPERIMENTS = [
    ("experiments.fig_birth",        "fig_birth",        "Fig. 2  (birth transition)"),
    ("experiments.exp_scalefree",    "fig_scalefree",    "Fig. 3  (scale-free robustness)"),
    ("experiments.exp_persistence",  "fig_persistence",  "Fig. 4  (identity persistence)"),
    ("experiments.exp_tissue",       "fig_tissue",       "Fig. 5  (2D tissue turnover)"),
    ("experiments.exp_cancer",       "fig_cancer",       "Fig. 6  (fragmentation)"),
    ("experiments.exp_ablation",     "fig_ablation",     "Fig. 7  (signature ablation)"),
    ("experiments.exp_controls",     "(table)",          "Table 1 (dissociation controls)"),
    ("experiments.exp_robustness",   "(text)",           "Sec. 4  (size/topology robustness)"),
]


def main() -> None:
    print("Reproducing all figures and reported numbers.\n")
    for mod_name, fig, ref in EXPERIMENTS:
        print(f"{'='*70}\n{ref}  ->  {fig}\n  module: {mod_name}\n{'-'*70}")
        t0 = time.time()
        mod = importlib.import_module(mod_name)
        if hasattr(mod, "main"):
            mod.main()
        print(f"  ({time.time()-t0:.1f}s)\n")
    print("Done. Figures written to paper/figs/.")


if __name__ == "__main__":
    sys.exit(main())
