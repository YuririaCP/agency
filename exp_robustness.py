"""Robustness of the birth transition: more seeds, system sizes, and graph types.

Confirms that the aggregate->agent transition at b_c = 2c is not an artifact of N,
of the ring topology, or of a lucky seed. Writes /tmp/robustness.txt.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hoagency import Collective, CollectiveConfig, ring_graph, grid_graph  # noqa: E402
from hoagency.metrics import robustness_after_perturbation, higher_order_correction  # noqa


def measure_recovery(cfg, adj, freeze_s=None, injure_frac=0.25):
    col = Collective.build(cfg, adj)
    col.equilibrate(800)
    if freeze_s is not None:
        col.s = np.array(freeze_s, float)
    E_before = col.E_global()
    k = int(injure_frac * cfg.n)
    mask = np.zeros(cfg.n, bool); mask[:k] = True
    col.set_clamp(mask, cfg.G - 2.0)
    E_peak = col.E_global()
    for _ in range(800):
        if freeze_s is not None:
            col.s = np.array(freeze_s, float)
        col.step()
    return robustness_after_perturbation(E_before, E_peak, col.E_global())


def threshold_for(graph_name, n, seeds, cost=0.5):
    bs = np.linspace(0.4, 1.6, 25)
    bc_emp = []
    for seed in seeds:
        Psi = []
        for b in bs:
            cfg = CollectiveConfig(n=n, benefit=b, cost=cost, seed=seed)
            adj = ring_graph(n) if graph_name == "ring" else grid_graph(int(round(n ** 0.5)))
            if graph_name == "grid":
                cfg.n = int(round(n ** 0.5)) ** 2
            Rc = measure_recovery(cfg, adj)
            Ri = measure_recovery(cfg, adj, freeze_s=np.zeros(cfg.n))
            Psi.append(higher_order_correction(Rc, Ri))
        Psi = np.array(Psi)
        # empirical threshold = first b where Psi exceeds 0.5
        idx = np.argmax(Psi > 0.5) if (Psi > 0.5).any() else -1
        bc_emp.append(bs[idx] if idx >= 0 else np.nan)
    return np.array(bc_emp)


def main():
    seeds = list(range(8))
    lines = ["Birth-transition robustness (theory: b_c = 2c = 1.0)\n"]
    for graph_name in ["ring", "grid"]:
        for n in ([36, 64, 100] if graph_name == "grid" else [32, 64, 128]):
            bc = threshold_for(graph_name, n, seeds)
            lines.append(f"{graph_name:5s} N={n:4d}  b_c empirical = "
                         f"{np.nanmean(bc):.3f} +/- {np.nanstd(bc):.3f}  "
                         f"(n_seeds={np.sum(~np.isnan(bc))})")
    txt = "\n".join(lines)
    with open("/tmp/robustness.txt", "w") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
