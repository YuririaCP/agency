"""Result 1 (birth): does a higher-order agent appear, and is it a transition?

We sweep the control parameter alpha (max weight a unit can place on the collective
goal) at several social-reinforcement strengths kappa, and at each setting measure
THREE things that are NOT the control parameter:

  - E_global_final : can the collective reach its goal G at all?
  - Q              : is error-correction goal-aligned across scales?
  - Psi            : the order parameter = higher-order error-correction, i.e. how
                     much damage the collective repairs that the same units in
                     isolation (alpha=0) cannot.

The honest question: does Psi rise sharply (a transition) or gradually (a crossover)
as alpha increases? We report whichever the simulation shows.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hoagency import (  # noqa: E402
    Collective, CollectiveConfig, ring_graph,
    alignment_Q, robustness_after_perturbation, higher_order_correction,
)

OUT = "/tmp/birth.txt"


def measure_recovery(cfg: CollectiveConfig, freeze_s=None, injure_frac=0.25) -> float:
    """Equilibrate, inflict a non-correctable injury, measure recovery of C toward G.

    Injury = clamp a contiguous fraction of units far from G (they cannot move). No
    single unit can fix this; only the rest of the collective can compensate the
    descriptor C by shifting modestly. If freeze_s holds commitment at 0, units act
    purely locally (each sits at its own g_i) and cannot compensate --- the
    individual counterfactual."""
    col = Collective.build(cfg, ring_graph(cfg.n))
    col.equilibrate(800)
    if freeze_s is not None:
        col.s = np.array(freeze_s, float)
    E_before = col.E_global()
    k = int(injure_frac * cfg.n)
    mask = np.zeros(cfg.n, bool); mask[:k] = True
    col.set_clamp(mask, cfg.G - 2.0)            # hold injured units far below G
    E_peak = col.E_global()
    for _ in range(800):
        if freeze_s is not None:
            col.s = np.array(freeze_s, float)
        col.step()
    E_after = col.E_global()
    return robustness_after_perturbation(E_before, E_peak, E_after)


def run():
    benefits = np.linspace(0.0, 2.0, 21)
    kappas = [0.0, 0.05, 0.15]
    rows = []
    for kappa in kappas:
        for b in benefits:
            cfg = CollectiveConfig(n=64, benefit=b, cost=0.5, kappa=kappa, seed=0)
            col = Collective.build(cfg, ring_graph(cfg.n))
            rec = col.run(1000)
            # Q = late-time alignment of unit moves with the collective goal
            Q = float(rec["align"][-200:].mean())
            R_coll = measure_recovery(cfg)
            # individual counterfactual: SAME benefit but commitment frozen at 0
            R_ind = measure_recovery(cfg, freeze_s=np.zeros(cfg.n))
            Psi = higher_order_correction(R_coll, R_ind)
            rows.append((kappa, b, rec["E_global_final"], Q, rec["mean_s"], R_coll, R_ind, Psi))
    return rows


def main():
    rows = run()
    lines = [f"{'kappa':>6}{'benefit':>8}{'E_glob':>9}{'Q':>8}{'mean_s':>8}"
             f"{'R_coll':>8}{'R_ind':>8}{'Psi':>8}"]
    for r in rows:
        lines.append(f"{r[0]:>6.2f}{r[1]:>8.2f}{r[2]:>9.4f}{r[3]:>8.3f}"
                     f"{r[4]:>8.3f}{r[5]:>8.3f}{r[6]:>8.3f}{r[7]:>8.3f}")
    txt = "\n".join(lines)
    with open(OUT, "w") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
