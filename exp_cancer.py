"""Result 2 (coda): fragmentation of the higher-order self.

A formed agent (b > b_c) is invaded by a growing fraction of 'defector' units whose
commitment is forced to zero and held there --- cells that keep their own goals but
stop contributing to the collective. This is the Q<0 / cancer regime: not a failure
of integration (the units still interact on the same graph) but a failure of
ALIGNMENT.

We sweep the defector fraction and track:
  - Q        : goal alignment (should fall, can go negative)
  - E_global : collective goal error (should rise: the agent loses its goal)
  - Psi      : higher-order error-correction (should collapse: the self fragments)

The point for the paper: alignment Q, not integration, is what distinguishes an
agent from a mere integrated aggregate --- cancer keeps interaction but loses Q.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hoagency import Collective, CollectiveConfig, ring_graph  # noqa: E402
from hoagency.metrics import (  # noqa: E402
    robustness_after_perturbation, higher_order_correction,
)

FIG = os.path.join(os.path.dirname(__file__), "..", "paper", "figs")


def run_with_defectors(defector_frac, benefit=1.4, seed=0, defect_goal=3.0,
                       defect_pull=0.30):
    cfg = CollectiveConfig(n=64, benefit=benefit, cost=0.5, seed=seed)
    col = Collective.build(cfg, ring_graph(cfg.n))
    col.equilibrate(1000)                       # form the agent first
    rng = np.random.default_rng(seed + 7)
    k = int(defector_frac * cfg.n)
    defectors = rng.choice(cfg.n, size=k, replace=False) if k else np.array([], int)

    def apply_defection(c):
        # ACTIVE defection: not just refusing to commit, but proliferating toward a
        # private goal far from G --- pulling the collective descriptor away.
        c.s[defectors] = 0.0
        c.x[defectors] += defect_pull * (defect_goal - c.x[defectors])

    align_acc = []
    for t in range(800):
        apply_defection(col)
        rec = col.run(1)
        align_acc.append(rec["align"][-1])
    Q = float(np.mean(align_acc[-200:]))
    E_glob = col.E_global()

    # higher-order correction with defectors present vs. units acting purely locally
    def recovery(freeze_local=False):
        c2 = Collective.build(cfg, ring_graph(cfg.n))
        c2.equilibrate(1000)
        s_keep = np.zeros(cfg.n)
        for _ in range(300):
            apply_defection(c2)
            if freeze_local:
                c2.s = s_keep
            c2.step()
        E_before = c2.E_global()
        mask = np.zeros(cfg.n, bool); mask[: cfg.n // 4] = True
        c2.set_clamp(mask, cfg.G - 2.0)
        E_peak = c2.E_global()
        for _ in range(600):
            apply_defection(c2)
            if freeze_local:
                c2.s = s_keep
            c2.step()
        return robustness_after_perturbation(E_before, E_peak, c2.E_global())

    Psi = higher_order_correction(recovery(False), recovery(True))
    return Q, E_glob, Psi


def main():
    os.makedirs(FIG, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fracs = np.linspace(0.0, 0.6, 13)
    seeds = [0, 1, 2, 3]
    Q = np.zeros((len(fracs), len(seeds)))
    Eg = np.zeros_like(Q)
    P = np.zeros_like(Q)
    for j, s in enumerate(seeds):
        for i, fr in enumerate(fracs):
            Q[i, j], Eg[i, j], P[i, j] = run_with_defectors(fr, seed=s)

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    def band(a, color, label, mk):
        m, sd = a.mean(1), a.std(1)
        ax.plot(fracs, m, mk + "-", color=color, lw=1.8, ms=4, label=label)
        ax.fill_between(fracs, m - sd, m + sd, color=color, alpha=0.15)
    band(P, "C3", r"$\Psi$ higher-order correction", "o")
    band(Q, "C2", r"$Q$ goal alignment", "^")
    band(Eg / Eg.max(), "C1", r"$E_\mathrm{global}$ (normalized)", "s")
    ax.axhline(0, color="0.7", lw=1)
    ax.set_xlabel("fraction of defecting units (proliferate toward private goal)")
    ax.set_ylabel("order parameters")
    ax.set_title("Fragmentation of the higher-order self")
    ax.legend(frameon=False, fontsize=8, loc="center right")
    fig.tight_layout()
    out = os.path.join(FIG, "fig_cancer.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=150)

    with open("/tmp/cancer_stats.txt", "w") as f:
        for i, fr in enumerate(fracs):
            f.write(f"defectors={fr:.2f}  Q={Q[i].mean():+.3f}  "
                    f"E_glob={Eg[i].mean():.3f}  Psi={P[i].mean():.3f}\n")
        f.write(f"wrote {os.path.relpath(out)}\n")


if __name__ == "__main__":
    main()
