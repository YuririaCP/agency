"""Figure: the aggregate -> agent transition (Result 1).

Fine sweep of the control parameter (benefit/cost ratio) showing that three
independent quantities switch on together at b_c = 2c:
  - mean commitment s            (do units join the collective?)
  - alignment Q                  (are local moves goal-aligned?)
  - order parameter Psi          (collective error-correction with no individual
                                  counterpart -- the operational birth of an agent)

The transition EMERGES from the replicator bifurcation; nothing sigmoidal is
imposed. Writes paper/figs/fig_birth.{pdf,png}.
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


def measure_recovery(cfg, freeze_s=None, injure_frac=0.25):
    col = Collective.build(cfg, ring_graph(cfg.n))
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


def sweep(cost=0.5, seeds=(0, 1, 2, 3)):
    bs = np.linspace(0.4, 1.6, 31)
    S = np.zeros((len(bs), len(seeds)))
    Q = np.zeros_like(S)
    P = np.zeros_like(S)
    for j, seed in enumerate(seeds):
        for i, b in enumerate(bs):
            cfg = CollectiveConfig(n=64, benefit=b, cost=cost, seed=seed)
            rec = Collective.build(cfg, ring_graph(cfg.n)).run(1200)
            S[i, j] = rec["mean_s"]
            Q[i, j] = rec["align"][-200:].mean()
            Rc = measure_recovery(cfg)
            Ri = measure_recovery(cfg, freeze_s=np.zeros(cfg.n))
            P[i, j] = higher_order_correction(Rc, Ri)
    return bs, S, Q, P, 2 * cost


def main():
    bs, S, Q, P, bc = sweep()
    os.makedirs(FIG, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.4, 3.6))

    def band(arr, color, label, marker):
        m, sd = arr.mean(1), arr.std(1)
        ax.plot(bs, m, marker + "-", ms=3, lw=1.7, color=color, label=label)
        ax.fill_between(bs, m - sd, m + sd, color=color, alpha=0.15)

    band(P, "C3", r"$\Psi$  higher-order error-correction", "o")
    band(S, "C0", r"$\langle s\rangle$  collective commitment", "s")
    band(Q, "C2", r"$Q$  goal alignment", "^")
    ax.axvline(bc, color="0.5", lw=1, ls=":")
    ax.text(bc + 0.02, 0.05, r"$b_c=2c$", color="0.4", fontsize=9)
    ax.set_xlabel(r"public benefit $b$ (cost $c=0.5$)")
    ax.set_ylabel("order parameters")
    ax.set_ylim(-0.1, 1.08)
    ax.set_title("Birth of a higher-order agent")
    ax.legend(frameon=False, fontsize=8, loc="center left")
    fig.tight_layout()
    out = os.path.join(FIG, "fig_birth.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=150)

    # quantify sharpness: width of the Psi rise from 0.1 to 0.9
    mP = P.mean(1)
    lo = bs[np.argmax(mP > 0.1)] if (mP > 0.1).any() else np.nan
    hi = bs[np.argmax(mP > 0.9)] if (mP > 0.9).any() else np.nan
    with open("/tmp/fig_birth_stats.txt", "w") as f:
        f.write(f"b_c (theory 2c) = {bc}\n")
        f.write(f"Psi rises 0.1->0.9 over b in [{lo:.3f}, {hi:.3f}], width={hi-lo:.3f}\n")
        f.write(f"Psi below bc (b=0.5): {mP[np.argmin(abs(bs-0.5))]:.3f}\n")
        f.write(f"Psi above bc (b=1.3): {mP[np.argmin(abs(bs-1.3))]:.3f}\n")
        f.write(f"wrote {os.path.relpath(out)}\n")


if __name__ == "__main__":
    main()
