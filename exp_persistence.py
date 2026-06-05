"""Result 3 (persistence): does the agent stay the same self across substrate change?

Once a higher-order agent exists (b > b_c), we subject it to substrate turnover:
repeatedly replace a fraction of its units with fresh ones (new private goals, reset
commitment, reset state), letting the collective re-equilibrate between rounds. The
MICROSTATE is largely or wholly overwritten; we ask whether the AGENTIVE SIGNATURE
is preserved.

Signature S_l = (attractor, memory, goal, Q, R, B). Identity persists if
D_self(S(t1), S(t2)) < eps even as the underlying units are swapped out --- a
formal Ship of Theseus.

Controls:
  * 'turnover' below threshold (b<b_c): no agent exists, so there is nothing to
    persist; the signature should be unstable / undefined.
  * fraction swept from 0 (no turnover) to 1.0 (total replacement over the run).
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hoagency import Collective, CollectiveConfig, ring_graph  # noqa: E402
from hoagency.metrics import (  # noqa: E402
    agentive_signature, self_distance, robustness_after_perturbation,
)

FIG = os.path.join(os.path.dirname(__file__), "..", "paper", "figs")


def quick_robustness(col, injure_frac=0.25, steps=400):
    """Non-destructive probe of current recovery ability (restores state after)."""
    x_save, s_save = col.x.copy(), col.s.copy()
    cm, cv = col.clamp_mask.copy(), col.clamp_val.copy()
    E_before = col.E_global()
    k = int(injure_frac * col.cfg.n)
    mask = np.zeros(col.cfg.n, bool); mask[:k] = True
    col.set_clamp(mask, col.cfg.G - 2.0)
    E_peak = col.E_global()
    for _ in range(steps):
        col.step()
    R = robustness_after_perturbation(E_before, E_peak, col.E_global())
    col.x, col.s, col.clamp_mask, col.clamp_val = x_save, s_save, cm, cv
    return R


def signature(col):
    """Functional (substrate-independent) signature: what the collective DOES."""
    x_save, s_save = col.x.copy(), col.s.copy()
    rec = col.run(120)
    Q = float(rec["align"][-60:].mean())
    col.x, col.s = x_save, s_save
    R = quick_robustness(col)
    B = float(1.0 / (1.0 + np.var(col.x)))      # boundary coherence: tightness of C
    goal_gap = abs(col.C() - col.cfg.G)
    return agentive_signature(goal_gap=goal_gap, Q=Q, R=R, B=B)


def replace_units(col, frac, rng):
    """Swap out a `frac` fraction of units for fresh ones (new goals, reset state)."""
    n = col.cfg.n
    k = int(frac * n)
    if k == 0:
        return
    idx = rng.choice(n, size=k, replace=False)
    col.g[idx] = (col.cfg.G + col.cfg.g_mean_offset) + col.cfg.g_spread * rng.standard_normal(k)
    col.x[idx] = col.g[idx]              # fresh cell starts at its own goal
    col.s[idx] = col.cfg.s_init          # fresh cell uncommitted
    col.clamp_mask[idx] = False


def run_turnover(benefit, turnover_per_round=0.2, reequil=400, rounds=10, seed=0):
    """Build an agent, record its initial signature, then repeatedly replace units
    and re-equilibrate; track D_self and cumulative fraction replaced.

    `reequil` is how long the collective gets to re-integrate fresh units between
    rounds. Plenty of time -> identity persists; too little -> the boundary cannot
    re-form faster than it is eroded and the self is lost."""
    cfg = CollectiveConfig(n=64, benefit=benefit, cost=0.5, seed=seed)
    col = Collective.build(cfg, ring_graph(cfg.n))
    col.equilibrate(1000)
    rng = np.random.default_rng(seed + 100)
    S0 = signature(col)
    micro0 = col.x.copy()
    Ds, micro_change, cum = [], [], []
    for r in range(rounds):
        replace_units(col, turnover_per_round, rng)
        col.equilibrate(reequil)
        Sr = signature(col)
        Ds.append(self_distance(S0, Sr))
        micro_change.append(float(1 - np.corrcoef(micro0, col.x)[0, 1]))
        cum.append(min(1.0, (r + 1) * turnover_per_round))
    return np.array(cum), np.array(Ds), np.array(micro_change)


def final_dself_vs_rate(benefit=1.4, seeds=(0, 1, 2, 3)):
    """Sweep re-integration time (inverse turnover rate): does identity survive?

    Fast turnover = short reequil. Returns, for each reequil budget, the final
    D_self after a fixed total replacement. The point: persistence has a limit ---
    below a critical re-integration budget the self does not survive substrate
    change."""
    budgets = [20, 40, 80, 150, 300, 600]
    out = np.zeros((len(budgets), len(seeds)))
    for j, s in enumerate(seeds):
        for i, rq in enumerate(budgets):
            _, D, _ = run_turnover(benefit, turnover_per_round=0.25,
                                   reequil=rq, rounds=8, seed=s)
            out[i, j] = D[-1]
    return np.array(budgets), out


def main():
    os.makedirs(FIG, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seeds = [0, 1, 2, 3]

    # Panel A: identity persists while substrate is replaced (slow turnover).
    D = np.array([run_turnover(1.4, reequil=400, seed=s)[1] for s in seeds])
    M = np.array([run_turnover(1.4, reequil=400, seed=s)[2] for s in seeds])
    cum = run_turnover(1.4, reequil=400, seed=0)[0]

    # Panel B: persistence has a limit --- too-fast turnover loses the self.
    budgets, Drate = final_dself_vs_rate(1.4, seeds)
    rate = 0.25 / np.array(budgets, float)   # replaced fraction per unit time

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.6, 3.7))

    axA.plot(cum, M.mean(0), "s--", color="0.5", lw=1.6, label="substrate change")
    axA.plot(cum, D.mean(0), "o-", color="C0", lw=1.9, label=r"$D_\mathrm{self}$ (identity)")
    axA.fill_between(cum, D.mean(0) - D.std(0), D.mean(0) + D.std(0),
                     color="C0", alpha=0.15)
    axA.axhline(0.15, color="0.7", lw=1, ls=":")
    axA.text(0.02, 0.17, r"$\epsilon$", color="0.4")
    axA.set_xlabel("cumulative fraction of units replaced")
    axA.set_ylabel("distance from original self")
    axA.set_title("(A) Identity persists; substrate does not")
    axA.legend(frameon=False, fontsize=8, loc="center left")

    mD, sD = Drate.mean(1), Drate.std(1)
    axB.plot(rate, mD, "o-", color="C3", lw=1.9)
    axB.fill_between(rate, mD - sD, mD + sD, color="C3", alpha=0.15)
    axB.axhline(0.15, color="0.7", lw=1, ls=":")
    axB.text(rate.min(), 0.17, r"$\epsilon$", color="0.4")
    axB.set_xscale("log")
    axB.set_xlabel("turnover rate (fraction replaced per re-integration budget)")
    axB.set_ylabel(r"final $D_\mathrm{self}$")
    axB.set_title("(B) Persistence has a limit")

    fig.tight_layout()
    out = os.path.join(FIG, "fig_persistence.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=150)

    with open("/tmp/persist_stats.txt", "w") as f:
        f.write(f"PANEL A @100% replacement: D_self={D.mean(0)[-1]:.3f} "
                f"(sd {D.std(0)[-1]:.3f}), substrate change={M.mean(0)[-1]:.3f}\n")
        f.write("PANEL B final D_self by re-integration budget:\n")
        for bg, m, s in zip(budgets, mD, sD):
            f.write(f"   reequil={bg:4d}  rate={0.25/bg:.4f}  D_self={m:.3f} (sd {s:.3f})\n")
        f.write(f"wrote {os.path.relpath(out)}\n")


if __name__ == "__main__":
    main()
