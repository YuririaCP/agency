"""Ablation of the functional signature sigma = (|C-G*|, Q, R, B).

The identity claim rests on D_self being able to discriminate two situations that
should be judged oppositely:

  RENEWAL  : the same agent after complete substrate turnover  -> SAME self  (small D)
  FRAGMENT : the agent driven into the cancer-like regime      -> OTHER self (large D)

A good signature gives D(renewal) < eps < D(fragment): the agent survives having its
matter replaced but not having its alignment inverted. We test whether each of the
four components of sigma is necessary for this discrimination by ablating one at a
time (setting its weight to zero) and asking whether the gap between RENEWAL and
FRAGMENT survives.

If removing a component collapses the gap -- renewal and fragmentation become
indistinguishable, or their order flips -- that component is necessary, and sigma is
shown to be minimal rather than arbitrary.

Writes /tmp/ablation.txt and paper/figs/fig_ablation.pdf.
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
from experiments.exp_persistence import run_turnover, signature  # noqa: E402

FIG = os.path.join(os.path.dirname(__file__), "..", "paper", "figs")
COMPONENTS = ["goal_gap", "Q", "R", "B"]


def quick_robustness(col, injure_frac=0.25, steps=400):
    x0, s0 = col.x.copy(), col.s.copy()
    cm, cv = col.clamp_mask.copy(), col.clamp_val.copy()
    Eb = col.E_global()
    k = int(injure_frac * col.cfg.n); mask = np.zeros(col.cfg.n, bool); mask[:k] = True
    col.set_clamp(mask, col.cfg.G - 2.0); Ep = col.E_global()
    for _ in range(steps):
        col.step()
    R = robustness_after_perturbation(Eb, Ep, col.E_global())
    col.x, col.s, col.clamp_mask, col.clamp_val = x0, s0, cm, cv
    return R


def sig_of(col):
    x0, s0 = col.x.copy(), col.s.copy()
    rec = col.run(120); Q = float(rec["align"][-60:].mean())
    col.x, col.s = x0, s0
    R = quick_robustness(col)
    B = float(1.0 / (1.0 + np.var(col.x)))
    return agentive_signature(goal_gap=abs(col.C() - col.cfg.G), Q=Q, R=R, B=B)


def build_agent(seed=0, b=1.4):
    cfg = CollectiveConfig(n=64, benefit=b, cost=0.5, seed=seed)
    col = Collective.build(cfg, ring_graph(cfg.n)); col.equilibrate(1000)
    return cfg, col


def renewed_signature(seed=0):
    """Agent after full substrate turnover (reuse persistence machinery)."""
    cfg, col = build_agent(seed)
    s0 = sig_of(col)
    rng = np.random.default_rng(seed + 100)
    for _ in range(5):                       # replace whole population in cohorts
        n = cfg.n; idx = rng.choice(n, size=n // 5, replace=False)
        col.g[idx] = (cfg.G + cfg.g_mean_offset) + cfg.g_spread * rng.standard_normal(idx.size)
        col.x[idx] = col.g[idx]; col.s[idx] = cfg.s_init; col.clamp_mask[idx] = False
        col.equilibrate(400)
    return s0, sig_of(col)


def fragmented_signature(seed=0, defect_frac=0.5, steps=800, pull=0.30):
    """Agent driven into the cancer-like regime (active defectors).

    defect_frac and the number of steps control how far the fragmentation has
    progressed. EARLY fragmentation (small defect_frac, few steps) inverts alignment
    Q while goal error and robustness have not yet collapsed -- the regime in which Q
    is the only early warning, and the one the paper claims integration-based measures
    miss."""
    cfg, col = build_agent(seed)
    s0 = sig_of(col)
    rng = np.random.default_rng(seed + 7)
    k = int(defect_frac * cfg.n)
    defect = rng.choice(cfg.n, size=k, replace=False)
    for _ in range(steps):
        col.s[defect] = 0.0
        col.x[defect] += pull * (3.0 - col.x[defect])
        col.step()
    return s0, sig_of(col)


def main():
    seeds = [0, 1, 2, 3]
    # weights for full sigma and for each ablation
    settings = {"full": dict(goal_gap=1, Q=1, R=1, B=1)}
    for c in COMPONENTS:
        w = dict(goal_gap=1, Q=1, R=1, B=1); w[c] = 0
        settings[f"no_{c}"] = w

    # Early fragmentation: alignment Q has inverted but goal error and robustness
    # have not yet collapsed -- the regime in which Q is the only early signal.
    pairs_renew = [renewed_signature(s) for s in seeds]
    pairs_frag = [fragmented_signature(s, defect_frac=0.18, steps=150, pull=0.20)
                  for s in seeds]

    rows = {}
    for name, w in settings.items():
        Dr = np.array([self_distance(a, b, w) for a, b in pairs_renew])
        Df = np.array([self_distance(a, b, w) for a, b in pairs_frag])
        rows[name] = (Dr, Df)

    lines = [f"{'signature':<12}{'D_renewal':>12}{'D_fragment':>12}{'gap(frag-renew)':>18}{'order ok?':>12}"]
    for name, (Dr, Df) in rows.items():
        gap = Df.mean() - Dr.mean()
        # the signature must rank fragmentation FARTHER than renewal: gap > 0
        ok = "YES" if gap > 0 else "NO (inverts)"
        lines.append(f"{name:<12}{Dr.mean():>12.3f}{Df.mean():>12.3f}{gap:>18.3f}{ok:>12}")
    txt = "\n".join(lines)
    with open("/tmp/ablation.txt", "w") as f:
        f.write(txt + "\n")
    print(txt)

    _plot(rows)


def _plot(rows):
    os.makedirs(FIG, exist_ok=True)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(rows.keys())
    labels = {"full": "full $\\sigma$", "no_goal_gap": "$-|C{-}G^*|$",
              "no_Q": "$-Q$", "no_R": "$-R$", "no_B": "$-B$"}
    x = np.arange(len(names)); w = 0.38
    Dr = [rows[n][0].mean() for n in names]; Drs = [rows[n][0].std() for n in names]
    Df = [rows[n][1].mean() for n in names]; Dfs = [rows[n][1].std() for n in names]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(x - w/2, Dr, w, yerr=Drs, capsize=3, color="C0", label="renewal (same self)")
    ax.bar(x + w/2, Df, w, yerr=Dfs, capsize=3, color="C3", label="fragmentation (other self)")
    ax.axhline(0.15, color="0.5", ls=":", lw=1); ax.text(-0.4, 0.17, r"$\epsilon$", color="0.4")
    ax.set_xticks(x); ax.set_xticklabels([labels[n] for n in names])
    ax.set_ylabel(r"$D_{\mathrm{self}}$ from original agent")
    ax.set_title("Ablation of the functional signature $\\sigma$")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIG, "fig_ablation.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=150)
    print(f"wrote {os.path.relpath(out)}")


if __name__ == "__main__":
    main()
