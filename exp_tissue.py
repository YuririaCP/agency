"""Spatial turnover in a 2D tissue layer (tissue-like robustness check).

The persistence result was established on a ring with random turnover. Here we ask
whether it survives a more tissue-like setting: an L x L epithelial sheet with local
(von Neumann) neighbourhoods, where replacement happens in contiguous square patches
-- local tissue renewal or injury -- rather than uniformly at random.

Same dynamics, same global target G*, same functional signature sigma = (|C-G*|,Q,R,B)
and D_self. We compare:

  (1) D_self vs cumulative fraction replaced, PATCH vs RANDOM turnover, slow reintegration
  (2) D_self vs patch size at fixed total replacement (does bigger-at-once break it?)

Prediction to be TESTED (not assumed): identity persists under localized turnover when
reintegration has time, and fails when patches are too large or too frequent. We report
whatever the simulation shows -- patch turnover could be harsher than random because a
contiguous patch wipes a whole region's aligned neighbours at once.

Writes /tmp/tissue.txt and paper/figs/fig_tissue.pdf.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hoagency import Collective, CollectiveConfig, grid_graph  # noqa: E402
from hoagency.metrics import self_distance  # noqa: E402
from experiments.exp_persistence import signature  # noqa: E402

FIG = os.path.join(os.path.dirname(__file__), "..", "paper", "figs")
EPS = 0.15


def build_tissue(L, seed, b=1.4):
    n = L * L
    cfg = CollectiveConfig(n=n, benefit=b, cost=0.5, seed=seed)
    col = Collective.build(cfg, grid_graph(L))
    col.equilibrate(1000)
    return cfg, col


def refresh(col, idx, rng):
    cfg = col.cfg
    col.g[idx] = (cfg.G + cfg.g_mean_offset) + cfg.g_spread * rng.standard_normal(len(idx))
    col.x[idx] = col.g[idx]; col.s[idx] = cfg.s_init; col.clamp_mask[idx] = False


def patch_indices(L, k, rng):
    """A contiguous k x k patch at a random location (wraps via clipping)."""
    r0 = rng.integers(0, max(1, L - k + 1)); c0 = rng.integers(0, max(1, L - k + 1))
    idx = [r * L + c for r in range(r0, min(L, r0 + k)) for c in range(c0, min(L, c0 + k))]
    return np.array(idx, dtype=int)


def run_patch_turnover(L=16, k=3, reequil=400, rounds=12, seed=0, random_mode=False):
    cfg, col = build_tissue(L, seed)
    rng = np.random.default_rng(seed + 100)
    s0 = signature(col)
    n = L * L
    per_round = k * k
    Ds, cum = [], []
    replaced = 0
    for _ in range(rounds):
        if random_mode:
            idx = rng.choice(n, size=per_round, replace=False)
        else:
            idx = patch_indices(L, k, rng)
        refresh(col, idx, rng)
        col.equilibrate(reequil)
        replaced += len(idx)
        Ds.append(self_distance(s0, signature(col)))
        cum.append(min(1.0, replaced / n))
    return np.array(cum), np.array(Ds)


def main():
    seeds = [0, 1, 2, 3]
    L = 16

    # (1) patch vs random, small patches, slow reintegration
    Dp = np.array([run_patch_turnover(L, k=3, reequil=400, seed=s)[1] for s in seeds])
    Dr = np.array([run_patch_turnover(L, k=3, reequil=400, seed=s, random_mode=True)[1]
                   for s in seeds])
    cum = run_patch_turnover(L, k=3, reequil=400, seed=0)[0]

    # (2) single-event injury of growing size, fixed reintegration time: how much of
    # the tissue can be lost AT ONCE and still recovered? Gradual vs catastrophic loss.
    ks = [2, 4, 6, 8, 10, 12]
    Db = np.zeros((len(ks), len(seeds)))
    fracs = []
    for i, k in enumerate(ks):
        fracs.append(min(1.0, k * k / (L * L)))
        for j, s in enumerate(seeds):
            cfg, col = build_tissue(L, s)
            s0 = signature(col)
            rng = np.random.default_rng(s + 100)
            idx = patch_indices(L, k, rng)
            refresh(col, idx, rng)
            col.equilibrate(600)
            Db[i, j] = self_distance(s0, signature(col))

    lines = ["Spatial (patch) vs random turnover on a 16x16 tissue, b=1.4:"]
    lines.append(f"  patch  D_self @full replacement = {Dp[:, -1].mean():.3f} (sd {Dp[:, -1].std():.3f})")
    lines.append(f"  random D_self @full replacement = {Dr[:, -1].mean():.3f} (sd {Dr[:, -1].std():.3f})")
    lines.append(f"  eps = {EPS}")
    lines.append("  single-event injury, fixed reintegration (600 steps):")
    for i, k in enumerate(ks):
        lines.append(f"    {fracs[i]*100:4.0f}% lost at once (k={k}): "
                     f"D_self={Db[i].mean():.3f} (sd {Db[i].std():.3f})")
    txt = "\n".join(lines)
    with open("/tmp/tissue.txt", "w") as f:
        f.write(txt + "\n")
    print(txt)

    _plot(cum, Dp, Dr, fracs, Db)


def _plot(cum, Dp, Dr, fracs, Db):
    os.makedirs(FIG, exist_ok=True)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.6, 3.7))

    for D, c, lab in [(Dp, "C0", "localized patches"), (Dr, "0.5", "random turnover")]:
        m, sd = D.mean(0), D.std(0)
        axA.plot(cum, m, "o-" if c == "C0" else "s--", ms=3, color=c, label=lab)
        axA.fill_between(cum, m - sd, m + sd, color=c, alpha=0.15)
    axA.axhline(EPS, color="0.6", ls=":", lw=1); axA.text(0.02, EPS + 0.01, r"$\epsilon$", color="0.4")
    axA.set_xlabel("cumulative fraction replaced")
    axA.set_ylabel(r"$D_{\mathrm{self}}$ from original")
    axA.set_title("(A) Patch vs random turnover (16$\\times$16 tissue)")
    axA.legend(frameon=False, fontsize=8)

    fr = np.array(fracs, float) * 100
    m, sd = Db.mean(1), Db.std(1)
    axB.errorbar(fr, m, yerr=sd, fmt="o-", color="C3", capsize=3)
    axB.axhline(EPS, color="0.6", ls=":", lw=1); axB.text(fr.min(), EPS + 0.01, r"$\epsilon$", color="0.4")
    axB.set_xlim(0, 60)
    axB.set_xlabel("% of tissue lost in a single event")
    axB.set_ylabel(r"final $D_{\mathrm{self}}$ after reintegration")
    axB.set_title("(B) Recovery from one-shot injury")
    fig.tight_layout()
    out = os.path.join(FIG, "fig_tissue.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=150)
    print(f"wrote {os.path.relpath(out)}")


if __name__ == "__main__":
    main()
