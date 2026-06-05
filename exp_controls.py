"""Controls: dissociate the ingredients of higher-order agency.

The thesis is that ALIGNMENT (Q), not mere integration/connectivity, is what makes a
collection an agent. These controls test that by dissociating the factors:

  C1  high connectivity, low benefit  -> integration present, Q absent: expect Psi~0.
       (a complete graph below b_c: everyone interacts, no one aligns)
  C2  high benefit, low connectivity  -> Q present on a sparse graph: expect Psi>0.
       (sparse random graph above b_c: alignment forms despite few edges)
  C3  topology sweep at fixed b>b_c   -> agent forms across ring/grid/random/complete:
       Psi>0 regardless of topology (existence of transition is topology-independent).
  C4  noise robustness                -> Psi stays high as state noise grows, until it
       swamps the dynamics.

Writes /tmp/controls.txt.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hoagency import (  # noqa: E402
    Collective, CollectiveConfig, ring_graph, grid_graph, complete_graph, random_graph,
)
from hoagency.metrics import robustness_after_perturbation, higher_order_correction  # noqa


def connectivity(adj):
    n = adj.shape[0]
    return float(adj.sum() / (n * (n - 1)))    # edge density in [0,1]


def measure(cfg, adj, freeze_s=None, injure_frac=0.25, eq=800, rec=800):
    col = Collective.build(cfg, adj)
    col.equilibrate(eq)
    if freeze_s is not None:
        col.s = np.array(freeze_s, float)
    Eb = col.E_global()
    k = int(injure_frac * cfg.n)
    mask = np.zeros(cfg.n, bool); mask[:k] = True
    col.set_clamp(mask, cfg.G - 2.0)
    Ep = col.E_global()
    for _ in range(rec):
        if freeze_s is not None:
            col.s = np.array(freeze_s, float)
        col.step()
    return robustness_after_perturbation(Eb, Ep, col.E_global())


def psi_and_Q(cfg, adj, seeds):
    Ps, Qs = [], []
    for s in seeds:
        c = CollectiveConfig(**{**cfg.__dict__, "seed": s})
        col = Collective.build(c, adj)
        rec = col.run(1000)
        Qs.append(float(rec["align"][-200:].mean()))
        Rc = measure(c, adj)
        Ri = measure(c, adj, freeze_s=np.zeros(c.n))
        Ps.append(higher_order_correction(Rc, Ri))
    return np.mean(Ps), np.std(Ps), np.mean(Qs), np.std(Qs)


def main():
    seeds = [0, 1, 2, 3]
    n = 64
    L = ["Controls dissociating alignment from connectivity\n"]

    # C1: high connectivity, low benefit (complete graph, b < b_c)
    adjK = complete_graph(n)
    cfg = CollectiveConfig(n=n, benefit=0.6, cost=0.5)
    P, Ps, Q, Qs = psi_and_Q(cfg, adjK, seeds)
    L.append(f"C1 complete graph, b=0.6 (<b_c): conn={connectivity(adjK):.2f}  "
             f"Psi={P:.3f}+/-{Ps:.3f}  Q={Q:+.3f}  -> integration high, agency absent")

    # C2: high benefit, low connectivity (sparse random graph, b > b_c)
    adjR = random_graph(n, mean_degree=3.0, seed=0)
    cfg = CollectiveConfig(n=n, benefit=1.4, cost=0.5)
    P, Ps, Q, Qs = psi_and_Q(cfg, adjR, seeds)
    L.append(f"C2 sparse random, b=1.4 (>b_c): conn={connectivity(adjR):.2f}  "
             f"Psi={P:.3f}+/-{Ps:.3f}  Q={Q:+.3f}  -> agency forms despite sparse graph")

    # C3: topology sweep at fixed b > b_c
    L.append("C3 topology sweep at b=1.4 (>b_c):")
    for name, adj in [("ring", ring_graph(n)), ("grid", grid_graph(8)),
                      ("random(k=4)", random_graph(n, 4.0, 0)),
                      ("complete", complete_graph(n))]:
        cfg = CollectiveConfig(n=adj.shape[0], benefit=1.4, cost=0.5)
        P, Ps, Q, Qs = psi_and_Q(cfg, adj, seeds)
        L.append(f"   {name:12s} conn={connectivity(adj):.3f}  "
                 f"Psi={P:.3f}+/-{Ps:.3f}  Q={Q:+.3f}")

    # C4: noise robustness on ring at b > b_c
    L.append("C4 noise sweep (ring, b=1.4):")
    for noise in [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]:
        cfg = CollectiveConfig(n=n, benefit=1.4, cost=0.5, noise=noise)
        P, Ps, Q, Qs = psi_and_Q(cfg, ring_graph(n), seeds)
        L.append(f"   noise={noise:.2f}  Psi={P:.3f}+/-{Ps:.3f}  Q={Q:+.3f}")

    # C5: HIGH commitment, LOW alignment -- units commit toward a WRONG goal.
    # b > b_c so commitment forms (high <s>), but G_perceived != G so the
    # coordinated effort points at the wrong target: Q and Psi should stay low.
    L.append("C5 committed but misaligned (ring, b=1.4, G_perceived != G):")
    for gp in [1.0, 0.5, 0.0, -0.5]:
        s_means = []
        Pl, Ql = [], []
        for s in seeds:
            cfg = CollectiveConfig(n=n, benefit=1.4, cost=0.5, G=1.0,
                                   G_perceived=gp, seed=s)
            col = Collective.build(cfg, ring_graph(n))
            rec = col.run(1000)
            s_means.append(rec["mean_s"]); Ql.append(float(rec["align"][-200:].mean()))
            Rc = measure(cfg, ring_graph(n))
            Ri = measure(cfg, ring_graph(n), freeze_s=np.zeros(n))
            Pl.append(higher_order_correction(Rc, Ri))
        L.append(f"   G_perc={gp:+.1f}  <s>={np.mean(s_means):.3f}  "
                 f"Q={np.mean(Ql):+.3f}  Psi={np.mean(Pl):.3f}")

    txt = "\n".join(L)
    with open("/tmp/controls.txt", "w") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
