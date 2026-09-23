"""Frontière efficiente : résout `target_return` sur une grille de μ cibles."""
from __future__ import annotations

import numpy as np

from .optimizers import portfolio_stats


def efficient_frontier(mu, Sigma, optimizer, *, rf=0.0, long_only=True,
                       w_max=1.0, n_points=40):
    """Renvoie (vols, rets) le long de la frontière efficiente."""
    w_mvp = optimizer.solve(mu, Sigma, objective="min_variance", rf=rf,
                            long_only=long_only, w_max=w_max)
    mu_lo = float(w_mvp @ mu)
    mu_hi = float(mu.max()) * (0.999 if long_only else 1.5)
    grid = np.linspace(mu_lo, mu_hi, n_points)
    vols, rets = [], []
    for m in grid:
        try:
            w = optimizer.solve(mu, Sigma, objective="target_return", rf=rf,
                                long_only=long_only, w_max=w_max, mu_target=m)
            if w is None:
                continue
            r, v, _ = portfolio_stats(w, mu, Sigma, rf)
            rets.append(r)
            vols.append(v)
        except Exception:
            continue
    return np.array(vols), np.array(rets)


__all__ = ["efficient_frontier"]
