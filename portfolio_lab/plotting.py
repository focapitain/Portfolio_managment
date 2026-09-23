"""Visualisations automatiques des résultats in-sample."""
from __future__ import annotations

import numpy as np

from .experiment import ExperimentResult
from .frontier import efficient_frontier

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:  # pragma: no cover
    plt = None
    sns = None


def plot_weights(result: ExperimentResult, threshold=0.005, ax=None):
    """Bar chart des poids significatifs (>0.5% par défaut)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5))
    s = result.weights[result.weights.abs() > threshold].sort_values()
    colors = ["crimson" if x < 0 else "steelblue" for x in s]
    s.plot(kind="bar", color=colors, ax=ax, edgecolor="black", lw=0.4)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"Poids — {result.config.objective} / {result.config.optimizer.name} "
                 f"({result.config.cov_estimator.name})")
    ax.set_ylabel("Poids")
    return ax


def plot_efficient_frontier(result: ExperimentResult, n_points=40, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 7))
    mu, Sig = result.mu_annual, result.Sigma_annual.values
    cfg = result.config
    vols, rets = efficient_frontier(mu, Sig, cfg.optimizer, rf=cfg.risk_free_annual,
                                    long_only=cfg.long_only, w_max=cfg.w_max, n_points=n_points)
    ax.plot(vols * 100, rets * 100, "g-", lw=2.5, label="Frontière efficiente")
    ax.scatter(np.sqrt(np.diag(Sig)) * 100, mu * 100, c="gray", s=25, alpha=0.5,
               label="Actifs individuels")
    ax.scatter(result.vol * 100, result.ret * 100, marker="*", s=260, c="red",
               edgecolor="black", zorder=5, label=f"Portefeuille ({cfg.objective})")
    ax.set_xlabel("Volatilité annualisée (%)")
    ax.set_ylabel("Rendement annualisé (%)")
    ax.set_title("Frontière efficiente")
    ax.legend()
    return ax


def plot_correlation(result: ExperimentResult, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))
    corr = result.returns.corr()
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True,
                cbar_kws={"label": "Corrélation"}, ax=ax)
    N = corr.shape[0]
    mean_off = (corr.values.sum() - N) / (N * (N - 1))
    ax.set_title(f"Matrice de corrélation (corr. moyenne hors-diag = {mean_off:.2f})")
    return ax


def plot_performance(result: ExperimentResult, ax=None):
    """Performance cumulée in-sample du portefeuille (poids figés)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5))
    port_ret = result.returns @ result.weights.reindex(result.returns.columns).fillna(0)
    cum = (1 + port_ret).cumprod()
    cum.plot(ax=ax, lw=2, color="darkgreen")
    ax.set_title("Performance cumulée du portefeuille (in-sample, base 1)")
    ax.set_ylabel("Valeur du portefeuille")
    return ax


__all__ = ["plot_weights", "plot_efficient_frontier", "plot_correlation",
           "plot_performance"]
