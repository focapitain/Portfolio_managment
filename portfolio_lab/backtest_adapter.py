"""Pont entre le framework `portfolio_lab` et le module `backtest` (non modifié).

`backtest.walk_forward_backtest` attend une fonction
    optimize_weights_fn(train_returns: pd.DataFrame) -> poids
qui reçoit des rendements PAR PÉRIODE (déjà calculés par le backtest) et renvoie
des poids. Ce module fabrique cette fonction à partir d'un `ExperimentConfig` :
il estime μ et Σ avec les estimateurs de la config, annualise, puis appelle
l'optimiseur de la config.

Aucune modification de `backtest.py` : on s'y branche uniquement via ses
paramètres `optimize_weights_fn`, `compute_returns_fn` et `benchmark_weights_fn`.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import backtest as bt
from .config import ExperimentConfig
from .frequencies import periods_per_year


def build_compute_returns_fn(config: ExperimentConfig) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Fonction prix -> rendements cohérente avec `config.return_mode`.

    Préserve les NaN PAR COLONNE (dropna how="all", pas "any") : la sélection des
    actifs réellement disponibles est faite par fenêtre dans le walk-forward, ce qui
    évite le look-ahead consistant à n'utiliser que les actifs présents sur TOUTE la
    période. (Diffère de `compute_returns`, qui produit un panel rectangulaire pour
    l'usage in-sample.)
    """
    mode = config.return_mode

    def _fn(prices: pd.DataFrame) -> pd.DataFrame:
        if mode == "log":
            r = np.log(prices / prices.shift(1))
        elif mode == "simple":
            r = prices.pct_change()
        else:
            raise ValueError(f"mode inconnu : {mode!r} (attendu 'simple' ou 'log')")
        return r.dropna(how="all")

    return _fn


def build_optimize_weights_fn(config: ExperimentConfig,
                              ppy: int) -> Callable[[pd.DataFrame], pd.Series]:
    """Closure compatible backtest : train_returns (T×N) -> poids (pd.Series).

    `ppy` = nombre de périodes par an (pour l'annualisation), aligné sur la
    fréquence du backtest.
    """
    cfg = config.resolved()

    def _optimize(train_returns: pd.DataFrame) -> pd.Series:
        cols = train_returns.columns
        mu_p = cfg.mean_estimator.estimate(train_returns)
        Sigma_p = cfg.cov_estimator.estimate(train_returns)
        mu = np.asarray(mu_p, dtype=float) * ppy
        Sigma = np.asarray(Sigma_p.values, dtype=float) * ppy
        w = cfg.optimizer.solve(
            mu, Sigma, objective=cfg.objective, rf=cfg.risk_free_annual,
            long_only=cfg.long_only, w_max=cfg.w_max, risk_aversion=cfg.risk_aversion,
        )
        return pd.Series(np.asarray(w, dtype=float).flatten(), index=cols)

    return _optimize


def run_walk_forward(prices: pd.DataFrame,
                     config: ExperimentConfig,
                     *,
                     backtest_config: Optional[bt.WalkForwardBacktestConfig] = None,
                     benchmark_config: Optional[ExperimentConfig] = None,
                     label: Optional[str] = None) -> bt.WalkForwardBacktestResult:
    """Lance un backtest walk-forward hors échantillon pour UNE config.

    `prices` doit déjà être à la bonne fréquence (resample en amont) et la
    `backtest_config.frequency` doit correspondre. Si `benchmark_config` est
    fourni, le benchmark utilise cette config ; sinon le backtest retombe sur
    l'équipondéré 1/N (comportement par défaut du module).
    """
    bt_cfg = backtest_config or bt.WalkForwardBacktestConfig()
    # Aligne le taux sans risque du backtest (utilisé pour le Sharpe) sur celui de la
    # config d'expérience, sauf si l'appelant l'a explicitement fixé sur le backtest.
    if bt_cfg.risk_free_annual == 0.0 and config.risk_free_annual:
        bt_cfg = replace(bt_cfg, risk_free_annual=config.risk_free_annual)
    ppy = periods_per_year(bt_cfg.frequency)

    optimize_fn = build_optimize_weights_fn(config, ppy)
    returns_fn = build_compute_returns_fn(config)

    benchmark_fn = None
    if benchmark_config is not None:
        benchmark_fn = build_optimize_weights_fn(benchmark_config, ppy)

    return bt.walk_forward_backtest(
        prices,
        optimize_fn,
        backtest_config=bt_cfg,
        compute_returns_fn=returns_fn,
        benchmark_weights_fn=benchmark_fn,
        label=label or config.optimizer.name if config.optimizer else "strategy",
    )


__all__ = ["build_compute_returns_fn", "build_optimize_weights_fn", "run_walk_forward"]
