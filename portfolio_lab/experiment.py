"""Pipeline central in-sample : config -> univers -> prix -> μ,Σ -> optimisation -> stats.

C'est l'évaluation « en échantillon » (poids estimés et mesurés sur la même
période). Pour l'évaluation HORS échantillon, voir `backtest_adapter` +
`backtest.walk_forward_backtest`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .data import clean_prices, download_prices, get_sp500_constituents
from .frequencies import annualize, resample_prices
from .optimizers import portfolio_stats
from .returns import compute_returns


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    tickers: list
    returns: pd.DataFrame
    mu_annual: np.ndarray
    Sigma_annual: pd.DataFrame
    weights: pd.Series
    ret: float
    vol: float
    sharpe: float

    def summary(self) -> pd.Series:
        return pd.Series({
            "objectif": self.config.objective,
            "optimiseur": self.config.optimizer.name,
            "covariance": self.config.cov_estimator.name,
            "fréquence": self.config.frequency,
            "N actifs": len(self.tickers),
            "rendement (%)": self.ret * 100,
            "volatilité (%)": self.vol * 100,
            "Sharpe": self.sharpe,
            "# poids > 1%": int((self.weights > 0.01).sum()),
            "Herfindahl": float((self.weights ** 2).sum()),
        })


def run_experiment(config: ExperimentConfig,
                   constituents: Optional[pd.DataFrame] = None,
                   prices: Optional[pd.DataFrame] = None) -> ExperimentResult:
    """Pipeline complet in-sample.

    `constituents` et `prices` peuvent être fournis pour éviter de re-télécharger
    à chaque expérience (indispensable pour benchmarker plusieurs configs sur les
    MÊMES données — exigence de contrôle des variables).
    """
    cfg = config.resolved()

    # 1) Univers
    if constituents is None:
        constituents = get_sp500_constituents()
    tickers = cfg.universe.select(constituents)

    # 2) Prix (téléchargés si non fournis), puis ré-échantillonnage
    if prices is None:
        prices = clean_prices(download_prices(tickers, cfg.start_date, cfg.end_date))
    avail = [t for t in tickers if t in prices.columns]
    px = resample_prices(prices[avail], cfg.frequency)

    # 3) Rendements
    rets = compute_returns(px, cfg.return_mode)

    # 4) Estimation μ, Σ (par période) puis annualisation
    mu_p = cfg.mean_estimator.estimate(rets)
    Sigma_p = cfg.cov_estimator.estimate(rets)
    mu, Sigma = annualize(mu_p, Sigma_p.values, cfg.frequency)

    # 5) Optimisation
    w = cfg.optimizer.solve(mu, Sigma, objective=cfg.objective, rf=cfg.risk_free_annual,
                            long_only=cfg.long_only, w_max=cfg.w_max,
                            risk_aversion=cfg.risk_aversion)
    w = np.asarray(w).flatten()

    # 6) Statistiques annualisées
    ret, vol, sharpe = portfolio_stats(w, mu, Sigma, cfg.risk_free_annual)
    return ExperimentResult(
        config=cfg, tickers=avail, returns=rets, mu_annual=mu,
        Sigma_annual=pd.DataFrame(Sigma, index=avail, columns=avail),
        weights=pd.Series(w, index=avail), ret=ret, vol=vol, sharpe=sharpe)


__all__ = ["ExperimentResult", "run_experiment"]
