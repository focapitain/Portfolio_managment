"""Métriques de performance et de risque sur une série de rendements.

Complète les métriques déjà calculées par `backtest.performance_summary`
(Sharpe, max drawdown, tracking error, turnover…) avec des mesures de risque
de queue souvent demandées en recherche quantitative.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sortino_ratio(returns: pd.Series, periods_per_year: int, rf_annual: float = 0.0) -> float:
    """Sortino = (rendement annualisé - rf) / volatilité des pertes (downside)."""
    r = returns.dropna()
    if r.empty:
        return np.nan
    rf_period = rf_annual / periods_per_year
    downside = r[r < rf_period] - rf_period
    dd = np.sqrt((downside ** 2).mean()) * np.sqrt(periods_per_year) if len(downside) else 0.0
    ann_ret = (1 + r).prod() ** (periods_per_year / len(r)) - 1
    return float((ann_ret - rf_annual) / dd) if dd > 0 else np.nan


def value_at_risk(returns: pd.Series, alpha: float = 0.05) -> float:
    """VaR historique (perte au quantile alpha). Retournée en valeur positive."""
    r = returns.dropna()
    if r.empty:
        return np.nan
    return float(-np.quantile(r, alpha))


def conditional_var(returns: pd.Series, alpha: float = 0.05) -> float:
    """CVaR / Expected Shortfall : perte moyenne au-delà de la VaR. Valeur positive."""
    r = returns.dropna()
    if r.empty:
        return np.nan
    var = np.quantile(r, alpha)
    tail = r[r <= var]
    return float(-tail.mean()) if len(tail) else np.nan


def calmar_ratio(returns: pd.Series, periods_per_year: int) -> float:
    """Calmar = CAGR / |max drawdown|."""
    r = returns.dropna()
    if r.empty:
        return np.nan
    nav = (1 + r).cumprod()
    cagr = nav.iloc[-1] ** (periods_per_year / len(r)) - 1
    mdd = (nav / nav.cummax() - 1).min()
    return float(cagr / abs(mdd)) if mdd < 0 else np.nan


def tail_metrics(returns: pd.Series, periods_per_year: int,
                 rf_annual: float = 0.0, alpha: float = 0.05) -> pd.Series:
    """Regroupe les métriques de queue pour une série de rendements."""
    return pd.Series({
        "Sortino": sortino_ratio(returns, periods_per_year, rf_annual),
        "Calmar": calmar_ratio(returns, periods_per_year),
        f"VaR {int(alpha*100)}% (%)": value_at_risk(returns, alpha) * 100,
        f"CVaR {int(alpha*100)}% (%)": conditional_var(returns, alpha) * 100,
        "Skew": float(returns.dropna().skew()),
        "Kurtosis": float(returns.dropna().kurtosis()),
    })


__all__ = ["sortino_ratio", "value_at_risk", "conditional_var", "calmar_ratio",
           "tail_metrics"]
