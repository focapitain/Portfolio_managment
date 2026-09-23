"""Évaluation PRÉDICTIVE des estimateurs de rendement espéré μ (hors échantillon).

Pendant de `cov_eval.py`, mais pour μ. On compare, fenêtre par fenêtre, le μ̂ estimé sur le
passé (train) au rendement moyen RÉALISÉ sur le futur immédiat (test, jamais vu).

Avertissement méthodologique
----------------------------
μ est le paramètre le plus BRUITÉ de Markowitz : le R² d'une prévision de rendement est
quasi nul. La métrique la plus pertinente n'est PAS l'erreur absolue (RMSE) mais
l'**Information Coefficient** (corrélation de RANG entre μ̂ et le réalisé, cross-section) :
l'optimisation ne dépend que de l'ORDRE relatif des actifs, pas du niveau exact. Un IC
moyen de 0.02–0.05 est déjà « bon » en pratique.

Toutes les grandeurs sont PAR PÉRIODE (jamais annualisées). La sélection d'actifs se fait
PAR FENÊTRE (anti look-ahead), via `cov_eval._iter_windows`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cov_eval import _iter_windows, regime_split  # réutilisation (DRY)


# --------------------------------------------------------------------------- #
#  Métriques ponctuelles (un μ̂ prédit vs un rendement réalisé)                #
# --------------------------------------------------------------------------- #
def _vecs(mu_pred, realized):
    a = np.asarray(mu_pred, dtype=float).flatten()
    b = np.asarray(realized, dtype=float).flatten()
    return a, b


def rmse_mean(mu_pred, realized) -> float:
    """Racine de l'erreur quadratique moyenne entre μ̂ et le rendement réalisé."""
    a, b = _vecs(mu_pred, realized)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae_mean(mu_pred, realized) -> float:
    """Erreur absolue moyenne."""
    a, b = _vecs(mu_pred, realized)
    return float(np.mean(np.abs(a - b)))


def wmape(mu_pred, realized) -> float:
    """WMAPE = Σ|μ̂−r| / Σ|r| : MAPE robuste (le MAPE brut explose quand r≈0,
    ce qui est la norme pour des rendements)."""
    a, b = _vecs(mu_pred, realized)
    denom = float(np.sum(np.abs(b)))
    return float(np.sum(np.abs(a - b)) / denom) if denom > 0 else float("nan")


def pearson_ic(mu_pred, realized) -> float:
    """Corrélation de Pearson cross-section (niveau) entre μ̂ et le réalisé."""
    a, b = _vecs(mu_pred, realized)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman_ic(mu_pred, realized) -> float:
    """Information Coefficient = corrélation de RANG (Spearman) cross-section.
    LA métrique clé : mesure si μ̂ ordonne correctement les actifs."""
    a, b = _vecs(mu_pred, realized)
    if len(a) < 3:
        return float("nan")
    ra = pd.Series(a).rank().values
    rb = pd.Series(b).rank().values
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def hit_rate(mu_pred, realized) -> float:
    """Proportion d'actifs dont le SIGNE prédit est correct (prévision directionnelle)."""
    a, b = _vecs(mu_pred, realized)
    return float(np.mean(np.sign(a) == np.sign(b)))


# --------------------------------------------------------------------------- #
#  Moteurs walk-forward (hors échantillon)                                     #
# --------------------------------------------------------------------------- #
def rolling_mean_eval(returns: pd.DataFrame, estimator, *, window: int = 252,
                      step: int = 21) -> pd.DataFrame:
    """Évalue un estimateur de μ en walk-forward hors échantillon.

    Pour chaque fenêtre : μ̂ = estimator.estimate(train) ; cible = rendement moyen PAR
    PÉRIODE réalisé sur le test. Renvoie un DataFrame (une ligne/fenêtre) avec RMSE, MAE,
    WMAPE, Pearson, IC (Spearman), hit-rate, et la vol réalisée (pour `regime_split`).
    """
    rows, idx = [], []
    for date, train, test in _iter_windows(returns, window, step):
        mu_hat = estimator.estimate(train)
        realized = test.mean().values                      # rendement moyen réalisé (par période)
        port = test.values @ (np.ones(test.shape[1]) / test.shape[1])
        rows.append({
            "rmse": rmse_mean(mu_hat, realized),
            "mae": mae_mean(mu_hat, realized),
            "wmape": wmape(mu_hat, realized),
            "pearson": pearson_ic(mu_hat, realized),
            "IC": spearman_ic(mu_hat, realized),
            "hit_rate": hit_rate(mu_hat, realized),
            "realized_vol": float(np.std(port, ddof=1)) if len(port) > 1 else float("nan"),
        })
        idx.append(date)
    if not rows:
        raise ValueError("rolling_mean_eval : aucune fenêtre exploitable (window/step trop grands ?).")
    return pd.DataFrame(rows, index=pd.Index(idx, name="test_start"))


def mean_eval_summary(returns: pd.DataFrame, estimators: dict, *, window: int = 252,
                      step: int = 21) -> pd.DataFrame:
    """Résumé (moyenne des fenêtres) pour un dict {nom: estimateur}. Une ligne/estimateur."""
    out = {}
    for name, est in estimators.items():
        out[name] = rolling_mean_eval(returns, est, window=window, step=step).mean(numeric_only=True)
    return pd.DataFrame(out).T


def window_sensitivity(returns: pd.DataFrame, estimator, windows, *, step: int = 21,
                       metric: str = "IC") -> pd.Series:
    """Sensibilité d'un estimateur à la TAILLE de fenêtre d'estimation.

    Renvoie la valeur moyenne de `metric` (défaut : IC) pour chaque fenêtre testée.
    Un bon estimateur est PLAT (peu sensible au choix de fenêtre).
    """
    vals = {}
    for w in windows:
        df = rolling_mean_eval(returns, estimator, window=w, step=step)
        vals[w] = float(df[metric].mean())
    return pd.Series(vals, name=metric).rename_axis("window")


__all__ = [
    "rmse_mean", "mae_mean", "wmape", "pearson_ic", "spearman_ic", "hit_rate",
    "rolling_mean_eval", "mean_eval_summary", "window_sensitivity", "regime_split",
]
