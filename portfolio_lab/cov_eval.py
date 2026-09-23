"""Évaluation STATISTIQUE et PRÉDICTIVE des estimateurs de covariance (hors échantillon).

Complémentaire de `metrics.py` (qui évalue des SÉRIES DE RENDEMENTS de portefeuille).
Ici on évalue la QUALITÉ DE Σ elle-même : capacité prédictive (log-vraisemblance des
rendements futurs), proximité à la covariance réalisée (RMSE/MAE/Frobenius), prévision de
volatilité, conditionnement, et stabilité temporelle.

Conventions
-----------
Toutes les grandeurs sont PAR PÉRIODE (jamais annualisées) : on compare un Σ prédit sur la
fenêtre d'estimation aux rendements/covariance RÉALISÉS sur la fenêtre de détention suivante.
Le moteur `rolling_cov_eval` est strictement HORS ÉCHANTILLON (train < test, sans recouvrement)
et sélectionne les actifs PAR FENÊTRE (mêmes garde-fous anti look-ahead que le backtest).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
#  Métriques ponctuelles (un Σ prédit vs une fenêtre réalisée)                 #
# --------------------------------------------------------------------------- #
def _as_matrix(Sigma) -> np.ndarray:
    return Sigma.values if isinstance(Sigma, pd.DataFrame) else np.asarray(Sigma, dtype=float)


def gaussian_loglik(test_returns, Sigma_pred, demean: bool = False) -> float:
    """Log-vraisemblance gaussienne MOYENNE des rendements de test sous N(0, Σ_pred).

    Mesure la capacité PRÉDICTIVE : un bon Σ rend les rendements futurs « probables ».
    Plus la valeur est HAUTE (moins négative), meilleur est l'estimateur.
    ll_t = -0.5 [ N ln(2π) + ln|Σ| + r_tᵀ Σ⁻¹ r_t ], moyennée sur les observations de test.
    """
    R = test_returns.values if isinstance(test_returns, pd.DataFrame) else np.asarray(test_returns, float)
    S = _as_matrix(Sigma_pred)
    if R.ndim == 1:
        R = R[None, :]
    if demean:
        R = R - R.mean(axis=0, keepdims=True)
    N = S.shape[0]
    # Jitter minimal pour la stabilité numérique de l'inversion / du log-déterminant.
    S = S + 1e-12 * np.eye(N)
    sign, logdet = np.linalg.slogdet(S)
    if sign <= 0:
        return float("nan")
    try:
        sol = np.linalg.solve(S, R.T)            # Σ⁻¹ rᵀ  (N × T_test)
    except np.linalg.LinAlgError:
        return float("nan")
    quad = np.einsum("ti,it->t", R, sol)         # r_tᵀ Σ⁻¹ r_t
    ll = -0.5 * (N * np.log(2 * np.pi) + logdet + quad)
    return float(np.mean(ll))


def realized_covariance(test_returns) -> np.ndarray:
    """Covariance échantillon de la fenêtre réalisée (cible « vraie » du test)."""
    df = test_returns if isinstance(test_returns, pd.DataFrame) else pd.DataFrame(test_returns)
    return df.cov().values


def frobenius_error(Sigma_pred, Sigma_realized) -> float:
    """Norme de Frobenius de l'écart Σ_pred − Σ_realized."""
    return float(np.linalg.norm(_as_matrix(Sigma_pred) - _as_matrix(Sigma_realized), ord="fro"))


def rmse_cov(Sigma_pred, Sigma_realized) -> float:
    """RMSE élément-à-élément entre Σ prédit et Σ réalisé."""
    D = _as_matrix(Sigma_pred) - _as_matrix(Sigma_realized)
    return float(np.sqrt(np.mean(D ** 2)))


def mae_cov(Sigma_pred, Sigma_realized) -> float:
    """MAE élément-à-élément entre Σ prédit et Σ réalisé."""
    D = _as_matrix(Sigma_pred) - _as_matrix(Sigma_realized)
    return float(np.mean(np.abs(D)))


def vol_forecast_error(test_returns, Sigma_pred, weights=None) -> dict:
    """Erreur de PRÉVISION de volatilité de portefeuille (défaut : équipondéré 1/N).

    Compare la vol PRÉDITE √(wᵀΣ_pred w) à la vol RÉALISÉE des rendements de test.
    Renvoie un dict {vol_pred, vol_real, abs_err} (par période).
    """
    R = test_returns.values if isinstance(test_returns, pd.DataFrame) else np.asarray(test_returns, float)
    S = _as_matrix(Sigma_pred)
    N = S.shape[0]
    w = np.ones(N) / N if weights is None else np.asarray(weights, float).flatten()
    vol_pred = float(np.sqrt(max(w @ S @ w, 0.0)))
    port = R @ w
    vol_real = float(np.std(port, ddof=1)) if len(port) > 1 else float("nan")
    return {"vol_pred": vol_pred, "vol_real": vol_real,
            "abs_err": abs(vol_pred - vol_real) if vol_real == vol_real else float("nan")}


def condition_number(Sigma) -> float:
    """Nombre de conditionnement λ_max/λ_min (stabilité numérique de l'inversion)."""
    ev = np.linalg.eigvalsh(_as_matrix(Sigma))
    lo = max(ev.min(), 1e-18)
    return float(ev.max() / lo)


# --------------------------------------------------------------------------- #
#  Moteurs walk-forward (hors échantillon)                                     #
# --------------------------------------------------------------------------- #
def _iter_windows(returns: pd.DataFrame, window: int, step: int):
    """Itère (date_test, train_returns, test_returns) en sélectionnant les actifs
    disponibles PAR FENÊTRE (pas de look-ahead sur la composition)."""
    for start in range(window, len(returns), step):
        train = returns.iloc[start - window:start].dropna(axis=1, how="any")
        test_w = returns.iloc[start:start + step]
        if test_w.empty:
            break
        usable = [c for c in train.columns if c in test_w.columns]
        test = test_w[usable].dropna(axis=1, how="any")
        train = train[test.columns]
        if test.empty or train.shape[1] == 0 or len(test) < 2:
            continue
        yield test.index[0], train, test


def rolling_cov_eval(returns: pd.DataFrame, estimator, *, window: int = 252,
                     step: int = 21, demean: bool = False) -> pd.DataFrame:
    """Évalue un estimateur de Σ en walk-forward hors échantillon.

    Pour chaque fenêtre : Σ_pred = estimator.estimate(train) ; on mesure log-vraisemblance,
    RMSE/MAE/Frobenius vs covariance réalisée du test, erreur de prévision de vol 1/N, et
    conditionnement. Renvoie un DataFrame indexé par date de début de test (une ligne/fenêtre).
    Utiliser `.mean()` pour le résumé, ou `regime_split` pour la lecture crise/hors-crise.
    """
    rows = []
    for date, train, test in _iter_windows(returns, window, step):
        Sigma_pred = estimator.estimate(train)
        Sigma_real = realized_covariance(test)
        vf = vol_forecast_error(test, Sigma_pred)
        rows.append({
            "loglik": gaussian_loglik(test, Sigma_pred, demean=demean),
            "rmse_cov": rmse_cov(Sigma_pred, Sigma_real),
            "mae_cov": mae_cov(Sigma_pred, Sigma_real),
            "frobenius": frobenius_error(Sigma_pred, Sigma_real),
            "vol_pred": vf["vol_pred"],
            "vol_real": vf["vol_real"],
            "vol_abs_err": vf["abs_err"],
            "condition_number": condition_number(Sigma_pred),
            "realized_vol": vf["vol_real"],   # sert au découpage par régime
        })
    if not rows:
        raise ValueError("rolling_cov_eval : aucune fenêtre exploitable (window/step trop grands ?).")
    idx = [d for d, _, _ in _iter_windows(returns, window, step)]
    return pd.DataFrame(rows, index=pd.Index(idx, name="test_start"))


def cov_eval_summary(returns: pd.DataFrame, estimators: dict, *, window: int = 252,
                     step: int = 21, demean: bool = False) -> pd.DataFrame:
    """Résumé (moyenne des fenêtres) pour un dict {nom: estimateur}. Une ligne par estimateur."""
    out = {}
    for name, est in estimators.items():
        df = rolling_cov_eval(returns, est, window=window, step=step, demean=demean)
        out[name] = df.mean(numeric_only=True)
    return pd.DataFrame(out).T


def temporal_stability(returns: pd.DataFrame, estimator, *, window: int = 252,
                       step: int = 21) -> float:
    """Instabilité temporelle d'un estimateur : distance de Frobenius RELATIVE moyenne
    entre deux Σ consécutifs (‖Σ_t − Σ_{t-1}‖_F / ‖Σ_{t-1}‖_F). Plus bas = plus stable."""
    prev = None
    dists = []
    for _, train, _ in _iter_windows(returns, window, step):
        S = _as_matrix(estimator.estimate(train))
        if prev is not None and prev.shape == S.shape:
            denom = np.linalg.norm(prev, ord="fro")
            if denom > 0:
                dists.append(np.linalg.norm(S - prev, ord="fro") / denom)
        prev = S
    return float(np.mean(dists)) if dists else float("nan")


def regime_split(per_window: pd.DataFrame, vol_col: str = "realized_vol",
                 q: float = 0.75) -> pd.DataFrame:
    """Sépare les fenêtres en CRISE (vol réalisée ≥ quantile q) et HORS-CRISE, et renvoie
    la moyenne des métriques par régime. Met en évidence la dégradation en stress."""
    v = per_window[vol_col]
    thr = v.quantile(q)
    crise = per_window[v >= thr].mean(numeric_only=True)
    calme = per_window[v < thr].mean(numeric_only=True)
    return pd.DataFrame({"hors_crise": calme, "crise": crise}).T


__all__ = [
    "gaussian_loglik", "realized_covariance", "frobenius_error", "rmse_cov", "mae_cov",
    "vol_forecast_error", "condition_number", "rolling_cov_eval", "cov_eval_summary",
    "temporal_stability", "regime_split",
]
