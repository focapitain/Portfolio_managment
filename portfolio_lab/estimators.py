"""Estimation de μ (rendement espéré) et Σ (covariance) — patron *Strategy*.

Interface commune : rendements (T×N) -> estimation PAR PÉRIODE.
- MeanEstimator.estimate(returns)       -> np.ndarray (N,)
- CovarianceEstimator.estimate(returns) -> pd.DataFrame (N×N)

Rappel méthodologique (RiskMetrics / Michaud) : les estimateurs avancés
améliorent Σ, pas μ. Or l'instabilité du portefeuille tangent est dominée par
l'erreur sur μ. On attend donc le gain le plus net sur le portefeuille à
variance minimale (MVP), qui ne dépend que de Σ.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
#  Moyenne                                                                     #
# --------------------------------------------------------------------------- #
class MeanEstimator(ABC):
    name = "base"

    @abstractmethod
    def estimate(self, returns: pd.DataFrame) -> np.ndarray:
        ...


class SampleMean(MeanEstimator):
    """μ = moyenne empirique. (Extensible : shrinkage, vues Black-Litterman…)"""
    name = "moyenne_empirique"

    def estimate(self, returns):
        return returns.mean().values


class JamesSteinMean(MeanEstimator):
    """Shrinkage de James-Stein vers la moyenne de marché (grand-mean).

    Réduit l'erreur d'estimation sur μ — l'angle mort de Markowitz. Utile pour
    les benchmarks d'objectif qui dépendent de μ (max_sharpe, max_return).
    """
    name = "james_stein"

    def estimate(self, returns):
        mu = returns.mean().values
        grand = float(mu.mean())
        T, N = returns.shape
        var = returns.var(ddof=1).values / max(T, 1)
        denom = float(((mu - grand) ** 2).sum())
        if denom <= 0:
            return mu
        shrink = min(1.0, float(((N - 3) * var.mean()) / denom))
        return grand + (1.0 - shrink) * (mu - grand)


class EWMAMean(MeanEstimator):
    """μ à pondération exponentielle (symétrique de EWMACovariance).

    Donne plus de poids aux rendements récents : μ_i = Σ_t w_t r_{i,t}, avec
    w_t ∝ (1-λ)λ^{T-1-t} normalisés (le plus récent domine). Plus réactif que la
    moyenne empirique aux changements de régime de rendement. λ=0.94 par défaut
    (convention RiskMetrics quotidienne). Reste PAR PÉRIODE comme les autres μ.
    """
    name = "ewma_mean"

    def __init__(self, lam: float = 0.94):
        self.lam = lam

    def estimate(self, returns):
        X = returns.values.astype(float)
        T = X.shape[0]
        w = (1 - self.lam) * self.lam ** np.arange(T - 1, -1, -1)
        w /= w.sum()
        return (X * w[:, None]).sum(axis=0)


class RollingMean(MeanEstimator):
    """μ = moyenne empirique sur les `window` derniers jours seulement.

    Plus réactif que la moyenne sur tout l'historique : capte une dérive récente.
    Reste PAR PÉRIODE. Compromis classique : fenêtre courte = réactif mais bruité,
    fenêtre longue = stable mais périmé.
    """
    name = "rolling_mean"

    def __init__(self, window: int = 63):
        self.window = window

    def estimate(self, returns):
        return returns.iloc[-self.window:].mean().values


class PCAFactorMean(MeanEstimator):
    """μ factoriel STATISTIQUE (returns-only) : débruitage par projection ACP.

    Idée APT/factorielle sans données externes : on ne garde de la moyenne empirique
    que la part qui vit dans le sous-espace des `n_factors` premières composantes
    principales des rendements (les facteurs communs dominants). Le reste — supposé
    idiosyncratique/bruit d'estimation — est écarté. μ̂ = Vₖ Vₖᵀ μ_empirique, où Vₖ
    sont les `n_factors` directions principales (SVD des rendements centrés).
    """
    name = "pca_factor"

    def __init__(self, n_factors: int = 3):
        self.n_factors = n_factors

    def estimate(self, returns):
        X = returns.values.astype(float)
        mu = X.mean(axis=0)
        Xc = X - mu
        # Directions principales = vecteurs singuliers droits des rendements centrés.
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        k = max(1, min(self.n_factors, Vt.shape[0]))
        Vk = Vt[:k].T                      # N × k
        return Vk @ (Vk.T @ mu)            # projection de μ sur le sous-espace factoriel


class BlackLittermanMean(MeanEstimator):
    """Estimateur de Black-Litterman (He & Litterman, 1999).

    Combine le prior CAPM (rendements d'équilibre implicites du marché) avec des
    **vues subjectives de l'investisseur** pour produire un vecteur μ postérieur.

    Formules (espace annualisé) :
      π      = δ · Σ_ann · w_mkt            prior CAPM (rendements d'équilibre)
      Ω      = τ · diag(P Σ_ann Pᵀ)         incertitude sur les vues (He & Litterman)
      M      = (τΣ_ann)⁻¹ + PᵀΩ⁻¹P          précision postérieure
      μ_BL   = M⁻¹[(τΣ_ann)⁻¹π + PᵀΩ⁻¹q]   μ postérieur annuel → converti en par-période

    Paramètres
    ----------
    mcaps : dict {ticker: capitalisation}
        Poids de marché proxy (capitalisations). Toute valeur positive (milliards, rangs…).
    views : dict {ticker: rendement_annuel}
        Vues absolues de l'investisseur, en rendement annuel (ex. {"AAPL": 0.12}).
    delta : float
        Aversion au risque du marché (réf. CAPM, typiquement 2–3).
    tau : float
        Incertitude sur le prior (typiquement 0.01–0.10 ; He & Litterman suggèrent 0.05).
    periods_per_year : int
        252 (daily), 52 (weekly), 12 (monthly). Nécessaire pour annualiser Σ.
    """
    name = "black_litterman"

    def __init__(self, mcaps: dict, views: dict, delta: float = 2.5,
                 tau: float = 0.05, periods_per_year: int = 252):
        self.mcaps = mcaps
        self.views = views
        self.delta = float(delta)
        self.tau = float(tau)
        self.ppy = int(periods_per_year)

    def estimate(self, returns: pd.DataFrame) -> np.ndarray:
        tickers = list(returns.columns)
        N = len(tickers)

        # Σ annualisée (base pour le prior et les vues)
        Sigma_pp  = returns.cov().values.astype(float)   # par période
        Sigma_ann = Sigma_pp * self.ppy

        # Poids de marché normalisés
        mcap_arr = np.array([self.mcaps.get(t, 1.0) for t in tickers], dtype=float)
        w_mkt    = mcap_arr / mcap_arr.sum()

        # Prior CAPM : π = δ Σ_ann w_mkt
        pi_ann = self.delta * Sigma_ann @ w_mkt

        # Vues absolues → matrice P (K×N) et vecteur q_ann (K,)
        view_tickers = [t for t in self.views if t in tickers]
        K = len(view_tickers)
        if K == 0:
            return pi_ann / self.ppy    # aucune vue : retourne le prior CAPM par période

        P     = np.zeros((K, N), dtype=float)
        q_ann = np.zeros(K, dtype=float)
        for k, ticker in enumerate(view_tickers):
            j        = tickers.index(ticker)
            P[k, j]  = 1.0
            q_ann[k] = float(self.views[ticker])

        # Ω : incertitude sur les vues proportionnelle à P Σ_ann Pᵀ
        Omega = self.tau * np.diag(np.diag(P @ Sigma_ann @ P.T))

        # Précision postérieure M et μ_BL (He & Litterman)
        tauSigma_inv = np.linalg.inv(self.tau * Sigma_ann)
        Omega_inv    = np.linalg.inv(Omega)
        M            = tauSigma_inv + P.T @ Omega_inv @ P
        mu_bl_ann    = np.linalg.solve(M, tauSigma_inv @ pi_ann + P.T @ Omega_inv @ q_ann)

        return mu_bl_ann / self.ppy     # converti en rendement par période


# --------------------------------------------------------------------------- #
#  Covariance                                                                  #
# --------------------------------------------------------------------------- #
class CovarianceEstimator(ABC):
    """Interface commune : rendements (T×N) -> matrice de covariance (N×N) PAR PÉRIODE."""
    name = "base"

    @abstractmethod
    def estimate(self, returns: pd.DataFrame) -> pd.DataFrame:
        ...


class EmpiricalCovariance(CovarianceEstimator):
    name = "empirique"

    def estimate(self, returns):
        return returns.cov()


class RollingCovariance(CovarianceEstimator):
    """Covariance équipondérée sur les `window` dernières observations."""
    name = "rolling"

    def __init__(self, window: int = 60):
        self.window = window

    def estimate(self, returns):
        return returns.iloc[-self.window:].cov()


class EWMACovariance(CovarianceEstimator):
    """Covariance EWMA (RiskMetrics). lam=0.94 (daily), 0.97 (monthly).

    Factorisation Σ = (X√w)ᵀ(X√w) => SYMÉTRIQUE et SEMI-DÉFINIE POSITIVE par
    construction. Un seul λ pour toute la matrice => PSD garantie (RiskMetrics, Éq. 5.29).
    """
    name = "ewma"

    def __init__(self, lam: float = 0.94, demean: bool = False):
        self.lam, self.demean = lam, demean

    def estimate(self, returns):
        X = returns.values.astype(float)
        if self.demean:
            X = X - X.mean(axis=0, keepdims=True)
        T = X.shape[0]
        w = (1 - self.lam) * self.lam ** np.arange(T - 1, -1, -1)
        w /= w.sum()
        Xw = X * np.sqrt(w)[:, None]
        return pd.DataFrame(Xw.T @ Xw, index=returns.columns, columns=returns.columns)


class LedoitWolfCovariance(CovarianceEstimator):
    """Shrinkage de Ledoit-Wolf vers une cible structurée. Robuste quand N≫T."""
    name = "ledoit-wolf"

    def estimate(self, returns):
        from sklearn.covariance import LedoitWolf
        S = LedoitWolf().fit(returns.values).covariance_
        return pd.DataFrame(S, index=returns.columns, columns=returns.columns)


class OASCovariance(CovarianceEstimator):
    """Oracle Approximating Shrinkage (Chen et al.). Intensité de shrinkage
    souvent mieux calibrée que Ledoit-Wolf sur petits échantillons gaussiens."""
    name = "oas"

    def estimate(self, returns):
        from sklearn.covariance import OAS
        S = OAS().fit(returns.values).covariance_
        return pd.DataFrame(S, index=returns.columns, columns=returns.columns)


class ConstantCorrelationCovariance(CovarianceEstimator):
    """Modèle à corrélation constante (Elton-Gruber) : on remplace toutes les
    corrélations par leur moyenne, en conservant les variances individuelles.
    Cible de shrinkage classique, très bien conditionnée."""
    name = "corr_constante"

    def __init__(self, shrink: float = 1.0):
        # shrink=1 -> corrélation 100% constante ; 0 -> covariance empirique
        self.shrink = float(np.clip(shrink, 0.0, 1.0))

    def estimate(self, returns):
        S = returns.cov()
        std = np.sqrt(np.diag(S.values))
        corr = S.values / np.outer(std, std)
        N = corr.shape[0]
        off = (corr.sum() - N) / (N * (N - 1)) if N > 1 else 0.0
        target_corr = np.full_like(corr, off)
        np.fill_diagonal(target_corr, 1.0)
        blended = (1 - self.shrink) * corr + self.shrink * target_corr
        cov = blended * np.outer(std, std)
        return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


class IEWMACovariance(CovarianceEstimator):
    """IEWMA — EWMA à DEUX demi-vies (Cvxgrp, « A Simple Method for Predicting
    Covariance Matrices », Johansson et al.).

    Idée : la VOLATILITÉ et la CORRÉLATION évoluent à des vitesses différentes. On
    estime donc :
      1. les volatilités σ_i par EWMA LENTE (demi-vie longue) de r_i² ;
      2. la corrélation R par EWMA RAPIDE (demi-vie courte) des rendements
         standardisés z_t = r_t / σ_t.
    Puis Σ = D R D avec D = diag(σ). La corrélation est obtenue par factorisation
    de Gram R = (Z√w)ᵀ(Z√w) (⇒ SYMÉTRIQUE et PSD), diagonale renormalisée à 1, donc
    Σ est PSD par construction. Plus stable en niveau de risque que l'EWMA simple,
    tout en restant réactif sur la structure de corrélation.
    """
    name = "iewma"

    def __init__(self, vol_halflife: float = 125.0, corr_halflife: float = 21.0,
                 demean: bool = False):
        self.vol_halflife = float(vol_halflife)
        self.corr_halflife = float(corr_halflife)
        self.demean = demean

    @staticmethod
    def _ewma_weights(T: int, halflife: float) -> np.ndarray:
        lam = np.exp(np.log(0.5) / halflife)            # demi-vie -> facteur de déclin
        w = (1 - lam) * lam ** np.arange(T - 1, -1, -1)
        return w / w.sum()

    def estimate(self, returns):
        X = returns.values.astype(float)
        if self.demean:
            X = X - X.mean(axis=0, keepdims=True)
        T = X.shape[0]

        # 1) Volatilités EWMA lentes : σ_i = sqrt( Σ_t w_t r_{i,t}² )
        w_vol = self._ewma_weights(T, self.vol_halflife)
        var = (X ** 2 * w_vol[:, None]).sum(axis=0)
        std = np.sqrt(np.maximum(var, 1e-300))

        # 2) Corrélation EWMA rapide sur les rendements standardisés
        Z = X / std[None, :]
        w_corr = self._ewma_weights(T, self.corr_halflife)
        Zw = Z * np.sqrt(w_corr)[:, None]
        R = Zw.T @ Zw                                    # matrice de Gram -> PSD
        d = np.sqrt(np.maximum(np.diag(R), 1e-300))
        R = R / np.outer(d, d)                           # diagonale renormalisée à 1

        # 3) Recomposition Σ = D R D
        cov = R * np.outer(std, std)
        return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


__all__ = [
    "MeanEstimator", "SampleMean", "JamesSteinMean", "EWMAMean", "RollingMean",
    "PCAFactorMean",
    "CovarianceEstimator", "EmpiricalCovariance", "RollingCovariance",
    "EWMACovariance", "LedoitWolfCovariance", "OASCovariance",
    "ConstantCorrelationCovariance", "IEWMACovariance",
]
