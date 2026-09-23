"""Optimisation : objectifs et méthodes configurables .

Toutes les grandeurs (μ, Σ, rf) sont ANNUALISÉES en entrée de `solve()`.

Deux familles :
  - Optimiseurs « Markowitz » paramétrés par un `objective`
    (max_sharpe | min_variance | max_return | max_utility | target_return) :
    AnalyticLagrangian, CVXPYOptimizer, ScipyOptimizer, DifferentialEvolutionOptimizer.
  - Allocations « basées risque » de référence (ignorent l'`objective`) :
    EqualWeightOptimizer, InverseVolatilityOptimizer, RiskParityOptimizer,
    MaxDiversificationOptimizer. Elles servent de comparateurs honnêtes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import minimize, differential_evolution

try:
    import cvxpy as cp
except ImportError:
    cp = None

RANDOM_SEED = 42

# Solveur épinglé pour la REPRODUCTIBILITÉ (sinon CVXPY choisit selon l'install → écarts
# au 4ᵉ chiffre entre machines). CLARABEL est livré avec cvxpy>=1.3 et gère QP ET SOCP.
_CVXPY_SOLVER = "CLARABEL"


# --------------------------------------------------------------------------- #
#  Statistiques et fonctions objectif                                         #
# --------------------------------------------------------------------------- #
def portfolio_stats(w, mu, Sigma, rf=0.0):
    """Rendement, volatilité et Sharpe (toutes grandeurs ANNUALISÉES en entrée)."""
    w = np.asarray(w).flatten()
    ret = float(w @ mu)
    vol = float(np.sqrt(max(w @ Sigma @ w, 0.0)))
    sharpe = (ret - rf) / vol if vol > 0 else np.nan
    return ret, vol, sharpe


def _neg_sharpe(w, mu, S, rf):
    v = np.sqrt(w @ S @ w)
    return -(w @ mu - rf) / v if v > 0 else 1e6


def _variance(w, mu, S, rf):
    return w @ S @ w


def _neg_return(w, mu, S, rf):
    return -(w @ mu)


def _neg_utility(w, mu, S, rf, ra):
    return -(w @ mu - 0.5 * ra * (w @ S @ w))


# --------------------------------------------------------------------------- #
#  Interface                                                                   #
# --------------------------------------------------------------------------- #
class Optimizer(ABC):
    name = "base"

    @abstractmethod
    def solve(self, mu, Sigma, *, objective, rf, long_only, w_max,
              risk_aversion, mu_target) -> np.ndarray:
        ...


# --------------------------------------------------------------------------- #
#  Optimiseurs Markowitz                                                       #
# --------------------------------------------------------------------------- #
class AnalyticLagrangian(Optimizer):
    """Forme fermée via multiplicateurs de Lagrange (contraintes d'ÉGALITÉ seulement).

    Constantes A,B,C,D : A=1ᵀΣ⁻¹μ, B=μᵀΣ⁻¹μ, C=1ᵀΣ⁻¹1, D=BC−A².
    Pas de contrainte long-only/plafond (solution potentiellement à découvert).
    """
    name = "lagrangien"

    def solve(self, mu, Sigma, *, objective="min_variance", rf=0.0, long_only=False,
              w_max=1.0, risk_aversion=1.0, mu_target=None):
        Si = np.linalg.inv(Sigma)
        ones = np.ones(len(mu))
        if objective == "min_variance":
            return Si @ ones / (ones @ Si @ ones)
        if objective == "max_sharpe":  # portefeuille tangent
            ex = mu - rf * ones
            return Si @ ex / (ones @ Si @ ex)
        if objective == "target_return":
            A = ones @ Si @ mu
            B = mu @ Si @ mu
            C = ones @ Si @ ones
            D = B * C - A * A
            l1 = (C * mu_target - A) / D
            l2 = (B - A * mu_target) / D
            return Si @ (l1 * mu + l2 * ones)
        raise ValueError(f"Lagrangien ne gère pas l'objectif {objective!r} (pas de forme fermée).")


class CVXPYOptimizer(Optimizer):
    """Optimiseur convexe (QP) — gère toutes les contraintes et passe à l'échelle."""
    name = "cvxpy"

    def solve(self, mu, Sigma, *, objective="max_sharpe", rf=0.0, long_only=True,
              w_max=1.0, risk_aversion=3.0, mu_target=None):
        if cp is None:
            raise ImportError("cvxpy requis : pip install cvxpy")
        N = len(mu)
        Sig = cp.psd_wrap(Sigma)
        if objective == "max_sharpe":
            # Reformulation convexe : min yᵀΣy s.c. (μ−rf)ᵀy=1, y≥0 ; puis w = y/Σyᵢ
            y = cp.Variable(N)
            cons = [(mu - rf) @ y == 1, cp.sum(y) >= 0]
            if long_only:
                cons += [y >= 0]
            if w_max < 1.0:
                cons += [y <= w_max * cp.sum(y)]
            cp.Problem(cp.Minimize(cp.quad_form(y, Sig)), cons).solve(solver=_CVXPY_SOLVER)
            # La reformulation suppose qu'il EXISTE un portefeuille à excès de rendement
            # positif. Sinon (ex. fenêtre baissière où tous les μ < rf) le problème est
            # infaisable : on lève une erreur EXPLICITE plutôt que de renvoyer un NaN
            # silencieux (le backtest la compte alors comme un fallback visible).
            if y.value is None or np.sum(y.value) <= 1e-12:
                raise ValueError(
                    "max_sharpe infaisable : aucun portefeuille à excès de rendement "
                    "positif sur cette fenêtre (tangent inexistant)."
                )
            return y.value / np.sum(y.value)
        w = cp.Variable(N)
        cons = [cp.sum(w) == 1]
        cons += [w >= 0, w <= w_max] if long_only else [w >= -w_max, w <= w_max]
        if objective == "min_variance":
            obj = cp.Minimize(cp.quad_form(w, Sig))
        elif objective == "max_return":
            obj = cp.Maximize(mu @ w)
        elif objective == "max_utility":
            obj = cp.Maximize(mu @ w - 0.5 * risk_aversion * cp.quad_form(w, Sig))
        elif objective == "target_return":
            cons += [mu @ w == mu_target]
            obj = cp.Minimize(cp.quad_form(w, Sig))
        else:
            raise ValueError(objective)
        cp.Problem(obj, cons).solve(solver=_CVXPY_SOLVER)
        return w.value


class ScipyOptimizer(Optimizer):
    """SLSQP — généraliste, tous objectifs + contraintes (gradient numérique)."""
    name = "slsqp"

    def solve(self, mu, Sigma, *, objective="max_sharpe", rf=0.0, long_only=True,
              w_max=1.0, risk_aversion=3.0, mu_target=None):
        N = len(mu)
        w0 = np.ones(N) / N
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        if objective == "target_return":
            cons.append({"type": "eq", "fun": lambda w: w @ mu - mu_target})
        bounds = [(0, w_max)] * N if long_only else [(-w_max, w_max)] * N
        f = {
            "max_sharpe": lambda w: _neg_sharpe(w, mu, Sigma, rf),
            "min_variance": lambda w: _variance(w, mu, Sigma, rf),
            "max_return": lambda w: _neg_return(w, mu, Sigma, rf),
            "target_return": lambda w: _variance(w, mu, Sigma, rf),
            "max_utility": lambda w: _neg_utility(w, mu, Sigma, rf, risk_aversion),
        }[objective]
        return minimize(f, w0, method="SLSQP", bounds=bounds, constraints=cons,
                        options={"maxiter": 500, "ftol": 1e-10}).x


class DifferentialEvolutionOptimizer(Optimizer):
    """Métaheuristique SANS GRADIENT (long-only). Lent => petits univers (N≲30)."""
    name = "differential_evolution"

    def __init__(self, maxiter: int = 200, seed: int = RANDOM_SEED):
        self.maxiter, self.seed = maxiter, seed

    def solve(self, mu, Sigma, *, objective="max_sharpe", rf=0.0, long_only=True,
              w_max=1.0, risk_aversion=3.0, mu_target=None):
        N = len(mu)

        def norm(x):
            x = np.clip(x, 0, None)
            s = x.sum()
            return x / s if s > 0 else np.ones(N) / N

        def f(x):
            w = norm(x)
            return {
                "max_sharpe": _neg_sharpe(w, mu, Sigma, rf),
                "min_variance": _variance(w, mu, Sigma, rf),
                "max_return": _neg_return(w, mu, Sigma, rf),
                "max_utility": _neg_utility(w, mu, Sigma, rf, risk_aversion),
            }[objective]

        res = differential_evolution(f, [(0, 1)] * N, seed=self.seed,
                                     maxiter=self.maxiter, tol=1e-8, polish=True)
        return norm(res.x)


# --------------------------------------------------------------------------- #
#  Allocations basées risque (comparateurs de référence)                      #
# --------------------------------------------------------------------------- #
def _cap_and_renormalize(w, w_max, n_iter=50):
    """Plafonne les poids à w_max et renormalise (somme=1), de façon itérative.

    Variable de contrôle : si les benchmarks ignorent w_max alors que les objectifs
    Markowitz le respectent, la comparaison n'est plus 'toutes choses égales'. On
    applique donc le même plafond. (Infaisable si w_max < 1/N : on renvoie alors 1/N.)
    """
    w = np.asarray(w, dtype=float)
    N = len(w)
    if w_max >= 1.0:
        return w / w.sum()
    if w_max * N < 1.0 - 1e-9:           # plafond incompatible avec somme=1
        return np.ones(N) / N
    for _ in range(n_iter):
        w = w / w.sum()
        over = w > w_max
        if not over.any():
            break
        excess = (w[over] - w_max).sum()
        w[over] = w_max
        under = ~over
        if under.any():
            w[under] += excess * w[under] / w[under].sum()
    return w / w.sum()


class EqualWeightOptimizer(Optimizer):
    """1/N. Référence robuste et difficile à battre hors échantillon (DeMiguel 2009)."""
    name = "equal_weight"

    def solve(self, mu, Sigma, *, objective=None, rf=0.0, long_only=True,
              w_max=1.0, risk_aversion=3.0, mu_target=None):
        N = len(mu)
        return _cap_and_renormalize(np.ones(N) / N, w_max)


class RandomWeightOptimizer(Optimizer):
    """Poids tirés au hasard (Dirichlet) puis plafonnés à w_max — référence « zéro
    intelligence » (le singe de Malkiel). Redessine un portefeuille à CHAQUE rebalancement :
    turnover élevé, donc coûts élevés (un rebalanceur aveugle se pénalise lui-même).
    Reproductible via la graine ; ignore μ et Σ."""
    name = "random_weight"

    def __init__(self, seed: int = RANDOM_SEED):
        self.rng = np.random.default_rng(seed)

    def solve(self, mu, Sigma, *, objective=None, rf=0.0, long_only=True,
              w_max=1.0, risk_aversion=3.0, mu_target=None):
        N = len(mu)
        w = self.rng.dirichlet(np.ones(N))
        return _cap_and_renormalize(w, w_max)


class InverseVolatilityOptimizer(Optimizer):
    """Poids ∝ 1/σ_i. Ignore les corrélations ; n'utilise que la diagonale de Σ."""
    name = "inverse_vol"

    def solve(self, mu, Sigma, *, objective=None, rf=0.0, long_only=True,
              w_max=1.0, risk_aversion=3.0, mu_target=None):
        vol = np.sqrt(np.diag(Sigma))
        inv = 1.0 / np.where(vol > 0, vol, np.inf)
        return _cap_and_renormalize(inv / inv.sum(), w_max)


class RiskParityOptimizer(Optimizer):
    """Parité de risque (Equal Risk Contribution) : chaque actif contribue
    également au risque total. Résolu par SLSQP sur les contributions."""
    name = "risk_parity"

    def solve(self, mu, Sigma, *, objective=None, rf=0.0, long_only=True,
              w_max=1.0, risk_aversion=3.0, mu_target=None):
        N = len(mu)
        Sigma = np.asarray(Sigma)

        def obj(w):
            port_var = w @ Sigma @ w
            mrc = Sigma @ w                      # contribution marginale
            rc = w * mrc                          # contribution au risque
            target = port_var / N
            return float(np.sum((rc - target) ** 2))

        hi = w_max if w_max < 1.0 else 1.0
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        bounds = [(1e-6, hi)] * N
        w0 = np.ones(N) / N
        res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 1000, "ftol": 1e-12})
        return res.x


class MaxDiversificationOptimizer(Optimizer):
    """Ratio de diversification maximal : (wᵀσ) / sqrt(wᵀΣw). Choueifaty-Coignard."""
    name = "max_diversification"

    def solve(self, mu, Sigma, *, objective=None, rf=0.0, long_only=True,
              w_max=1.0, risk_aversion=3.0, mu_target=None):
        N = len(mu)
        Sigma = np.asarray(Sigma)
        vol = np.sqrt(np.diag(Sigma))

        def neg_dr(w):
            denom = np.sqrt(max(w @ Sigma @ w, 1e-18))
            return -(w @ vol) / denom

        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        bounds = [(0.0, w_max)] * N if long_only else [(-w_max, w_max)] * N
        w0 = np.ones(N) / N
        res = minimize(neg_dr, w0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 1000, "ftol": 1e-12})
        return res.x


__all__ = [
    "portfolio_stats", "Optimizer",
    "AnalyticLagrangian", "CVXPYOptimizer", "ScipyOptimizer",
    "DifferentialEvolutionOptimizer",
    "EqualWeightOptimizer", "RandomWeightOptimizer", "InverseVolatilityOptimizer",
    "RiskParityOptimizer", "MaxDiversificationOptimizer",
]
