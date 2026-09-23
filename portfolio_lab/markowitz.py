"""Benchmark institutionnel des formulations de Markowitz (QP/SOCP convexes).

Contenu
-------
- `ConstraintSet` / `PenaltySet` : description déclarative des contraintes et des
  pénalités d'une formulation (budget, bornes, levier, turnover, coûts, robustesse μ).
- `MarkowitzPlusPlus` : optimiseur CVXPY UNIFIÉ qui consomme ces deux objets et un
  portefeuille hérité `w_prev`. Couvre les 5 variantes de la Phase 2.
- `make_variant` : fabrique les 5 formulations standard (standard, linéaire,
  opérationnelle, robuste, ++).
- `markowitz_walk_forward` : harnais walk-forward HORS ÉCHANTILLON DÉDIÉ qui suit
  `w_prev`, le levier et les coûts cumulés. **Ne modifie pas `backtest.py`** (le moteur
  générique reste intact) : Markowitz++ a des besoins propres (mémoire du portefeuille
  précédent) qui justifient un harnais séparé.
- `calibrate_markowitzpp` : calibration par montée de coordonnées (+25 % / −20 %).
- `weight_sensitivity` : carte de stabilité des poids sous perturbation ±x % des inputs.

Convention : μ, Σ ANNUALISÉS en entrée de `solve` (le harnais s'en charge).
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import _cagr, _max_drawdown, _periods_per_year
from .cov_eval import _iter_windows
from .metrics import calmar_ratio, sortino_ratio

try:
    import cvxpy as cp
except ImportError:  # pragma: no cover
    cp = None

# Solveur épinglé (reproductibilité) : CLARABEL gère QP ET SOCP (terme robuste norm2).
_CVXPY_SOLVER = "CLARABEL"


# --------------------------------------------------------------------------- #
#  Description déclarative d'une formulation                                   #
# --------------------------------------------------------------------------- #
@dataclass
class ConstraintSet:
    """Contraintes DURES du problème d'optimisation."""
    long_only: bool = True
    w_min: float | None = None          # borne basse par actif (si long_only -> 0)
    w_max: float = 1.0                  # borne haute par actif
    leverage: float | None = None       # Σ|wᵢ| ≤ leverage  (None = pas de limite)
    turnover_max: float | None = None   # ‖w − w_prev‖₁ ≤ turnover_max
    budget: float = 1.0                 # 1ᵀw = budget


@dataclass
class PenaltySet:
    """Pénalités MOLLES ajoutées à l'objectif (les γ de Markowitz++)."""
    gamma_risk: float = 5.0             # aversion au risque : −½γ wᵀΣw
    gamma_turn: float = 0.0             # −γ‖w−w_prev‖₁ (inertie)
    gamma_trade: float = 0.0            # −γ·(coûts κᵀ|w−w_prev|)
    gamma_lev: float = 0.0              # −γ·(Σ|wᵢ|−1)_+  (pénalise le sur-levier)
    kappa_mu: float = 0.0              # robustesse μ ellipsoïdale : −κ‖Ω^{1/2}w‖₂


# --------------------------------------------------------------------------- #
#  Optimiseur unifié                                                           #
# --------------------------------------------------------------------------- #
class MarkowitzPlusPlus:
    """Optimiseur convexe unifié (QP si pas de robustesse, SOCP sinon).

        max_w  μᵀw − ½γ_risk wᵀΣw − γ_turn‖w−w₀‖₁ − γ_trade·κᵀ|w−w₀|
               − γ_lev (Σ|wᵢ|−1)_+ − κ_μ‖Ω^{1/2}w‖₂
        s.c.   1ᵀw = budget,  bornes,  Σ|wᵢ| ≤ L,  ‖w−w₀‖₁ ≤ T_max
    """
    name = "markowitz++"

    def __init__(self, constraints: ConstraintSet | None = None,
                 penalties: PenaltySet | None = None, tc_bps: float = 10.0):
        self.constraints = constraints or ConstraintSet()
        self.penalties = penalties or PenaltySet()
        self.tc_bps = float(tc_bps)

    def solve(self, mu, Sigma, *, rf: float = 0.0, w_prev=None, mu_uncertainty=None):
        if cp is None:
            raise ImportError("cvxpy requis : pip install cvxpy")
        cs, pen = self.constraints, self.penalties
        mu = np.asarray(mu, float).flatten()
        N = len(mu)
        w = cp.Variable(N)
        Sig = cp.psd_wrap(np.asarray(Sigma, float))

        obj = mu @ w - 0.5 * pen.gamma_risk * cp.quad_form(w, Sig)
        cons = [cp.sum(w) == cs.budget]

        lo = 0.0 if cs.long_only else (cs.w_min if cs.w_min is not None else -cs.w_max)
        cons += [w <= cs.w_max, w >= lo]
        if cs.leverage is not None:
            cons += [cp.norm(w, 1) <= cs.leverage]

        if w_prev is not None:
            w0 = np.asarray(w_prev, float).flatten()
            has_prev = np.abs(w0).sum() > 1e-9          # faux à la 1ʳᵉ allocation (cash)
            trade = cp.norm(w - w0, 1)
            if pen.gamma_turn:
                obj = obj - pen.gamma_turn * trade
            if pen.gamma_trade:
                obj = obj - pen.gamma_trade * (self.tc_bps / 1e4) * trade
            # La contrainte DURE de turnover ne s'applique PAS à la construction initiale
            # depuis le cash (sinon Σ|w|≤T_max est incompatible avec le budget Σw=1).
            if cs.turnover_max is not None and has_prev:
                cons += [trade <= cs.turnover_max]
        if pen.gamma_lev:
            obj = obj - pen.gamma_lev * cp.pos(cp.norm(w, 1) - 1.0)
        if pen.kappa_mu and mu_uncertainty is not None:
            root = np.sqrt(np.maximum(np.asarray(mu_uncertainty, float).flatten(), 0.0))
            obj = obj - pen.kappa_mu * cp.norm(cp.multiply(root, w), 2)

        prob = cp.Problem(cp.Maximize(obj), cons)
        prob.solve(solver=_CVXPY_SOLVER)
        if w.value is None:
            raise ValueError("Markowitz++ infaisable sur cette fenêtre (contraintes incompatibles).")
        return np.asarray(w.value).flatten()


# --------------------------------------------------------------------------- #
#  Les 5 variantes de la Phase 2                                               #
# --------------------------------------------------------------------------- #
def make_variant(name: str, *, w_max: float = 0.10, tc_bps: float = 10.0,
                 gamma_risk: float = 5.0) -> MarkowitzPlusPlus:
    """Fabrique une des 5 formulations standardisées."""
    if name == "standard":          # 1. mean-variance nu (budget + long-only, pas de cap)
        return MarkowitzPlusPlus(ConstraintSet(long_only=True, w_max=1.0),
                                 PenaltySet(gamma_risk=gamma_risk), tc_bps)
    if name == "linear":            # 2. bornes par actif + levier = 1
        return MarkowitzPlusPlus(ConstraintSet(long_only=True, w_max=w_max, leverage=1.0),
                                 PenaltySet(gamma_risk=gamma_risk), tc_bps)
    if name == "operational":       # 3. turnover borné + coûts dans l'objectif
        return MarkowitzPlusPlus(ConstraintSet(long_only=True, w_max=w_max, turnover_max=0.30),
                                 PenaltySet(gamma_risk=gamma_risk, gamma_trade=1.0), tc_bps)
    if name == "robust":            # 4. robustesse μ (ellipsoïde)
        return MarkowitzPlusPlus(ConstraintSet(long_only=True, w_max=w_max),
                                 PenaltySet(gamma_risk=gamma_risk, kappa_mu=2.0), tc_bps)
    if name == "pp":                # 5. Markowitz++ (tout, modéré)
        return MarkowitzPlusPlus(ConstraintSet(long_only=True, w_max=w_max,
                                               turnover_max=0.40, leverage=1.0),
                                 PenaltySet(gamma_risk=gamma_risk, gamma_trade=1.0,
                                            kappa_mu=1.0, gamma_lev=0.0), tc_bps)
    raise ValueError(f"variante inconnue : {name!r}")


# --------------------------------------------------------------------------- #
#  Harnais walk-forward dédié (suit w_prev, levier, coûts) — backtest.py intact #
# --------------------------------------------------------------------------- #
@dataclass
class MarkowitzBacktestResult:
    label: str
    history: pd.DataFrame
    weights_history: pd.DataFrame
    metrics: pd.Series


def _returns_nan_tolerant(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.sort_index().ffill().pct_change().dropna(how="all")


def markowitz_walk_forward(prices: pd.DataFrame, optimizer: MarkowitzPlusPlus, *,
                           mean_estimator, cov_estimator, window: int = 252, step: int = 21,
                           tc_bps: float = 10.0, rf_annual: float = 0.04,
                           frequency: str = "daily", label: str = "markowitz",
                           returns: pd.DataFrame | None = None) -> MarkowitzBacktestResult:
    """Backtest walk-forward HORS ÉCHANTILLON propre à Markowitz++.

    Suit le portefeuille précédent `w_prev` (pour turnover/coûts dans l'objectif), le
    **levier réalisé** (Σ|wᵢ|) et les **coûts cumulés**. Sélection d'actifs par fenêtre.
    """
    ppy = _periods_per_year(frequency)
    rets = returns if returns is not None else _returns_nan_tolerant(prices)
    cost_rate = (tc_bps or 0.0) / 1e4

    nav = 1.0
    prev = pd.Series(dtype=float)           # poids hérités (indexés par ticker)
    rows, whist, idx = [], [], []

    for date, train, test in _iter_windows(rets, window, step):
        cols = list(test.columns)
        mu = np.asarray(mean_estimator.estimate(train), float) * ppy
        Sigma = np.asarray(cov_estimator.estimate(train).values, float) * ppy
        # Var(μ̂_annuel) = ppy·diag(Σ_ann)/T  (≈ diag(Σ_ann) pour une fenêtre d'~1 an).
        # C'est l'incertitude ÉNORME sur μ : avec 1 an de données, se(μ̂) ≈ vol annuelle.
        se2 = np.diag(Sigma) * ppy / max(len(train), 1)
        w_prev_vec = prev.reindex(cols).fillna(0.0).values

        try:
            w = optimizer.solve(mu, Sigma, rf=rf_annual, w_prev=w_prev_vec, mu_uncertainty=se2)
        except Exception as exc:                       # repli : on garde l'allocation précédente
            print(f"[WARN] {label}: fallback (prev/EW) sur {date.date()}: {exc}")
            w = w_prev_vec if w_prev_vec.sum() > 0 else np.ones(len(cols)) / len(cols)
        w = pd.Series(np.asarray(w, float).flatten(), index=cols)

        turnover = float((w - prev.reindex(cols).fillna(0.0)).abs().sum())
        leverage = float(w.abs().sum())
        period = (test @ w).copy()
        if len(period) and cost_rate:
            period.iloc[0] -= turnover * cost_rate

        path = []
        for r in period:
            nav *= 1.0 + r
            path.append(nav)
        rows.append(pd.DataFrame({"strategy_return": period.values}, index=period.index)
                    .assign(turnover=0.0, leverage=leverage, cost=0.0, nav=path))
        rows[-1].iloc[0, rows[-1].columns.get_loc("turnover")] = turnover
        rows[-1].iloc[0, rows[-1].columns.get_loc("cost")] = turnover * cost_rate
        whist.append(pd.DataFrame([w], index=[date]))
        idx.append(date)
        prev = w

    if not rows:
        raise ValueError("Backtest Markowitz vide : pas de fenêtre exploitable.")
    history = pd.concat(rows).sort_index()
    weights = pd.concat(whist).sort_index()
    metrics = _markowitz_metrics(history, ppy, rf_annual)
    return MarkowitzBacktestResult(label, history, weights, metrics)


def _markowitz_metrics(history: pd.DataFrame, ppy: int, rf_annual: float) -> pd.Series:
    ret = history["strategy_return"].dropna()
    nav = history["nav"].dropna()
    cagr = _cagr(nav, ppy)
    vol = float(ret.std(ddof=1) * np.sqrt(ppy))
    sharpe = (cagr - rf_annual) / vol if vol > 0 else np.nan
    rebal_turn = history.loc[history["turnover"] > 0, "turnover"]
    return pd.Series({
        "CAGR (%)": cagr * 100,
        "Vol annualized (%)": vol * 100,
        "Sharpe": sharpe,
        "Sortino": sortino_ratio(ret, ppy, rf_annual),
        "Calmar": calmar_ratio(ret, ppy),
        "Max drawdown (%)": _max_drawdown(nav) * 100,
        "Average turnover": float(rebal_turn.mean()) if len(rebal_turn) else 0.0,
        "Cumulative cost (%)": float(history["cost"].sum()) * 100,
        "Max leverage": float(history["leverage"].max()),
        "Weight stability (Δw)": float(history.loc[history["turnover"] > 0, "turnover"].mean())
                                 if len(rebal_turn) else 0.0,
        "Rebalances": int((history["turnover"] > 0).sum()),
    })


def run_markowitz_benchmark(prices, variants: dict, *, mean_estimator, cov_estimator,
                            window=252, step=21, tc_bps=10.0, rf_annual=0.04,
                            frequency="daily") -> dict:
    """Lance le harnais pour chaque variante {nom: MarkowitzPlusPlus} sur les MÊMES prix."""
    rets = _returns_nan_tolerant(prices)
    out = {}
    for name, opt in variants.items():
        out[name] = markowitz_walk_forward(
            prices, opt, mean_estimator=mean_estimator, cov_estimator=cov_estimator,
            window=window, step=step, tc_bps=tc_bps, rf_annual=rf_annual,
            frequency=frequency, label=name, returns=rets)
    return out


def markowitz_metrics_table(results: dict) -> pd.DataFrame:
    return pd.DataFrame({n: r.metrics for n, r in results.items()}).T


# --------------------------------------------------------------------------- #
#  Calibration par montée de coordonnées (+25 % / −20 %)                       #
# --------------------------------------------------------------------------- #
def default_score(metrics: pd.Series, lam_turnover: float = 0.5,
                  lam_drawdown: float = 0.25) -> float:
    """Objectif composite de calibration : Sharpe pénalisé par turnover et drawdown."""
    return float(metrics["Sharpe"]
                 - lam_turnover * metrics["Average turnover"]
                 - lam_drawdown * abs(metrics["Max drawdown (%)"]) / 100.0)


def calibrate_markowitzpp(prices, template: MarkowitzPlusPlus, *, mean_estimator,
                          cov_estimator, params=("gamma_risk", "gamma_trade"),
                          window=252, step=21, tc_bps=10.0, rf_annual=0.04,
                          frequency="daily", up=1.25, down=0.80, max_passes=3,
                          seed_zero=0.5, score_fn=default_score,
                          bounds=(1e-3, 1e3)) -> tuple:
    """Calibre les hyperparamètres de `template.penalties` sur l'IS fourni (`prices`).

    Montée de coordonnées (ta procédure) : pour chaque paramètre, ×1.25 ; si le score
    s'améliore on garde, sinon ×0.80 ; si ça s'améliore on garde, sinon on revient.
    Balayage répété jusqu'à convergence (`max_passes` ou aucun gain). Renvoie
    (penalties calibrées, meilleur score, historique des évaluations).
    """
    rets = _returns_nan_tolerant(prices)
    pen = deepcopy(template.penalties)
    lo, hi = bounds

    def evaluate(p: PenaltySet) -> float:
        opt = MarkowitzPlusPlus(template.constraints, p, template.tc_bps)
        res = markowitz_walk_forward(prices, opt, mean_estimator=mean_estimator,
                                     cov_estimator=cov_estimator, window=window, step=step,
                                     tc_bps=tc_bps, rf_annual=rf_annual, frequency=frequency,
                                     returns=rets)
        return score_fn(res.metrics)

    best = evaluate(pen)
    history = [("init", dict(vars(pen)), best)]

    for _ in range(max_passes):
        improved = False
        for k in params:
            cur = getattr(pen, k)
            for factor in (up, down):
                trial = seed_zero if cur == 0 else float(np.clip(cur * factor, lo, hi))
                if trial == cur:
                    continue
                setattr(pen, k, trial)
                s = evaluate(pen)
                history.append((k, dict(vars(pen)), s))
                if s > best + 1e-9:
                    best, cur, improved = s, trial, True
                    break
                setattr(pen, k, cur)        # revenir
        if not improved:
            break
    return pen, best, history


# --------------------------------------------------------------------------- #
#  Analyse de sensibilité (±x % sur les inputs)                                #
# --------------------------------------------------------------------------- #
def weight_sensitivity(mu, Sigma, optimizer: MarkowitzPlusPlus, *, pct: float = 0.20,
                       w_prev=None, n_dirs: int = 12, seed: int = 42) -> dict:
    """Stabilité des poids sous perturbation ±`pct` de μ et Σ.

    Renvoie {mean_l1, max_l1, base_weights} : variation L1 moyenne/maximale des poids
    optimaux face à des perturbations aléatoires de ±pct sur μ et sur l'échelle de Σ.
    Un portefeuille robuste a un mean_l1 FAIBLE (peu sensible aux erreurs d'estimation).
    """
    rng = np.random.default_rng(seed)
    mu = np.asarray(mu, float).flatten()
    se2 = np.diag(Sigma)            # Var(μ̂_ann) ≈ diag(Σ_ann) pour une fenêtre d'~1 an
    w0 = optimizer.solve(mu, Sigma, w_prev=w_prev, mu_uncertainty=se2)
    dists = []
    for _ in range(n_dirs):
        dmu = mu * (1.0 + pct * rng.uniform(-1, 1, len(mu)))
        scale = 1.0 + pct * rng.uniform(-1, 1)
        try:
            w = optimizer.solve(dmu, Sigma * scale, w_prev=w_prev, mu_uncertainty=se2)
        except Exception:
            continue
        dists.append(float(np.abs(w - w0).sum()))
    return {"mean_l1": float(np.mean(dists)) if dists else np.nan,
            "max_l1": float(np.max(dists)) if dists else np.nan,
            "base_weights": w0}


__all__ = [
    "ConstraintSet", "PenaltySet", "MarkowitzPlusPlus", "make_variant",
    "MarkowitzBacktestResult", "markowitz_walk_forward", "run_markowitz_benchmark",
    "markowitz_metrics_table", "default_score", "calibrate_markowitzpp",
    "weight_sensitivity",
]
