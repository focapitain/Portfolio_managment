"""Couche SERVICE : pont caché entre l'UI Streamlit et `portfolio_lab`.

Règle d'or de l'architecture démo : **l'UI n'appelle jamais `portfolio_lab` directement**.
Elle passe par ce module qui (a) traduit des paramètres primitifs (chaînes/nombres) en
objets de recherche, (b) lance les calculs lourds (téléchargement, backtest CVXPY) **une
seule fois** grâce au cache, (c) renvoie des structures simples (DataFrames) que les pages
consomment. Les modules de `portfolio_lab` restent strictement inchangés.

Le décorateur de cache est `st.cache_data` si Streamlit tourne, sinon une identité — ainsi
ce module est **testable hors navigateur** (CI / sanity-checks).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from . import registry as reg
from .registry import pl  # portfolio_lab (déjà sur le sys.path via registry)
from portfolio_lab.backtest import WalkForwardBacktestConfig


# --------------------------------------------------------------------------- #
#  Journalisation TERMINAL (traçabilité du déroulement)                        #
# --------------------------------------------------------------------------- #
# Comme les fonctions ci-dessous sont mises en cache, ces messages n'apparaissent que
# lors d'un VRAI calcul (cache miss) : on voit donc dans le terminal chaque étape réellement
# exécutée (chargement des prix, backtest, comparaison) avec sa durée.
log = logging.getLogger("demo")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s │ demo │ %(message)s", "%H:%M:%S"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)
    log.propagate = False


# --------------------------------------------------------------------------- #
#  Cache : st.cache_data si dispo, sinon no-op (tests hors Streamlit)          #
# --------------------------------------------------------------------------- #
try:
    import streamlit as st

    def cache(func=None, **kwargs):
        return st.cache_data(func, show_spinner=False, **kwargs) if func else \
            st.cache_data(show_spinner=False, **kwargs)
except Exception:                                       # pragma: no cover (hors Streamlit)
    def cache(func=None, **kwargs):
        if func is not None and callable(func):
            return func
        def deco(f):
            return f
        return deco


# --------------------------------------------------------------------------- #
#  Résultat structuré d'un backtest (tout en DataFrames picklables)            #
# --------------------------------------------------------------------------- #
@dataclass
class BacktestBundle:
    history: pd.DataFrame          # strategy_nav, benchmark_nav, strategy_return, turnover, …
    weights: pd.DataFrame          # date de rebalancement × ticker
    metrics: pd.Series             # CAGR, Vol, Sharpe, MaxDD, turnover, fallback…
    prices: pd.DataFrame           # prix (ré-échantillonnés) réellement utilisés
    returns: pd.DataFrame          # rendements par période
    params: dict = field(default_factory=dict)
    index_nav: Optional[pd.Series] = None   # NAV de l'indice S&P 500 (base 1), aligné sur l'OOS
    random_nav: Optional[pd.Series] = None  # NAV d'un portefeuille à POIDS ALÉATOIRES (base 1)

    # --- raccourcis lecture (pas de calcul lourd) ---
    @property
    def nav(self) -> pd.Series:
        return self.history["strategy_nav"]

    @property
    def benchmark_nav(self) -> pd.Series:
        return self.history["benchmark_nav"]

    @property
    def n_effective(self) -> int:
        return int(self.prices.shape[1])

    @property
    def rebalance_dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.weights.index)


# --------------------------------------------------------------------------- #
#  Données                                                                     #
# --------------------------------------------------------------------------- #
@cache
def load_prices(universe_kind: str, n: int, start: str, end: str, frequency: str,
                use_synthetic: bool, seed: int = 42,
                tickers: Optional[tuple] = None) -> pd.DataFrame:
    """Prix nettoyés et ré-échantillonnés à la fréquence demandée (mis en cache).

    `tickers` (tuple) force un univers explicite (ex. univers curaté de la page Rejouer),
    sinon on passe par le sélecteur `universe_kind`.
    """
    t0 = time.time()
    src = "SYNTHÉTIQUE" if use_synthetic else "yfinance"
    log.info("Chargement prix [%s] univers=%s n=%s période=%s→%s freq=%s …",
             src, universe_kind, n, start, end, frequency)
    constituents = pl.get_sp500_constituents()
    if tickers is not None:
        universe = [t for t in tickers if t in constituents.index] or list(tickers)
    else:
        universe = reg.make_universe(universe_kind, n, seed).select(constituents)

    if use_synthetic:
        sub = constituents.reindex(universe).dropna(how="all")
        if sub.empty:
            sub = constituents.head(max(n, 10))
        prices = pl.make_synthetic_prices(sub, start, end, seed=seed)
    else:
        prices = pl.clean_prices(pl.download_prices(universe, start, end))

    prices = pl.resample_prices(prices.sort_index().ffill(), frequency)
    prices = prices.dropna(how="all")
    log.info("→ prix prêts : %d actifs × %d points (%.1fs)",
             prices.shape[1], prices.shape[0], time.time() - t0)
    return prices


# --------------------------------------------------------------------------- #
#  Indice de marché réel (S&P 500) — coût d'opportunité du gérant passif       #
# --------------------------------------------------------------------------- #
@cache
def load_index_nav(start: str, end: str, frequency: str, use_synthetic: bool,
                   symbol: str = "^GSPC") -> Optional[pd.Series]:
    """Série de PRIX (base brute) de l'indice S&P 500 sur la période, ré-échantillonnée.

    Renvoie `None` en mode synthétique (pas d'indice réel hors-ligne) ou si le
    téléchargement échoue : l'appelant trace alors simplement stratégie vs 1/N.
    La normalisation en base 1 est faite à l'alignement (cf. `run_backtest`).
    """
    if use_synthetic:
        return None
    try:
        px = pl.clean_prices(pl.download_prices([symbol], start, end))
        if px.empty:
            return None
        px = pl.resample_prices(px.sort_index().ffill(), frequency).dropna(how="all")
        s = px.iloc[:, 0].dropna()
        log.info("→ indice %s prêt : %d points", symbol, len(s))
        return s if len(s) > 1 else None
    except Exception as e:                               # pragma: no cover (réseau)
        log.info("indice %s indisponible (%s) — overlay ignoré", symbol, type(e).__name__)
        return None


def _aligned_index_nav(index_px: Optional[pd.Series],
                       hist_index: pd.Index) -> Optional[pd.Series]:
    """Aligne les prix de l'indice sur l'axe temporel de l'OOS et renormalise en base 1
    (même base que `strategy_nav`/`benchmark_nav`) pour une comparaison visuelle honnête."""
    if index_px is None:
        return None
    aligned = index_px.reindex(hist_index).ffill().dropna()
    if len(aligned) < 2:
        return None
    return (aligned / aligned.iloc[0]).rename("sp500_nav")


# --------------------------------------------------------------------------- #
#  Backtest principal (un seul run alimente les pages 1→5, 7)                  #
# --------------------------------------------------------------------------- #
@cache
def run_backtest(universe_kind: str, n: int, start: str, end: str, frequency: str,
                 mu_name: str, cov_name: str, method_name: str, w_max: float,
                 tc_bps: float, rf: float, est_window: int, holding: int,
                 use_synthetic: bool, seed: int = 42,
                 tickers: Optional[tuple] = None) -> BacktestBundle:
    """Lance un backtest walk-forward complet et renvoie un `BacktestBundle` (caché)."""
    t0 = time.time()
    log.info("▶ RUN backtest : μ=%s · Σ=%s · %s · w_max=%.0f%% · coûts=%.0fbps · win=%d/hold=%d",
             mu_name, cov_name, method_name, w_max * 100, tc_bps, est_window, holding)
    prices = load_prices(universe_kind, n, start, end, frequency, use_synthetic, seed, tickers)
    returns = pl.compute_returns(prices, "simple")

    method = reg.PORTFOLIO_METHODS[method_name]
    cfg = pl.ExperimentConfig(
        universe=reg.make_universe(universe_kind, n, seed),
        frequency=frequency, return_mode="simple",
        cov_estimator=reg.COV_ESTIMATORS[cov_name](),
        mean_estimator=reg.MU_ESTIMATORS[mu_name](),
        objective=method["objective"], optimizer=method["optimizer"](),
        long_only=True, w_max=w_max, risk_free_annual=rf,
    )
    bt_cfg = WalkForwardBacktestConfig(
        estimation_window=est_window, holding_period=holding,
        transaction_cost_bps=tc_bps, long_only=True, frequency=frequency,
        risk_free_annual=rf,
    )
    res = pl.run_walk_forward(prices, cfg, backtest_config=bt_cfg, label=method_name)
    fb = int(res.metrics.get("Fallback count", 0))
    log.info("✓ backtest terminé : %d rebalancements · Sharpe=%.2f · %d fallback(s) · %.1fs",
             len(res.weights_history), float(res.metrics.get("Sharpe", float("nan"))), fb,
             time.time() - t0)
    index_px = load_index_nav(start, end, frequency, use_synthetic)
    index_nav = _aligned_index_nav(index_px, res.history.index)

    # Benchmark « zéro intelligence » : poids aléatoires (Dirichlet), redessinés à chaque
    # rebalancement, mêmes contraintes/coûts que la stratégie. Aligné en base 1 sur l'OOS.
    random_nav = None
    try:
        from dataclasses import replace
        cfg_rand = replace(cfg, optimizer=pl.RandomWeightOptimizer(seed=seed))
        res_rand = pl.run_walk_forward(prices, cfg_rand, backtest_config=bt_cfg,
                                       label="Poids aléatoires")
        rn = res_rand.history["strategy_nav"].reindex(res.history.index).ffill().dropna()
        if len(rn) > 1:
            random_nav = (rn / rn.iloc[0]).rename("random_nav")
    except Exception as e:                               # pragma: no cover
        log.info("benchmark aléatoire indisponible (%s)", type(e).__name__)

    return BacktestBundle(
        history=res.history, weights=res.weights_history.fillna(0.0), metrics=res.metrics,
        prices=prices, returns=returns, index_nav=index_nav, random_nav=random_nav,
        params=dict(universe_kind=universe_kind, n=n, start=start, end=end,
                    frequency=frequency, mu=mu_name, cov=cov_name, method=method_name,
                    w_max=w_max, tc_bps=tc_bps, rf=rf, est_window=est_window,
                    holding=holding, use_synthetic=use_synthetic),
    )


# --------------------------------------------------------------------------- #
#  Comparaison de méthodes (page 6)                                            #
# --------------------------------------------------------------------------- #
@cache
def compare_methods(kind: str, names: tuple, universe_kind: str, n: int, start: str,
                    end: str, frequency: str, mu_name: str, cov_name: str,
                    method_name: str, w_max: float, tc_bps: float, rf: float,
                    est_window: int, holding: int, use_synthetic: bool,
                    seed: int = 42) -> dict:
    """Compare plusieurs covariances (kind='covariance') OU plusieurs portefeuilles
    (kind='portfolio'), tout le reste figé. Renvoie un dict de DataFrames (picklable).
    """
    from dataclasses import replace

    t0 = time.time()
    log.info("▶ COMPARAISON [%s] : %s", kind, ", ".join(names))
    prices = load_prices(universe_kind, n, start, end, frequency, use_synthetic, seed)
    returns = pl.compute_returns(prices, "simple")
    base_method = reg.PORTFOLIO_METHODS[method_name]
    base = pl.ExperimentConfig(
        universe=reg.make_universe(universe_kind, n, seed), frequency=frequency,
        return_mode="simple", cov_estimator=reg.COV_ESTIMATORS[cov_name](),
        mean_estimator=reg.MU_ESTIMATORS[mu_name](), objective=base_method["objective"],
        optimizer=base_method["optimizer"](), long_only=True, w_max=w_max,
        risk_free_annual=rf,
    )
    bt_cfg = WalkForwardBacktestConfig(estimation_window=est_window, holding_period=holding,
                                       transaction_cost_bps=tc_bps, frequency=frequency,
                                       risk_free_annual=rf)

    variants = {}
    for nm in names:
        if kind == "covariance":
            variants[nm] = replace(base, cov_estimator=reg.COV_ESTIMATORS[nm]())
        else:                                            # portfolio
            m = reg.PORTFOLIO_METHODS[nm]
            variants[nm] = replace(base, optimizer=m["optimizer"](), objective=m["objective"])

    results = pl.run_oos_benchmark(prices, variants, backtest_config=bt_cfg)
    metrics = pl.metrics_table(results)
    tails = pl.tail_metrics_table(results, periods_per_year_=pl.periods_per_year(frequency),
                                  rf_annual=rf)
    stability = pl.weight_stability(results)
    navs = {nm: results[nm].history["strategy_nav"] for nm in names}

    out = {"metrics": metrics, "tails": tails, "stability": stability, "navs": navs}

    # Qualité statistique des Σ comparées (conditionnement, log-vraisemblance) — page 6.
    # `rolling_cov_eval` a besoin d'une fenêtre de test ≥ 2 pour la covariance réalisée :
    # on impose donc un pas d'évaluation ≥ 4 (indépendant du holding du backtest). Non
    # bloquant : si l'historique est trop court, on saute proprement cette table.
    if kind == "covariance":
        try:
            ests = {nm: reg.COV_ESTIMATORS[nm]() for nm in names}
            out["cov_quality"] = pl.cov_eval_summary(returns, ests, window=est_window,
                                                     step=max(holding, 4))
        except Exception:
            pass
    log.info("✓ comparaison terminée : %d variantes · %.1fs", len(names), time.time() - t0)
    return out


@cache
def decision_table(universe_kind: str, n: int, start: str, end: str, frequency: str,
                   mu_name: str, cov_name: str, method_name: str, w_max: float, tc_bps: float,
                   rf: float, est_window: int, holding: int, use_synthetic: bool,
                   seed: int = 42, tickers: Optional[tuple] = None) -> pd.DataFrame:
    """Pour CHAQUE rebalancement, reconstitue les INTRANTS de la décision : volatilité
    estimée du portefeuille √(wᵀΣw) annualisée, corrélation moyenne de la fenêtre, turnover,
    et un flag rebalancement. Sert le panneau « Décision du portefeuille » (page Simulation).
    """
    t0 = time.time()
    b = run_backtest(universe_kind, n, start, end, frequency, mu_name, cov_name, method_name,
                     w_max, tc_bps, rf, est_window, holding, use_synthetic, seed, tickers)
    returns, weights = b.returns, b.weights
    cov_est = reg.COV_ESTIMATORS[cov_name]()
    ppy = pl.periods_per_year(frequency)
    idx = returns.index
    prev = None
    rows, dates = [], []
    for d in weights.index:
        pos = int(idx.get_indexer([d])[0])
        if pos < 0:
            pos = int(idx.searchsorted(d))
        if pos < 2:
            continue
        train = returns.iloc[max(0, pos - est_window):pos].dropna(axis=1, how="any")
        if train.shape[1] < 2 or len(train) < 3:
            continue
        w = weights.loc[d].reindex(train.columns).fillna(0.0).values
        Sigma = cov_est.estimate(train).values * ppy
        est_vol = float(np.sqrt(max(w @ Sigma @ w, 0.0)))
        c = train.corr().values
        nn = c.shape[0]
        avg_corr = float((c.sum() - nn) / (nn * (nn - 1))) if nn > 1 else np.nan
        wn = weights.loc[d]
        turn = float((wn - (prev.reindex(wn.index).fillna(0.0) if prev is not None else 0.0)).abs().sum())
        rows.append({"est_vol": est_vol, "avg_corr": avg_corr, "turnover": turn,
                     "rebalance": bool(turn > 0.01)})
        dates.append(d)
        prev = wn
    log.info("✓ decision_table : %d rebalancements analysés (%.1fs)", len(rows), time.time() - t0)
    return pd.DataFrame(rows, index=pd.Index(dates, name="date"))


def report_markdown(universe_kind: str, n: int, start: str, end: str, frequency: str,
                    mu_name: str, cov_name: str, method_name: str, w_max: float, tc_bps: float,
                    rf: float, est_window: int, holding: int, use_synthetic: bool,
                    seed: int = 42, tickers: Optional[tuple] = None) -> str:
    """Construit un compte rendu Markdown HORODATÉ de la simulation courante (sans écrire de
    fichier) : paramètres lisibles, métriques OOS, contrôles de santé (dégénérescence,
    fallbacks). Réutilise `RunReport` de `portfolio_lab`. Destiné au bouton de téléchargement.
    """
    from portfolio_lab.reporting import RunReport
    from .constraints import degeneracy_check

    b = run_backtest(universe_kind, n, start, end, frequency, mu_name, cov_name, method_name,
                     w_max, tc_bps, rf, est_window, holding, use_synthetic, seed, tickers)

    rep = RunReport("demo", title="Rapport de simulation — Portfolio Lab")
    rep.param("Source de données", "SYNTHÉTIQUE" if use_synthetic else "réelle (yfinance)")
    rep.param("Univers", f"{reg.label(universe_kind)} — {b.n_effective} actifs")
    rep.param("Période", f"{start} → {end} ({reg.label(frequency)})")
    rep.param("Rendement μ", reg.label(mu_name))
    rep.param("Covariance Σ", reg.label(cov_name))
    rep.param("Méthode", reg.label(method_name))
    rep.param("Poids max / action", f"{w_max:.0%}")
    rep.param("Frais par transaction", f"{tc_bps:.0f} bps")
    rep.param("Taux sans risque (annuel)", f"{rf:.1%}")
    rep.param("Historique observé", f"{est_window} périodes")
    rep.param("Fréquence de rebalancement", f"{holding} période(s)")

    # Métriques OOS (arrondi sûr : on n'arrondit que les valeurs numériques).
    m = b.metrics.apply(lambda v: round(v, 3) if isinstance(v, (int, float)) else v).to_frame("Valeur")
    rep.add_table("Métriques de performance (hors échantillon)", m, round_to=None)

    dg = degeneracy_check(b.n_effective, w_max)
    rep.add_check("Portefeuille non dégénéré (N×w_max > 1.2)", dg["ok"], dg["message"])
    fb = int(b.metrics.get("Fallback count", 0))
    rep.add_check("Aucun fallback d'optimisation", fb == 0,
                  f"{fb} fenêtre(s) où l'optimisation était infaisable.")
    rep.add_note("_Rapport généré depuis la démo interactive Portfolio Lab._")
    return rep.to_markdown()


@cache
def asset_sectors(tickers: tuple) -> pd.Series:
    """Mapping ticker → secteur GICS (pour treemap/sunburst). Robuste hors-ligne."""
    try:
        constituents = pl.get_sp500_constituents()
        s = constituents["sector"].reindex(list(tickers))
    except Exception:
        s = pd.Series(index=list(tickers), dtype=object)
    return s.fillna("Autre")


__all__ = ["BacktestBundle", "load_prices", "load_index_nav", "run_backtest",
           "compare_methods", "decision_table", "report_markdown", "asset_sectors", "cache"]
