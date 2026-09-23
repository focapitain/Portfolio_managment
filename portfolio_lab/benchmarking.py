"""Moteur de benchmarking partagé par les notebooks de comparaison.

Idée directrice (rigueur méthodologique) : on part d'UN `base_config` et de
DONNÉES UNIQUES, et on ne fait varier qu'UNE dimension à la fois via
`dataclasses.replace`. Tout le reste (univers, fréquence, fenêtres, coûts) est
donc contrôlé par construction.

Fonctions clés :
  - run_oos_benchmark : lance un backtest walk-forward pour chaque variante.
  - metrics_table      : agrège les métriques de toutes les variantes.
  - tail_metrics_table : ajoute Sortino/Calmar/VaR/CVaR par variante.
  - weight_stability   : mesure la (in)stabilité temporelle des poids.
  - allocate           : traduit des poids en allocation monétaire (exemple concret).
  - train_test_split_evaluation : ajuste sur le passé, applique tel quel sur le
    futur, compare à l'équipondéré (exemple « passé / futur / résultats réels »).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np
import pandas as pd

from . import backtest as bt
from .backtest_adapter import build_optimize_weights_fn, run_walk_forward
from .config import ExperimentConfig
from .frequencies import periods_per_year, resample_prices
from .metrics import tail_metrics
from .returns import compute_returns


# --------------------------------------------------------------------------- #
#  Benchmark hors échantillon (walk-forward)                                  #
# --------------------------------------------------------------------------- #
def run_oos_benchmark(prices: pd.DataFrame,
                      variants: dict[str, ExperimentConfig],
                      *,
                      backtest_config: Optional[bt.WalkForwardBacktestConfig] = None,
                      benchmark_config: Optional[ExperimentConfig] = None,
                      ) -> dict[str, bt.WalkForwardBacktestResult]:
    """Lance un backtest walk-forward pour chaque variante sur les MÊMES prix.

    `prices` doit déjà être à la fréquence du `backtest_config`.
    """
    results: dict[str, bt.WalkForwardBacktestResult] = {}
    for name, cfg in variants.items():
        results[name] = run_walk_forward(
            prices, cfg, backtest_config=backtest_config,
            benchmark_config=benchmark_config, label=name,
        )
    return results


def metrics_table(results: dict[str, bt.WalkForwardBacktestResult]) -> pd.DataFrame:
    """Concatène les `.metrics` de chaque variante en une table comparative."""
    return pd.DataFrame({name: res.metrics for name, res in results.items()}).T


def tail_metrics_table(results: dict[str, bt.WalkForwardBacktestResult],
                       periods_per_year_: int, rf_annual: float = 0.0,
                       alpha: float = 0.05) -> pd.DataFrame:
    """Métriques de queue (Sortino, Calmar, VaR, CVaR) par variante."""
    rows = {}
    for name, res in results.items():
        r = res.history["strategy_return"]
        rows[name] = tail_metrics(r, periods_per_year_, rf_annual, alpha)
    return pd.DataFrame(rows).T


def weight_stability(results: dict[str, bt.WalkForwardBacktestResult]) -> pd.DataFrame:
    """Mesure la stabilité des poids dans le temps pour chaque variante.

    - turnover_moyen : variation L1 moyenne des poids entre rebalancements.
    - herfindahl_moyen : concentration moyenne (1/N = diversifié, 1 = concentré).
    - n_actifs_effectifs_moyen : 1 / Herfindahl.
    """
    rows = {}
    for name, res in results.items():
        W = res.weights_history.fillna(0.0)
        turnover = W.diff().abs().sum(axis=1).mean()
        herf = (W ** 2).sum(axis=1).mean()
        rows[name] = pd.Series({
            "turnover moyen": float(turnover),
            "Herfindahl moyen": float(herf),
            "N actifs effectifs": float(1.0 / herf) if herf > 0 else np.nan,
        })
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- #
#  Exemple concret : allocation et split passé / futur                        #
# --------------------------------------------------------------------------- #
def allocate(weights: pd.Series, capital: float, min_weight: float = 1e-4) -> pd.DataFrame:
    """Traduit des poids en allocation monétaire (montant à investir par actif)."""
    w = weights[weights.abs() > min_weight].sort_values(ascending=False)
    df = pd.DataFrame({
        "poids (%)": (w * 100).round(2),
        "montant": (w * capital).round(2),
    })
    df.index.name = "actif"
    return df


def train_test_split_evaluation(prices: pd.DataFrame,
                                config: ExperimentConfig,
                                *,
                                split_date: str,
                                capital: float = 100_000.0,
                                frequency: str = "daily",
                                rf_annual: float = 0.04,
                                benchmark_config: Optional[ExperimentConfig] = None,
                                ) -> dict:
    """Exemple « passé / futur / résultats réels ».

    1. Découpe les prix en PASSÉ (≤ split_date) et FUTUR (> split_date).
    2. Estime μ, Σ et optimise les poids UNIQUEMENT sur le passé.
    3. Alloue `capital` selon ces poids.
    4. Applique ces poids FIGÉS sur le futur et mesure la valeur réelle atteinte.
    5. Compare à un benchmark équipondéré 1/N (ou `benchmark_config`).

    C'est une évaluation hors échantillon honnête (aucune info du futur n'entre
    dans la décision), complémentaire du walk-forward (qui ré-optimise).
    """
    ppy = periods_per_year(frequency)
    px = resample_prices(prices.sort_index().ffill().dropna(axis=1, how="any"), frequency)

    past_px = px.loc[:split_date]
    future_px = px.loc[split_date:]
    if len(past_px) < 30 or len(future_px) < 5:
        raise ValueError("Split insuffisant : élargir l'historique ou déplacer split_date.")

    # 1-2) Optimisation sur le passé (mêmes estimateurs que la config)
    train_returns = compute_returns(past_px, config.return_mode)
    optimize_fn = build_optimize_weights_fn(config, ppy)
    raw_w = optimize_fn(train_returns)
    w = raw_w.clip(lower=0.0) if config.long_only else raw_w
    w = w / w.sum()

    # 3) Allocation monétaire
    allocation = allocate(w, capital)

    # 4) Performance réelle sur le futur (poids figés)
    future_returns = compute_returns(future_px, config.return_mode)
    future_returns = future_returns.reindex(columns=w.index).fillna(0.0)
    strat_ret = future_returns @ w
    strat_nav = capital * (1 + strat_ret).cumprod()

    # 5) Benchmark équipondéré (ou config dédiée)
    if benchmark_config is None:
        bench_w = pd.Series(1.0 / len(w), index=w.index)
    else:
        bench_w = build_optimize_weights_fn(benchmark_config, ppy)(train_returns)
        bench_w = bench_w.clip(lower=0.0)
        bench_w = (bench_w / bench_w.sum()).reindex(w.index).fillna(0.0)
    bench_ret = future_returns @ bench_w
    bench_nav = capital * (1 + bench_ret).cumprod()

    def _summ(nav, ret):
        total = float(nav.iloc[-1] / capital - 1)
        ann = float(nav.iloc[-1] / capital) ** (ppy / len(ret)) - 1
        vol = float(ret.std(ddof=1) * np.sqrt(ppy))
        sharpe = (ann - rf_annual) / vol if vol > 0 else np.nan
        mdd = float((nav / nav.cummax() - 1).min())
        return {
            "valeur finale": float(nav.iloc[-1]),
            "P&L": float(nav.iloc[-1] - capital),
            "rendement total (%)": total * 100,
            "rendement annualisé (%)": ann * 100,
            "volatilité (%)": vol * 100,
            "Sharpe": sharpe,
            "max drawdown (%)": mdd * 100,
        }

    comparison = pd.DataFrame({
        "Stratégie": _summ(strat_nav, strat_ret),
        "Équipondéré 1/N": _summ(bench_nav, bench_ret),
    }).T

    return {
        "split_date": split_date,
        "capital": capital,
        "weights": w,
        "allocation": allocation,
        "strategy_nav": strat_nav,
        "benchmark_nav": bench_nav,
        "strategy_return": strat_ret,
        "benchmark_return": bench_ret,
        "comparison": comparison,
        "n_past": len(past_px),
        "n_future": len(future_px),
    }


__all__ = [
    "run_oos_benchmark", "metrics_table", "tail_metrics_table", "weight_stability",
    "allocate", "train_test_split_evaluation",
]
