from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

try:
	import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
	plt = None


@dataclass
class WalkForwardBacktestConfig:
	estimation_window: int = 252
	holding_period: int = 21
	transaction_cost_bps: float = 10.0
	long_only: bool = True
	frequency: str = "daily"
	fallback: str = "equal_weight"
	risk_free_annual: float = 0.0   # utilisé pour le Sharpe (stratégie ET benchmark)


@dataclass
class WalkForwardBacktestResult:
	label: str
	history: pd.DataFrame
	weights_history: pd.DataFrame
	metrics: pd.Series


def _normalize_weights(weights: np.ndarray | pd.Series, index: pd.Index, long_only: bool) -> pd.Series:
	w = pd.Series(np.asarray(weights, dtype=float).flatten(), index=index)
	w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0)
	if long_only:
		w = w.clip(lower=0.0)
	total = float(w.sum())
	if total <= 0:
		return pd.Series(1.0 / len(index), index=index)
	return w / total


def _max_drawdown(nav: pd.Series) -> float:
	running_max = nav.cummax()
	drawdown = nav / running_max - 1.0
	return float(drawdown.min())


def _cagr(nav: pd.Series, periods_per_year: int) -> float:
	if nav.empty:
		return np.nan
	return float(nav.iloc[-1] ** (periods_per_year / len(nav)) - 1.0)


def performance_summary(history: pd.DataFrame, periods_per_year: int,
                        rf_annual: float = 0.0) -> pd.Series:
	strategy_returns = history["strategy_return"].dropna()
	benchmark_returns = history["benchmark_return"].dropna()
	nav = history["strategy_nav"].dropna()
	benchmark_nav = history["benchmark_nav"].dropna()

	if len(strategy_returns) == 0:
		raise ValueError("Backtest empty: no valid out-of-sample periods.")

	total_return = float(nav.iloc[-1] - 1.0)
	benchmark_total_return = float(benchmark_nav.iloc[-1] - 1.0)
	annualized_return = _cagr(nav, periods_per_year)
	annualized_vol = float(strategy_returns.std(ddof=1) * np.sqrt(periods_per_year))
	# Sharpe = (rendement annualisé EXCÉDENTAIRE) / volatilité. On soustrait bien rf.
	sharpe = (annualized_return - rf_annual) / annualized_vol if annualized_vol > 0 else np.nan
	benchmark_annualized_return = _cagr(benchmark_nav, periods_per_year)
	benchmark_annualized_vol = float(benchmark_returns.std(ddof=1) * np.sqrt(periods_per_year))
	benchmark_sharpe = ((benchmark_annualized_return - rf_annual) / benchmark_annualized_vol
	                    if benchmark_annualized_vol > 0 else np.nan)
	active_return = strategy_returns - benchmark_returns.reindex(strategy_returns.index).fillna(0.0)
	tracking_error = float(active_return.std(ddof=1) * np.sqrt(periods_per_year)) if len(active_return) > 1 else np.nan

	bench_turnover = (float(history["benchmark_turnover"].fillna(0.0).mean())
	                  if "benchmark_turnover" in history.columns else 0.0)
	fallback_count = (int(history["fallback"].sum())
	                  if "fallback" in history.columns else 0)

	return pd.Series(
		{
			"Total return (%)": total_return * 100,
			"CAGR (%)": annualized_return * 100,
			"Vol annualized (%)": annualized_vol * 100,
			"Sharpe": sharpe,
			"Max drawdown (%)": _max_drawdown(nav) * 100,
			"Benchmark total return (%)": benchmark_total_return * 100,
			"Benchmark CAGR (%)": benchmark_annualized_return * 100,
			"Benchmark vol annualized (%)": benchmark_annualized_vol * 100,
			"Benchmark Sharpe": benchmark_sharpe,
			"Tracking error (%)": tracking_error * 100 if pd.notna(tracking_error) else np.nan,
			"Average turnover": float(history["turnover"].fillna(0.0).mean()),
			"Average benchmark turnover": bench_turnover,
			"Rebalance count": int(history["rebalance"].sum()),
			"Fallback count": fallback_count,
		}
	)


def _default_compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
	# how="all" (et non "any") : on préserve les NaN PAR COLONNE pour que la sélection
	# d'actifs disponibles puisse se faire par fenêtre (cf. boucle walk-forward).
	return prices.pct_change().dropna(how="all")


def _periods_per_year(frequency: str) -> int:
	mapping = {"daily": 252, "weekly": 52, "monthly": 12, "yearly": 1}
	if frequency not in mapping:
		raise ValueError(f"Unknown frequency: {frequency!r}")
	return mapping[frequency]


def walk_forward_backtest(
	prices: pd.DataFrame,
	optimize_weights_fn: Callable[[pd.DataFrame], np.ndarray | pd.Series],
	*,
	backtest_config: Optional[WalkForwardBacktestConfig] = None,
	compute_returns_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
	benchmark_weights_fn: Optional[Callable[[pd.DataFrame], np.ndarray | pd.Series]] = None,
	label: Optional[str] = None,
) -> WalkForwardBacktestResult:
	bt_cfg = backtest_config or WalkForwardBacktestConfig()
	returns_fn = compute_returns_fn or _default_compute_returns

	# NB look-ahead : on NE supprime PAS globalement les colonnes ayant un NaN sur toute
	# la période (cela utiliserait la connaissance du futur sur quels actifs « survivent »).
	# On garde toutes les colonnes ; la sélection des actifs disponibles se fait PAR FENÊTRE
	# dans la boucle. Biais de survivance RÉSIDUEL : la composition de l'univers reste celle
	# d'aujourd'hui (cf. data.get_sp500_constituents) — correction point-in-time hors périmètre.
	px = prices.sort_index().ffill()
	returns = returns_fn(px)
	if returns.empty:
		raise ValueError("No returns computed from the provided prices.")

	window = bt_cfg.estimation_window
	step = bt_cfg.holding_period
	if len(returns) <= window + step:
		raise ValueError("Not enough observations for backtest. Reduce window or extend history.")

	strategy_nav = 1.0
	benchmark_nav = 1.0
	prev_weights = pd.Series(0.0, index=returns.columns)
	prev_benchmark_weights = pd.Series(0.0, index=returns.columns)
	history_frames: list[pd.DataFrame] = []
	weights_history: list[pd.DataFrame] = []
	cost_rate = (bt_cfg.transaction_cost_bps or 0.0) / 1e4

	for start in range(window, len(returns), step):
		# Sélection PAR FENÊTRE : on ne garde que les actifs sans NaN sur la fenêtre
		# d'estimation ET sur la fenêtre de détention (actif présent à ce moment-là).
		train_returns = returns.iloc[start - window : start].dropna(axis=1, how="any")
		test_window = returns.iloc[start : start + step]
		if test_window.empty:
			break
		usable = [c for c in train_returns.columns if c in test_window.columns]
		test_returns = test_window[usable].dropna(axis=1, how="any")
		train_returns = train_returns[test_returns.columns]
		if test_returns.empty or train_returns.shape[1] == 0:
			continue

		fallback_used = 0
		try:
			weights = optimize_weights_fn(train_returns)
			weights = _normalize_weights(weights, train_returns.columns, bt_cfg.long_only)
		except Exception as exc:
			fallback_used = 1
			if bt_cfg.fallback == "previous" and prev_weights.abs().sum() > 0:
				weights = _normalize_weights(prev_weights, prev_weights.index, bt_cfg.long_only)
			else:
				print(f"[WARN] fallback equal-weight for {label or 'strategy'}: {exc}")
				weights = pd.Series(1.0 / len(test_returns.columns), index=test_returns.columns)

		weights = _normalize_weights(weights.reindex(test_returns.columns).fillna(0.0), test_returns.columns, bt_cfg.long_only)
		if benchmark_weights_fn is None:
			benchmark_weights = pd.Series(1.0 / len(test_returns.columns), index=test_returns.columns)
		else:
			benchmark_raw = benchmark_weights_fn(train_returns)
			benchmark_weights = _normalize_weights(benchmark_raw, train_returns.columns, True)
			benchmark_weights = _normalize_weights(
				benchmark_weights.reindex(test_returns.columns).fillna(0.0), test_returns.columns, True
			)
		turnover = float((weights - prev_weights.reindex(weights.index).fillna(0.0)).abs().sum())
		# Coûts SYMÉTRIQUES : le benchmark 1/N rebalance aussi, il paie donc ses coûts.
		benchmark_turnover = float(
			(benchmark_weights - prev_benchmark_weights.reindex(benchmark_weights.index).fillna(0.0)).abs().sum()
		)
		strategy_period = (test_returns @ weights).copy()
		benchmark_period = (test_returns @ benchmark_weights).copy()

		if len(strategy_period) > 0 and cost_rate:
			strategy_period.iloc[0] -= turnover * cost_rate
			benchmark_period.iloc[0] -= benchmark_turnover * cost_rate

		frame = pd.DataFrame(
			{
				"strategy_return": strategy_period,
				"benchmark_return": benchmark_period,
			}
		)
		frame["turnover"] = 0.0
		frame["benchmark_turnover"] = 0.0
		frame["rebalance"] = 0
		frame["fallback"] = 0
		frame.iloc[0, frame.columns.get_loc("turnover")] = turnover
		frame.iloc[0, frame.columns.get_loc("benchmark_turnover")] = benchmark_turnover
		frame.iloc[0, frame.columns.get_loc("rebalance")] = 1
		frame.iloc[0, frame.columns.get_loc("fallback")] = fallback_used

		strategy_path = []
		benchmark_path = []
		for strat_ret, bench_ret in zip(frame["strategy_return"], frame["benchmark_return"]):
			strategy_nav *= 1.0 + strat_ret
			benchmark_nav *= 1.0 + bench_ret
			strategy_path.append(strategy_nav)
			benchmark_path.append(benchmark_nav)

		frame["strategy_nav"] = strategy_path
		frame["benchmark_nav"] = benchmark_path
		history_frames.append(frame)
		weights_history.append(pd.DataFrame([weights], index=[test_returns.index[0]]))
		prev_weights = weights
		prev_benchmark_weights = benchmark_weights

	if not history_frames:
		raise ValueError("Backtest empty: no usable out-of-sample windows.")
	history = pd.concat(history_frames).sort_index()
	weights_df = pd.concat(weights_history).sort_index()
	metrics = performance_summary(history, _periods_per_year(bt_cfg.frequency),
	                              rf_annual=bt_cfg.risk_free_annual)
	return WalkForwardBacktestResult(
		label=label or "strategy",
		history=history,
		weights_history=weights_df,
		metrics=metrics,
	)


def plot_backtest_nav(results: dict[str, WalkForwardBacktestResult], ax=None):
	if plt is None:
		raise ImportError("matplotlib is required for plotting")
	if ax is None:
		_, ax = plt.subplots(figsize=(12, 6))
	first_result = next(iter(results.values()))
	first_result.history["benchmark_nav"].plot(ax=ax, lw=2, color="black", ls="--", label="Benchmark equal-weight")
	for name, result in results.items():
		result.history["strategy_nav"].plot(ax=ax, lw=2, label=name)
	ax.set_title("Walk-forward out-of-sample backtest")
	ax.set_ylabel("Cumulative value (base 1)")
	ax.legend()
	return ax


__all__ = [
	"WalkForwardBacktestConfig",
	"WalkForwardBacktestResult",
	"performance_summary",
	"walk_forward_backtest",
	"plot_backtest_nav",
]
