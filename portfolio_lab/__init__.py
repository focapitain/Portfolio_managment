"""portfolio_lab — plateforme modulaire de recherche en optimisation de portefeuille.

Construit sur la théorie moderne du portefeuille (Markowitz) et conçu pour le
benchmarking reproductible hors échantillon sur l'univers S&P 500.

Exemple minimal
---------------
>>> from portfolio_lab import ExperimentConfig, run_experiment, StratifiedBySector, EWMACovariance
>>> cfg = ExperimentConfig(universe=StratifiedBySector(60), cov_estimator=EWMACovariance())
>>> res = run_experiment(cfg)        # télécharge les données et optimise
>>> res.summary()

Backtest hors échantillon
-------------------------
>>> from portfolio_lab import run_walk_forward
>>> from portfolio_lab.backtest import WalkForwardBacktestConfig
>>> bt_cfg = WalkForwardBacktestConfig(estimation_window=252, holding_period=21)
>>> result = run_walk_forward(prices, cfg, backtest_config=bt_cfg)
>>> result.metrics
"""
from __future__ import annotations

from .config import ExperimentConfig
from .data import (clean_prices, download_prices, get_sp500_constituents,
                   make_synthetic_prices)
from .estimators import (BlackLittermanMean, ConstantCorrelationCovariance,
                         CovarianceEstimator, EmpiricalCovariance, EWMACovariance,
                         EWMAMean, IEWMACovariance, JamesSteinMean,
                         LedoitWolfCovariance, MeanEstimator, OASCovariance,
                         PCAFactorMean, RollingCovariance, RollingMean, SampleMean)
from .experiment import ExperimentResult, run_experiment
from .frequencies import annualize, periods_per_year, resample_prices
from .frontier import efficient_frontier
from .metrics import (calmar_ratio, conditional_var, sortino_ratio, tail_metrics,
                      value_at_risk)
from .cov_eval import (condition_number, cov_eval_summary, frobenius_error,
                       gaussian_loglik, mae_cov, realized_covariance, regime_split,
                       rmse_cov, rolling_cov_eval, temporal_stability,
                       vol_forecast_error)
from .mean_eval import (hit_rate, mae_mean, mean_eval_summary, pearson_ic,
                        rmse_mean, rolling_mean_eval, spearman_ic,
                        window_sensitivity, wmape)
from .markowitz import (ConstraintSet, MarkowitzBacktestResult, MarkowitzPlusPlus,
                        PenaltySet, calibrate_markowitzpp, default_score, make_variant,
                        markowitz_metrics_table, markowitz_walk_forward,
                        run_markowitz_benchmark, weight_sensitivity)
from .optimizers import (AnalyticLagrangian, CVXPYOptimizer,
                         DifferentialEvolutionOptimizer, EqualWeightOptimizer,
                         InverseVolatilityOptimizer, MaxDiversificationOptimizer,
                         Optimizer, RandomWeightOptimizer, RiskParityOptimizer,
                         ScipyOptimizer, portfolio_stats)
from .returns import compute_returns
from .universe import (AllUniverse, FirstNAlphabetical, RandomN,
                       StratifiedBySector, TopN, UniverseSelector)
from .backtest_adapter import (build_compute_returns_fn, build_optimize_weights_fn,
                               run_walk_forward)
from .benchmarking import (allocate, metrics_table, run_oos_benchmark,
                           tail_metrics_table, train_test_split_evaluation,
                           weight_stability)
from .reporting import RunReport, describe

__version__ = "2.0.0"

__all__ = [
    # config & pipeline
    "ExperimentConfig", "ExperimentResult", "run_experiment",
    # données
    "get_sp500_constituents", "download_prices", "clean_prices", "make_synthetic_prices",
    # univers
    "UniverseSelector", "AllUniverse", "TopN", "FirstNAlphabetical", "RandomN",
    "StratifiedBySector",
    # rendements & fréquences
    "compute_returns", "periods_per_year", "resample_prices", "annualize",
    # estimateurs
    "MeanEstimator", "SampleMean", "JamesSteinMean", "EWMAMean", "RollingMean",
    "PCAFactorMean",
    "CovarianceEstimator", "EmpiricalCovariance", "RollingCovariance",
    "EWMACovariance", "LedoitWolfCovariance", "OASCovariance",
    "ConstantCorrelationCovariance", "IEWMACovariance",
    # optimiseurs
    "Optimizer", "AnalyticLagrangian", "CVXPYOptimizer", "ScipyOptimizer",
    "DifferentialEvolutionOptimizer", "EqualWeightOptimizer", "RandomWeightOptimizer",
    "InverseVolatilityOptimizer", "RiskParityOptimizer",
    "MaxDiversificationOptimizer", "portfolio_stats",
    # frontière & visualisation
    "efficient_frontier",
    # métriques
    "sortino_ratio", "value_at_risk", "conditional_var", "calmar_ratio", "tail_metrics",
    # évaluation statistique de la covariance
    "gaussian_loglik", "realized_covariance", "frobenius_error", "rmse_cov", "mae_cov",
    "vol_forecast_error", "condition_number", "rolling_cov_eval", "cov_eval_summary",
    "temporal_stability", "regime_split",
    # évaluation prédictive du rendement μ
    "rmse_mean", "mae_mean", "wmape", "pearson_ic", "spearman_ic", "hit_rate",
    "rolling_mean_eval", "mean_eval_summary", "window_sensitivity",
    # benchmark des formulations de Markowitz
    "ConstraintSet", "PenaltySet", "MarkowitzPlusPlus", "make_variant",
    "MarkowitzBacktestResult", "markowitz_walk_forward", "run_markowitz_benchmark",
    "markowitz_metrics_table", "default_score", "calibrate_markowitzpp",
    "weight_sensitivity",
    # backtest
    "build_compute_returns_fn", "build_optimize_weights_fn", "run_walk_forward",
    # benchmarking
    "run_oos_benchmark", "metrics_table", "tail_metrics_table", "weight_stability",
    "allocate", "train_test_split_evaluation",
    # reporting / journalisation des runs
    "RunReport", "describe",
]
