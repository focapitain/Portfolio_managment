"""Objet de configuration central : décrit UNE expérience de bout en bout.

Changer un champ suffit à relancer une variante. C'est l'objet passé à
`run_experiment()` et le pivot de tous les benchmarks (via `dataclasses.replace`).
"""
from __future__ import annotations

from dataclasses import dataclass

from .estimators import (CovarianceEstimator, EmpiricalCovariance, MeanEstimator,
                         SampleMean)
from .optimizers import CVXPYOptimizer, Optimizer
from .universe import TopN, UniverseSelector


@dataclass
class ExperimentConfig:
    """Tous les leviers du framework en un seul endroit."""
    # --- Univers ---
    universe: UniverseSelector | None = None       # TopN(50), RandomN(30), StratifiedBySector(60)...
    start_date: str = "2022-01-01"
    end_date: str = "2025-12-31"

    # --- Données ---
    frequency: str = "daily"                        # daily | weekly | monthly | yearly
    return_mode: str = "simple"                     # simple | log

    # --- Estimation ---
    cov_estimator: CovarianceEstimator | None = None
    mean_estimator: MeanEstimator | None = None
    risk_free_annual: float = 0.04                  # taux sans risque ANNUEL

    # --- Optimisation ---
    objective: str = "max_sharpe"                   # max_sharpe | min_variance | max_return | max_utility
    optimizer: Optimizer | None = None
    long_only: bool = True
    w_max: float = 1.0                              # plafond par actif (1.0 = pas de plafond)
    risk_aversion: float = 3.0                      # pour l'objectif "max_utility"

    def resolved(self) -> "ExperimentConfig":
        """Remplit les valeurs par défaut."""
        if self.universe is None:
            self.universe = TopN(50)
        if self.cov_estimator is None:
            self.cov_estimator = EmpiricalCovariance()
        if self.mean_estimator is None:
            self.mean_estimator = SampleMean()
        if self.optimizer is None:
            self.optimizer = CVXPYOptimizer()
        return self


__all__ = ["ExperimentConfig"]
