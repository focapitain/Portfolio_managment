"""Fréquences temporelles, ré-échantillonnage et annualisation.

Toutes les conventions de passage « par période → annuel » sont centralisées ici
pour qu'un seul endroit fasse foi (cf. notebook V2, section 2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = {"daily": 252, "weekly": 52, "monthly": 12, "yearly": 1}
RESAMPLE_RULE = {"daily": None, "weekly": "W-FRI", "monthly": "ME", "yearly": "YE"}


def periods_per_year(frequency: str) -> int:
    if frequency not in PERIODS_PER_YEAR:
        raise ValueError(f"Fréquence inconnue : {frequency!r}")
    return PERIODS_PER_YEAR[frequency]


def resample_prices(prices: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Ré-échantillonne les prix en prenant le DERNIER prix de chaque période."""
    if frequency not in RESAMPLE_RULE:
        raise ValueError(f"Fréquence inconnue : {frequency!r}")
    rule = RESAMPLE_RULE[frequency]
    return prices if rule is None else prices.resample(rule).last()


def annualize(mu_period: np.ndarray, Sigma_period: np.ndarray, frequency: str):
    """Passe des rendements/covariance par période à l'annuel (hypothèse i.i.d.)."""
    k = periods_per_year(frequency)
    return mu_period * k, Sigma_period * k


__all__ = ["PERIODS_PER_YEAR", "RESAMPLE_RULE", "periods_per_year",
           "resample_prices", "annualize"]
