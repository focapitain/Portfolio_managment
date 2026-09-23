"""Calcul des rendements période-sur-période."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_returns(prices: pd.DataFrame, mode: str = "simple") -> pd.DataFrame:
    """Rendements période-sur-période.

    mode="simple" : (P_t / P_{t-1}) - 1      (composables par produit)
    mode="log"    : ln(P_t / P_{t-1})        (additifs dans le temps)
    """
    if mode == "simple":
        return prices.pct_change().dropna(how="any")
    if mode == "log":
        return np.log(prices / prices.shift(1)).dropna(how="any")
    raise ValueError(f"mode inconnu : {mode!r} (attendu 'simple' ou 'log')")


__all__ = ["compute_returns"]
