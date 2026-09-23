"""Détection des contraintes ACTIVÉES sur une allocation (Page 7 « Rejouer l'histoire »).

But pédagogique : montrer *pourquoi* le portefeuille a pris telle forme — quelles bornes
ont « mordu ». Pur pandas/numpy, aucune dépendance à Streamlit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def which_constraints_active(weights: pd.Series, *, w_max: float, budget: float = 1.0,
                             leverage: float | None = None, tol: float = 1e-3) -> dict:
    """Diagnostique les contraintes saturées par un vecteur de poids.

    Renvoie un dict lisible : nombre d'actifs au plafond, actifs à zéro (borne basse),
    respect du budget, levier réalisé, concentration (Herfindahl, N effectifs).
    """
    w = pd.Series(weights).fillna(0.0)
    at_cap = w[w >= w_max - tol]
    at_floor = w[w <= tol]
    gross = float(w.abs().sum())
    herf = float((w ** 2).sum())
    return {
        "n_at_cap": int(at_cap.shape[0]),
        "assets_at_cap": list(at_cap.index),
        "n_at_floor": int(at_floor.shape[0]),
        "budget_ok": bool(abs(float(w.sum()) - budget) < 1e-6),
        "gross_leverage": gross,
        "leverage_binding": bool(leverage is not None and gross >= leverage - tol),
        "herfindahl": herf,
        "effective_n": float(1.0 / herf) if herf > 0 else np.nan,
        "max_weight": float(w.max()),
    }


def turnover_between(w_prev: pd.Series, w_now: pd.Series) -> float:
    """Turnover L1 = Σ|w_now − w_prev| (aligné sur l'union des actifs)."""
    idx = w_prev.index.union(w_now.index)
    a = pd.Series(w_prev).reindex(idx).fillna(0.0)
    b = pd.Series(w_now).reindex(idx).fillna(0.0)
    return float((b - a).abs().sum())


def degeneracy_check(n_effective: int, w_max: float) -> dict:
    """Garde-fou hérité de l'audit : N×w_max ≈ 1 ⇒ l'espace admissible se réduit à 1/N.

    Renvoie {budget, ok, message} pour un bandeau d'avertissement dans l'UI.
    """
    budget = n_effective * w_max
    ok = budget > 1.2 or w_max >= 1.0
    if w_max >= 1.0:
        msg = "Pas de plafond (w_max=100%) : optimisation libre."
    elif ok:
        msg = f"N×w_max = {n_effective}×{w_max:.0%} = {budget:.1f} — l'optimiseur a du jeu."
    else:
        msg = (f"⚠️ N×w_max = {n_effective}×{w_max:.0%} = {budget:.2f} ≤ 1.2 : l'espace "
               f"admissible se réduit à l'équipondéré (tous les objectifs convergent vers 1/N). "
               f"Augmentez N ou relevez w_max.")
    return {"budget": budget, "ok": ok, "message": msg}


__all__ = ["which_constraints_active", "turnover_between", "degeneracy_check"]
