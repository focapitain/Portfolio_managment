"""Registres nom → objet `portfolio_lab` (factories).

Toute la démo manipule des **chaînes** (sélectionnées dans l'UI) plutôt que des objets :
c'est indispensable pour le cache Streamlit (les arguments doivent être hachables) et ça
garde l'UI découplée du package de recherche. Ce module centralise la traduction
chaîne → instance d'estimateur / d'optimiseur / de sélecteur d'univers.

Aucune modification de `portfolio_lab` : on ne fait qu'instancier ses classes publiques.
"""
from __future__ import annotations

import os
import sys

# Rendre le package portfolio_lab importable quel que soit le cwd de lancement de Streamlit.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import portfolio_lab as pl  # noqa: E402


# --------------------------------------------------------------------------- #
#  Estimateurs de rendement μ                                                  #
# --------------------------------------------------------------------------- #
MU_ESTIMATORS = {
    "SampleMean":  lambda: pl.SampleMean(),
    "James-Stein": lambda: pl.JamesSteinMean(),
    "EWMA":        lambda: pl.EWMAMean(lam=0.94),
    #"RollingMean": lambda: pl.RollingMean(window=63),
    #"PCA-Factor":  lambda: pl.PCAFactorMean(n_factors=3),
}

# --------------------------------------------------------------------------- #
#  Estimateurs de covariance Σ                                                 #
# --------------------------------------------------------------------------- #
COV_ESTIMATORS = {
    "Empirical":     lambda: pl.EmpiricalCovariance(),
    "Rolling":       lambda: pl.RollingCovariance(window=120),
    "EWMA":          lambda: pl.EWMACovariance(lam=0.94),
    #"IEWMA":         lambda: pl.IEWMACovariance(vol_halflife=125, corr_halflife=21),
    "Ledoit-Wolf":   lambda: pl.LedoitWolfCovariance(),
    #"OAS":           lambda: pl.OASCovariance(),
    #"Constant-Corr": lambda: pl.ConstantCorrelationCovariance(),
}

# --------------------------------------------------------------------------- #
#  Méthodes de portefeuille (objectif + optimiseur)                            #
# --------------------------------------------------------------------------- #
# Chaque méthode = (fabrique d'optimiseur, objectif passé au solveur). Les baselines de
# risque ignorent l'objectif ; on met "min_variance" par convention neutre.
PORTFOLIO_METHODS = {
    #"Equal Weight":        {"optimizer": lambda: pl.EqualWeightOptimizer(),        "objective": "min_variance"},
    "Min Variance":        {"optimizer": lambda: pl.CVXPYOptimizer(),              "objective": "min_variance"},
    "Mean Variance":       {"optimizer": lambda: pl.CVXPYOptimizer(),              "objective": "max_sharpe"},
    "Max Utility":         {"optimizer": lambda: pl.CVXPYOptimizer(),              "objective": "max_utility"},
    #"Risk Parity":         {"optimizer": lambda: pl.RiskParityOptimizer(),         "objective": "min_variance"},
    #"Max Diversification": {"optimizer": lambda: pl.MaxDiversificationOptimizer(), "objective": "min_variance"},
}

# --------------------------------------------------------------------------- #
#  Sélecteurs d'univers                                                        #
# --------------------------------------------------------------------------- #
UNIVERSE_KINDS = ["StratifiedBySector",  "RandomN", "All"] #"TopN"


def make_universe(kind: str, n: int, seed: int = 42):
    """Fabrique un sélecteur d'univers `portfolio_lab` à partir d'un nom."""
    if kind == "All":
        return pl.AllUniverse()
    #if kind == "TopN":
    #    return pl.TopN(n)
    if kind == "RandomN":
        return pl.RandomN(n, seed=seed)
    return pl.StratifiedBySector(n_total=n, seed=seed)


def describe_method(name: str) -> str:
    """Phrase courte expliquant ce que fait une méthode de portefeuille (pour l'UI)."""
    return {
        "Equal Weight":        "1/N — référence robuste, ignore μ et Σ.",
        "Min Variance":        "Minimise wᵀΣw — ne dépend que de Σ.",
        "Mean Variance":       "Portefeuille tangent (max Sharpe) — dépend de μ ET Σ.",
        "Max Utility":         "max μᵀw − ½γ·wᵀΣw — compromis rendement/risque.",
        #"Risk Parity":         "Contribution au risque égalisée entre actifs.",
        #"Max Diversification": "Maximise le ratio de diversification (Choueifaty).",
    }.get(name, "")


# --------------------------------------------------------------------------- #
#  Libellés HUMAINS (UI) — on N'AGIT PAS sur les clés internes                 #
# --------------------------------------------------------------------------- #
# Les clés ci-dessus restent les identifiants techniques (utilisés par le cache Streamlit et
# par portfolio_lab). Ce dict ne sert qu'à l'AFFICHAGE : on montre un libellé clair à
# l'utilisateur, mais on continue de manipuler la clé d'origine en interne.
DISPLAY_NAMES = {
    # Univers
    "StratifiedBySector": "Diversifié par secteur",
    "RandomN":            "Tirage aléatoire",
    "All":                "Tout le S&P 500",
    # Méthodes de portefeuille
    "Equal Weight":       "Équipondéré (1/N)",
    "Min Variance":       "Risque minimum",
    "Mean Variance":      "Markowitz (rendement/risque)",
    "Max Utility":        "Compromis rendement/risque",
    # Estimateurs μ
    "SampleMean":         "Moyenne historique",
    "James-Stein":        "James-Stein (μ régularisé)",
    "EWMA":               "EWMA (μ pondéré récent)",
    # Estimateurs Σ
    "Empirical":          "Empirique",
    "Rolling":            "Fenêtre glissante",
    "Ledoit-Wolf":        "Ledoit-Wolf (shrinkage)",
    # Fréquences
    "daily":              "Quotidien",
    "weekly":             "Hebdomadaire",
    "monthly":            "Mensuel",
}


def label(name: str) -> str:
    """Libellé d'affichage pour un identifiant interne (fallback = identifiant brut)."""
    return DISPLAY_NAMES.get(name, name)


def labelize(names) -> dict:
    """Construit un mapping {libellé affiché → clé interne} pour un selectbox.

    Permet d'afficher des libellés clairs tout en récupérant la clé technique :
        opts = labelize(reg.UNIVERSE_KINDS)
        shown = st.selectbox("Univers", list(opts))
        internal = opts[shown]
    """
    out: dict[str, str] = {}
    for k in names:
        disp = label(k)
        if disp in out:                      # collision improbable : on garde l'unicité
            disp = f"{disp} ({k})"
        out[disp] = k
    return out


__all__ = [
    "MU_ESTIMATORS", "COV_ESTIMATORS", "PORTFOLIO_METHODS", "UNIVERSE_KINDS",
    "make_universe", "describe_method", "DISPLAY_NAMES", "label", "labelize",
]
