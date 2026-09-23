"""Sélection de l'univers d'investissement (patron *Strategy*).

Chaque sélecteur prend le DataFrame des composants (indexé par ticker, colonne
'sector') et renvoie une liste de tickers. Étendre = ajouter une classe.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

RANDOM_SEED = 42


class UniverseSelector(ABC):
    """Sélectionne une liste de tickers parmi les composants disponibles."""
    name = "base"

    @abstractmethod
    def select(self, constituents: pd.DataFrame) -> list[str]:
        ...


class AllUniverse(UniverseSelector):
    """Tout l'univers. Attention : N≈500 (coûteux pour les optimiseurs non convexes)."""
    name = "all"

    def select(self, constituents):
        return constituents.index.tolist()


class TopN(UniverseSelector):
    """Les N PREMIERS de la table source, dans son ordre natif.

    ⚠️ ATTENTION : la table Wikipédia des composants du S&P 500 est triée par ORDRE
    ALPHABÉTIQUE du ticker, PAS par capitalisation. `TopN(25)` renvoie donc A, AAPL,
    ABBV, ABNB… (les 25 premiers tickers de A à Z), et NON les 25 plus grosses
    capitalisations. Ne pas interpréter ce sélecteur comme « grandes capitalisations ».
    Un vrai tri par capitalisation (`TopNByMarketCap`) nécessiterait les market caps
    (données externes, hors périmètre actuel). Voir l'alias `FirstNAlphabetical`.
    """
    name = "top_n"

    def __init__(self, n: int = 50):
        self.n = n

    def select(self, constituents):
        return constituents.index[: self.n].tolist()


# Alias sémantiquement honnête : c'est bien un échantillon par ordre alphabétique du
# ticker, à utiliser dans les notebooks pour ne pas suggérer un tri par capitalisation.
FirstNAlphabetical = TopN


class RandomN(UniverseSelector):
    """N actifs tirés aléatoirement (reproductible via seed)."""
    name = "random_n"

    def __init__(self, n: int = 30, seed: int = RANDOM_SEED):
        self.n, self.seed = n, seed

    def select(self, constituents):
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(constituents), size=min(self.n, len(constituents)), replace=False)
        return constituents.index[idx].tolist()


class StratifiedBySector(UniverseSelector):
    """Échantillonnage stratifié : ~même nombre d'actifs par secteur GICS.

    Améliore la diversification structurelle de l'univers (≠ concentration tech).
    """
    name = "stratified"

    def __init__(self, n_total: int = 60, seed: int = RANDOM_SEED):
        self.n_total, self.seed = n_total, seed

    def select(self, constituents):
        rng = np.random.default_rng(self.seed)
        sectors = constituents["sector"].dropna().unique()
        per = max(1, self.n_total // len(sectors))
        picks: list[str] = []
        for s in sectors:
            members = constituents.index[constituents["sector"] == s].tolist()
            k = min(per, len(members))
            picks += list(rng.choice(members, size=k, replace=False))
        return picks[: self.n_total]


__all__ = ["UniverseSelector", "AllUniverse", "TopN", "FirstNAlphabetical",
           "RandomN", "StratifiedBySector", "RANDOM_SEED"]
