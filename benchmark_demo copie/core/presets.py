"""Scénarios historiques, univers curatés et annotations d'événements .

Pour « rejouer l'histoire » sans tomber sur le biais de survivance ni les trous de cotation,
on propose un univers de **grandes capitalisations à long historique** (cotées avant 2007).
Aucune dépendance Streamlit.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
#  Scénarios historiques : (début, fin) + événements à annoter                 #
# --------------------------------------------------------------------------- #
SCENARIOS = {
    "Crise financière 2008": {
        "start": "2007-06-01", "end": "2009-12-31",
        "events": [("2008-09-15", "Faillite Lehman"),
                   ("2009-03-09", "Creux du marché")],
        "note": "Choc systémique : corrélations → 1, volatilité extrême.",
    },
    "COVID-19 2020": {
        "start": "2019-06-01", "end": "2021-06-30",
        "events": [("2020-02-19", "Sommet pré-COVID"),
                   ("2020-03-23", "Creux COVID"),
                   ("2020-11-09", "Annonce vaccin")],
        "note": "Krach éclair puis reprise en V — test de réactivité.",
    },
    "Marché haussier 2017": {
        "start": "2016-06-01", "end": "2018-01-31",
        "events": [("2017-01-01", "Rallye régulier")],
        "note": "Faible volatilité, tendance porteuse — régime 'facile'.",
    },
    "Resserrement 2022": {
        "start": "2021-06-01", "end": "2023-06-30",
        "events": [("2022-01-03", "Pic avant baisse"),
                   ("2022-10-12", "Creux 2022")],
        "note": "Hausse des taux, bear market lent — rotation de styles.",
    },
}

# Grandes capitalisations cotées de longue date (pré-2007) → réduit le biais de listing
# pour les rejeux historiques. (Liste indicative, pas un univers de recherche.)
CURATED_LONG_HISTORY = (
    "AAPL", "MSFT", "JNJ", "PG", "KO", "PEP", "WMT", "XOM", "CVX", "JPM",
    "BAC", "WFC", "HD", "MCD", "DIS", "IBM", "INTC", "CSCO", "ORCL", "T",
    "VZ", "PFE", "MRK", "ABT", "MMM", "CAT", "BA", "GE", "HON", "UNH",
    "LOW", "TGT", "COST", "NKE", "SBUX", "GS", "MS", "AXP", "USB", "C",
)


def scenario_names() -> list[str]:
    return list(SCENARIOS)


def get_scenario(name: str) -> dict:
    return SCENARIOS[name]


__all__ = ["SCENARIOS", "CURATED_LONG_HISTORY", "scenario_names", "get_scenario"]
