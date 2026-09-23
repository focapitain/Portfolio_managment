"""Accès aux données : composants du S&P 500, prix ajustés, nettoyage.

Inclut aussi un générateur de prix SYNTHÉTIQUES (`make_synthetic_prices`) pour
exécuter le code hors-ligne (CI, démonstration sans réseau). Les données
synthétiques ne servent JAMAIS de résultat de recherche : elles ne valident que
la plomberie du pipeline.
"""
from __future__ import annotations

from io import StringIO

import numpy as np
import pandas as pd

# Fallback minimal si Wikipédia est inaccessible (réseau coupé, etc.)
_FALLBACK = {
    "AAPL": "Information Technology", "MSFT": "Information Technology", "NVDA": "Information Technology",
    "AMZN": "Consumer Discretionary", "GOOGL": "Communication Services", "META": "Communication Services",
    "BRK-B": "Financials", "JPM": "Financials", "V": "Financials", "JNJ": "Health Care",
    "UNH": "Health Care", "XOM": "Energy", "PG": "Consumer Staples", "HD": "Consumer Discretionary",
    "MA": "Financials", "LLY": "Health Care", "AVGO": "Information Technology", "CVX": "Energy",
}


def get_sp500_constituents() -> pd.DataFrame:
    """DataFrame indexé par ticker avec une colonne 'sector' (GICS)."""
    import requests

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        df = pd.read_html(StringIO(response.text))[0]
        df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)  # BRK.B -> BRK-B (Yahoo)
        out = df.set_index("Symbol")[["GICS Sector"]].rename(columns={"GICS Sector": "sector"})
        print(f"[OK] {len(out)} composants récupérés depuis Wikipédia.")
        return out
    except Exception as e:  # pragma: no cover - dépend du réseau
        print(f"[FALLBACK] Wikipédia inaccessible ({e}). Liste codée en dur.")
        return pd.DataFrame.from_dict(_FALLBACK, orient="index", columns=["sector"])


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Prix de clôture AJUSTÉS (dividendes/splits) via yfinance."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("yfinance requis : pip install yfinance") from exc
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, threads=True)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    return prices.sort_index()


def clean_prices(prices: pd.DataFrame, max_missing: float = 0.05) -> pd.DataFrame:
    """Élimine les tickers trop incomplets, forward-fill les trous résiduels."""
    keep = prices.columns[prices.isna().mean() <= max_missing]
    prices = prices[keep].ffill().dropna(how="any")
    return prices


def make_synthetic_prices(constituents: pd.DataFrame, start: str = "2018-01-01",
                          end: str = "2025-12-31", seed: int = 42) -> pd.DataFrame:
    """Panel de prix SYNTHÉTIQUES (mouvement brownien géométrique corrélé par secteur).

    Hors-ligne uniquement : permet de faire tourner notebooks et tests sans réseau.
    La structure de corrélation intra-secteur est volontairement réaliste (≈0.5)
    pour que les estimateurs de covariance produisent des sorties non triviales.
    """
    rng = np.random.default_rng(seed)
    tickers = list(constituents.index)
    sectors = constituents["sector"].fillna("Unknown")
    n = len(tickers)
    dates = pd.bdate_range(start=start, end=end)
    T = len(dates)

    # facteur marché commun + facteur sectoriel + bruit idiosyncratique
    market = rng.normal(0.0004, 0.010, T)
    uniq_sectors = sectors.unique()
    sector_factor = {s: rng.normal(0.0001, 0.006, T) for s in uniq_sectors}

    rets = np.zeros((T, n))
    for j, tk in enumerate(tickers):
        beta = rng.uniform(0.7, 1.4)
        load_sec = rng.uniform(0.3, 0.8)
        idio = rng.normal(0.0, rng.uniform(0.008, 0.018), T)
        drift = rng.uniform(-0.0001, 0.0005)
        rets[:, j] = drift + beta * market + load_sec * sector_factor[sectors.iloc[j]] + idio

    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


__all__ = ["get_sp500_constituents", "download_prices", "clean_prices",
           "make_synthetic_prices"]
