"""Analytique DÉMO pure (sans Streamlit, sans modifier `portfolio_lab`).

Comble les quelques besoins de visualisation non couverts par les modules de recherche :
métriques glissantes (vol/Sharpe/drawdown), détection simple de régimes (bull/bear/crise),
statistiques de distribution (asymétrie, kurtosis, normalité), ACF/PACF, et corrélation
moyenne glissante. Toutes les fonctions prennent/renvoient des objets pandas/numpy : elles
sont testables hors navigateur.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
#  Métriques glissantes (Page 3)                                               #
# --------------------------------------------------------------------------- #
def drawdown_series(nav: pd.Series) -> pd.Series:
    """Drawdown courant = NAV / max courant − 1 (≤ 0)."""
    nav = nav.dropna()
    return nav / nav.cummax() - 1.0


def rolling_metrics(history: pd.DataFrame, ppy: int, window: int = 26,
                    rf_annual: float = 0.0) -> pd.DataFrame:
    """Volatilité, Sharpe et drawdown glissants à partir de `strategy_return`.

    `window` est en nombre de périodes (26 ≈ 6 mois en hebdo, 126 ≈ 6 mois en daily).
    """
    r = history["strategy_return"].dropna()
    roll_vol = r.rolling(window).std(ddof=1) * np.sqrt(ppy)
    roll_mean = r.rolling(window).mean() * ppy
    roll_sharpe = (roll_mean - rf_annual) / roll_vol.replace(0.0, np.nan)
    out = pd.DataFrame({
        "rolling_vol": roll_vol,
        "rolling_sharpe": roll_sharpe,
        "drawdown": drawdown_series(history["strategy_nav"]),
    })
    return out


# --------------------------------------------------------------------------- #
#  Régimes de marché (Page 5 / Page 7)                                         #
# --------------------------------------------------------------------------- #
def tag_regimes(nav: pd.Series, returns: pd.Series | None = None, *,
                dd_bear: float = -0.10, vol_window: int = 21,
                vol_quantile: float = 0.85, ppy: int = 252) -> pd.DataFrame:
    """Étiquette chaque date en régime : 'crise', 'bear', 'bull'.

    Règle simple et lisible (pas un modèle à états cachés) :
      - 'crise'  : volatilité glissante dans le quantile haut (`vol_quantile`) ;
      - 'bear'   : drawdown courant ≤ `dd_bear` (−10 % par défaut) ;
      - 'bull'   : le reste.
    La crise prime sur le bear, le bear sur le bull.
    """
    nav = nav.dropna()
    dd = drawdown_series(nav)
    regime = pd.Series("bull", index=nav.index, dtype=object)
    regime[dd <= dd_bear] = "bear"
    if returns is not None and len(returns.dropna()) > vol_window:
        rv = returns.reindex(nav.index).rolling(vol_window).std(ddof=1) * np.sqrt(ppy)
        thr = rv.quantile(vol_quantile)
        regime[rv >= thr] = "crise"
    return pd.DataFrame({"regime": regime, "drawdown": dd})


def regime_spans(regime: pd.Series) -> list[dict]:
    """Convertit une série de régimes en segments contigus {start, end, regime}
    (pour tracer des bandes colorées sous un axe temporel)."""
    regime = regime.dropna()
    if regime.empty:
        return []
    spans, cur, t0 = [], regime.iloc[0], regime.index[0]
    prev = regime.index[0]
    for t, v in regime.items():
        if v != cur:
            spans.append({"start": t0, "end": prev, "regime": cur})
            cur, t0 = v, t
        prev = t
    spans.append({"start": t0, "end": prev, "regime": cur})
    return spans


# --------------------------------------------------------------------------- #
#  Distribution des rendements (Page 5)                                        #
# --------------------------------------------------------------------------- #
def distribution_stats(returns: pd.Series) -> dict:
    """Moments et test de normalité (Jarque-Bera) d'une série de rendements."""
    from scipy import stats
    r = pd.Series(returns).dropna()
    if len(r) < 8:
        return {"mean": np.nan, "std": np.nan, "skew": np.nan, "kurtosis": np.nan,
                "jb_pvalue": np.nan, "is_normal": None, "n": len(r)}
    jb_stat, jb_p = stats.jarque_bera(r.values)
    return {
        "mean": float(r.mean()), "std": float(r.std(ddof=1)),
        "skew": float(r.skew()), "kurtosis": float(r.kurtosis()),  # excès de kurtosis
        "jb_pvalue": float(jb_p), "is_normal": bool(jb_p > 0.05), "n": int(len(r)),
    }


def normal_overlay(returns: pd.Series, n_points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Renvoie (x, densité normale) ajustée à la moyenne/écart-type — pour superposer
    la cloche gaussienne sur l'histogramme et juger visuellement de la non-normalité."""
    from scipy import stats
    r = pd.Series(returns).dropna().values
    if len(r) < 2:
        return np.array([]), np.array([])
    x = np.linspace(r.min(), r.max(), n_points)
    return x, stats.norm.pdf(x, loc=r.mean(), scale=r.std(ddof=1))


# --------------------------------------------------------------------------- #
#  Autocorrélations (Page 5)                                                   #
# --------------------------------------------------------------------------- #
def acf_pacf(series: pd.Series, nlags: int = 20) -> dict:
    """ACF et PACF d'une série (typiquement les rendements ou leurs carrés pour l'ARCH).

    Renvoie {lags, acf, pacf, conf}. `conf` = bande de confiance ±1.96/√T (bruit blanc).
    """
    from statsmodels.tsa.stattools import acf, pacf
    s = pd.Series(series).dropna().values
    nlags = int(min(nlags, max(1, len(s) // 2 - 1)))
    a = acf(s, nlags=nlags, fft=True)
    p = pacf(s, nlags=nlags)
    conf = 1.96 / np.sqrt(len(s))
    return {"lags": np.arange(len(a)), "acf": a, "pacf": p, "conf": float(conf)}


# --------------------------------------------------------------------------- #
#  Corrélations (Page 5)                                                       #
# --------------------------------------------------------------------------- #
def correlation_matrix(returns: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    return returns.dropna(how="any").corr(method=method)


def average_correlation(returns: pd.DataFrame) -> float:
    """Corrélation moyenne hors-diagonale (mesure de co-mouvement du marché)."""
    c = correlation_matrix(returns).values
    n = c.shape[0]
    return float((c.sum() - n) / (n * (n - 1))) if n > 1 else np.nan


def rolling_average_correlation(returns: pd.DataFrame, window: int = 63) -> pd.Series:
    """Corrélation moyenne hors-diagonale GLISSANTE : monte en crise (tout corrèle)."""
    R = returns.dropna(how="any")
    out = {}
    for i in range(window, len(R) + 1):
        win = R.iloc[i - window:i]
        c = win.corr().values
        n = c.shape[0]
        out[R.index[i - 1]] = (c.sum() - n) / (n * (n - 1)) if n > 1 else np.nan
    return pd.Series(out, name="avg_corr")


def interpret(metrics: pd.Series, returns: pd.DataFrame, port_returns: pd.Series,
              regime: pd.Series | None = None) -> list[dict]:
    """Génère des INTERPRÉTATIONS automatiques (lecture experte) à partir des résultats.

    Renvoie une liste de {level, text} (level ∈ good/warn/info) — affichée telle quelle
    dans la page Analyse pour que le public comprenne ce que disent les chiffres.
    """
    out: list[dict] = []
    sharpe = float(metrics.get("Sharpe", float("nan")))
    bench_sh = float(metrics.get("Benchmark Sharpe", float("nan")))
    cagr = float(metrics.get("CAGR (%)", float("nan")))
    vol = float(metrics.get("Vol annualized (%)", float("nan")))
    mdd = float(metrics.get("Max drawdown (%)", float("nan")))
    turn = float(metrics.get("Average turnover", float("nan")))

    # Sharpe vs benchmark
    if sharpe == sharpe and bench_sh == bench_sh:
        if sharpe > bench_sh + 0.05:
            out.append({"level": "good", "text": f"La stratégie **bat le 1/N** en Sharpe "
                        f"({sharpe:.2f} vs {bench_sh:.2f}) : l'optimisation apporte de la valeur nette de coûts."})
        elif sharpe < bench_sh - 0.05:
            out.append({"level": "warn", "text": f"Le **1/N fait mieux** ({bench_sh:.2f} vs {sharpe:.2f}) — "
                        "résultat classique (DeMiguel 2009) : l'erreur d'estimation sur μ/Σ ronge le gain théorique."})
        else:
            out.append({"level": "info", "text": f"Sharpe comparable au 1/N ({sharpe:.2f} ≈ {bench_sh:.2f}) : "
                        "le bénéfice de l'optimisation est marginal une fois les coûts payés."})

    if mdd == mdd:
        sev = "warn" if mdd < -25 else "info"
        out.append({"level": sev, "text": f"Perte maximale (drawdown) de **{mdd:.0f}%** : "
                    "à rapprocher des épisodes de stress (2020, 2022) visibles sur les régimes."})

    # Distribution
    ds = distribution_stats(port_returns)
    if ds["kurtosis"] == ds["kurtosis"]:
        if ds["kurtosis"] > 1:
            out.append({"level": "warn", "text": f"**Queues épaisses** (kurtosis {ds['kurtosis']:.1f} > 0) : "
                        "les chocs extrêmes sont plus fréquents qu'une loi normale ne le prédit → "
                        "le risque de perte rare est sous-estimé par la variance seule."})
        if ds["skew"] < -0.3:
            out.append({"level": "warn", "text": f"**Asymétrie négative** (skew {ds['skew']:.2f}) : "
                        "les grosses variations sont surtout à la baisse."})
        if ds["is_normal"]:
            out.append({"level": "info", "text": "Les rendements ne s'écartent pas significativement "
                        "de la normalité sur cet échantillon."})

    # Corrélations
    ac = average_correlation(returns)
    if ac == ac:
        out.append({"level": "info", "text": f"Corrélation moyenne entre actifs ≈ **{ac:.2f}**. "
                    "Elle tend à **monter en crise** (cf. corrélation glissante) — la diversification "
                    "s'érode quand on en a le plus besoin."})

    # Turnover
    if turn == turn:
        if turn > 0.15:
            out.append({"level": "warn", "text": f"Turnover élevé (**{turn:.0%}/rebal**) : sensible aux coûts ; "
                        "un estimateur de Σ plus stable (Ledoit-Wolf, IEWMA) réduirait la rotation."})
        else:
            out.append({"level": "good", "text": f"Turnover maîtrisé (**{turn:.0%}/rebal**) : "
                        "allocation stable, frais de transaction contenus."})

    if regime is not None and len(regime):
        share = float((regime == "crise").mean())
        out.append({"level": "info", "text": f"**{share:.0%}** des périodes sont en régime de **forte "
                    "volatilité** — c'est là que se joue l'essentiel du drawdown."})
    return out


__all__ = [
    "drawdown_series", "rolling_metrics", "tag_regimes", "regime_spans",
    "distribution_stats", "normal_overlay", "acf_pacf", "correlation_matrix",
    "average_correlation", "rolling_average_correlation", "interpret",
]
