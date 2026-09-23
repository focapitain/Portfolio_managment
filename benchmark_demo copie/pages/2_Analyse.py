"""Page Analyse — performance, risque et marché réunis, avec lectures automatiques.

Fusion des anciennes pages Performance + Marché : NAV, drawdown, vol/Sharpe glissants,
distribution (skew/kurtosis), corrélations, régimes de marché sur les graphiques, et des
**interprétations automatiques** des résultats.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import datetime as _dt

import streamlit as st

from components.sidebar import render_sidebar, require_committed, get_bundle, page_header
from components import charts as C
from components import ui
from core import analytics, services
from core.registry import pl

params, committed = render_sidebar()
page_header("Analyse", "Performance, risque & marché",
            "Ce que disent les chiffres — et leur interprétation automatique.")
params = require_committed(committed)
bundle = get_bundle(params)

ppy = pl.periods_per_year(params["frequency"])
port_ret = bundle.history["strategy_return"].dropna()
regime_df = analytics.tag_regimes(bundle.nav, bundle.returns.mean(axis=1), ppy=ppy)

# --- Lecture automatique (en tête) ---------------------------------------------------------
#st.markdown("### 🔎 Lecture automatique")
#insights = analytics.interpret(bundle.metrics, bundle.returns, port_ret, regime_df["regime"])
#ui.render_insights(st, insights)


st.divider()
tab1, tab2, tab3 = st.tabs(["   Performance", " Distribution & corrélations", " Régimes"])

with tab1:
    m = bundle.metrics
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("CAGR", f"{m['CAGR (%)']:.1f}%")
    k2.metric("Sharpe", f"{m['Sharpe']:.2f}")
    k3.metric("Volatilité", f"{m['Vol annualized (%)']:.1f}%")
    k4.metric("Max drawdown", f"{m['Max drawdown (%)']:.1f}%")
    k5.metric("Turnover", f"{m['Average turnover']:.1%}")
    st.plotly_chart(C.nav_chart(bundle.history, log=True, index_nav=bundle.index_nav,
                                random_nav=bundle.random_nav),
                    use_container_width=True)
    win = st.slider("Fenêtre glissante (périodes)", 8, 156,
                    26 if params["frequency"] == "weekly" else 126, key="an_win")
    rolling = analytics.rolling_metrics(bundle.history, ppy, window=win, rf_annual=params["rf"])
    st.plotly_chart(C.drawdown_chart(rolling["drawdown"]), use_container_width=True)
    st.plotly_chart(C.rolling_chart(rolling), use_container_width=True)

with tab2:
    asset = st.selectbox("Série à analyser", ["Portefeuille"] + list(bundle.returns.columns),
                         key="an_asset")
    series = port_ret if asset == "Portefeuille" else bundle.returns[asset]
    x, y = analytics.normal_overlay(series)
    c1, c2 = st.columns([0.55, 0.45])
    with c1:
        st.plotly_chart(C.distribution_hist(series, x, y), use_container_width=True)
    with c2:
        ds = analytics.distribution_stats(series)
        st.metric("Asymétrie (skew)", f"{ds['skew']:.2f}")
        st.metric("Kurtosis (excès)", f"{ds['kurtosis']:.2f}")
        st.metric("Normal (Jarque-Bera)?", "Oui" if ds["is_normal"] else "Non",
                  help=f"p-value = {ds['jb_pvalue']:.3f}")
    st.plotly_chart(C.correlation_heatmap(analytics.correlation_matrix(bundle.returns)),
                    use_container_width=True)
    cwin = st.slider("Fenêtre de corrélation glissante", 12, 156,
                     52 if params["frequency"] == "weekly" else 63, key="an_corrwin")
    st.plotly_chart(C.rolling_corr_chart(
        analytics.rolling_average_correlation(bundle.returns, window=cwin)), use_container_width=True)

with tab3:
    spans = analytics.regime_spans(regime_df["regime"])
    fig = C.nav_chart(bundle.history, log=True, index_nav=bundle.index_nav,
                      random_nav=bundle.random_nav)
    C.add_regime_bands(fig, spans)
    st.plotly_chart(fig, use_container_width=True)
    vc = regime_df["regime"].value_counts()
    cols = st.columns(len(vc))
    for col, (name, cnt) in zip(cols, vc.items()):
        col.metric(name.capitalize(), f"{cnt} périodes", f"{cnt / len(regime_df):.0%}")
    st.caption("Bandes : 🟩 bull · 🟥 bear (drawdown < −10 %) · 🟧 crise (forte volatilité). "
               "Les drawdowns se concentrent dans les zones rouges/oranges.")

# --- Export du rapport (Markdown horodaté) -------------------------------------------------
st.divider()
st.markdown("### Exporter le rapport")
st.caption("Compte rendu reproductible : paramètres, métriques OOS et contrôles de santé "
           "(dégénérescence N×w_max, fallbacks).")
with st.spinner("Génération du rapport…"):
    report_md = services.report_markdown(**params)
_stamp = _dt.datetime.now().strftime("%Y-%m-%d_%H%M")
with st.expander("Aperçu du rapport"):
    st.markdown(report_md)
    
st.download_button("Télécharger le rapport (Markdown)", report_md,
                   file_name=f"rapport_simulation__{_stamp}.md", mime="text/markdown",
                   type="primary")
