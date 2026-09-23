"""Page Simulation — le cœur de la démo : comment le portefeuille DÉCIDE.

NAV animée (lecture auto) + rebalancements sur la courbe, slider temporel, panneau
« Décision du portefeuille » (vol estimée, corrélation, turnover, rebalancement) et frise du
processus Observation → Estimation μ → Estimation Σ → Optimisation → Rebalancement → Mise à jour.
Mode REPLAY : période complète, COVID, bear, bull.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import pandas as pd
import streamlit as st

from components.sidebar import render_sidebar, require_committed, get_bundle, page_header
from components import charts as C
from components import ui
from core import services, presets
from core.constraints import which_constraints_active

params, committed = render_sidebar()
page_header("Simulation", "Le gérant en direct — observer, optimiser, rebalancer",
            "Lancez la lecture automatique, ou déplacez le curseur pour disséquer une décision.")
params = require_committed(committed)

# --- Mode REPLAY : période complète ou épisode historique ---------------------------------
modes = ["Période complète"] + presets.scenario_names()
choice = st.radio("Mode de simulation", modes, horizontal=True, key="sim_mode")
_PPY = {"daily": 252, "weekly": 52, "monthly": 12}
if choice == "Période complète":
    args, tickers = dict(params), None
    st.caption("Fenêtre = période choisie dans la configuration.")
else:
    sc = presets.get_scenario(choice)
    args = {**params, "start": sc["start"], "end": sc["end"]}
    tickers = presets.CURATED_LONG_HISTORY          # univers stable (réduit le biais de listing)
    st.caption(f"**{choice}** — {sc['note']}  ·  {sc['start']} → {sc['end']}  ·  univers large-caps.")
    # Clamp VISIBLE de la fenêtre : un épisode dure ~1-3 ans ; une fenêtre de 252 semaines le
    # couvrirait en entier → zéro rebalancement. On la réduit à ~T/3 et on l'ANNONCE (jamais
    # de modification silencieuse du protocole — cf. audit §8).
    span_years = max((pd.Timestamp(sc["end"]) - pd.Timestamp(sc["start"])).days, 1) / 365.25
    T_scen = int(span_years * _PPY.get(params["frequency"], 52))
    win_eff = min(int(params["est_window"]), max(8, T_scen // 3))
    if win_eff < int(params["est_window"]):
        args["est_window"] = win_eff
        st.info(f"Fenêtre d'estimation réduite à **{win_eff}** périodes (au lieu de "
                f"{params['est_window']}) pour laisser ~{T_scen - win_eff} périodes de "
                f"simulation sur cet épisode court.", icon="ℹ️")

with st.spinner("Simulation…"):
    bundle = services.run_backtest(**args, tickers=tickers)
    dtable = services.decision_table(**args, tickers=tickers)

if bundle.weights.empty:
    st.warning("Pas de rebalancements sur cette fenêtre (élargissez la période).")
    st.stop()

m = bundle.metrics
k1, k2, k3, k4 = st.columns(4)
k1.metric("Sharpe", f"{m['Sharpe']:.2f}")
k2.metric("CAGR", f"{m['CAGR (%)']:.1f}%")
k3.metric("Max drawdown", f"{m['Max drawdown (%)']:.1f}%")
k4.metric("Turnover moy.", f"{m['Average turnover']:.1%}")

# --- Hero animé : NAV qui se remplit + donut + marqueurs de rebalancement ------------------
st.plotly_chart(C.manager_animation(bundle.history, bundle.weights, top=12,
                                    index_nav=bundle.index_nav, random_nav=bundle.random_nav),
                use_container_width=True)

st.divider()
st.markdown("###  Décision du portefeuille")
st.caption("Choisissez un rebalancement : la frise montre comment la décision a été prise, "
           "et le panneau ses intrants (risque estimé, corrélations, rotation).")

dates = list(bundle.weights.index)
labels = [str(pd.Timestamp(d).date()) for d in dates]
sel = st.select_slider("Date de décision", options=labels, value=labels[len(labels) // 2],
                       key="sim_date")
i = labels.index(sel)
d = dates[i]
w_now = bundle.weights.loc[d]
w_prev = bundle.weights.iloc[i - 1] if i > 0 else pd.Series(0.0, index=w_now.index)
delta = w_now - w_prev.reindex(w_now.index).fillna(0.0)
ca = which_constraints_active(w_now, w_max=params["w_max"])
row = dtable.loc[d] if d in dtable.index else None

# Situer la date sur la courbe.
st.plotly_chart(C.nav_locator(bundle.history, bundle.weights, d, index_nav=bundle.index_nav,
                              random_nav=bundle.random_nav),
                use_container_width=True)

# Panneau « Décision » : intrants du jour.
est_vol = float(row["est_vol"]) if row is not None else float("nan")
avg_corr = float(row["avg_corr"]) if row is not None else float("nan")
turn = float(row["turnover"]) if row is not None else float((delta.abs().sum()))
rebal = bool(row["rebalance"]) if row is not None else (turn > 0.01)

d1, d2, d3, d4 = st.columns(4)
d1.metric("Volatilité estimée", f"{est_vol:.1%}" if est_vol == est_vol else "—",
          help="√(wᵀΣw) annualisée — le risque que le modèle PRÉVOIT pour ce portefeuille.")
d2.metric("Corrélation moyenne", f"{avg_corr:.2f}" if avg_corr == avg_corr else "—",
          help="Co-mouvement moyen des actifs sur la fenêtre d'estimation.")
d3.metric("Turnover", f"{turn:.1%}")
d4.metric("Rebalancement", "Oui" if rebal else "Maintien")

# Frise du processus de décision, valuée à la date choisie.
steps = [
    {"icon": " ", "label": "Observation", "value": f"{params['est_window']} {params['frequency']}"},
    {"icon": " ", "label": "Estimation μ", "value": params["mu_name"]},
    {"icon": " ", "label": "Estimation Σ", "value": f"{est_vol:.0%} vol" if est_vol == est_vol else params["cov_name"]},
    {"icon": " ", "label": "Optimisation", "value": params["method_name"]},
    {"icon": " ", "label": "Rebalancement", "value": f"{turn:.0%}"},
    {"icon": " ", "label": "Mise à jour", "value": f"{int((w_now > 1e-4).sum())} positions"},
]
st.markdown(ui.decision_timeline(steps), unsafe_allow_html=True)

left, right = st.columns([0.46, 0.54])
with left:
    st.markdown(f"""
**Lecture de cette décision ({pd.Timestamp(d).date()})**

- Le modèle prévoit une **volatilité de {est_vol:.1%}** pour ce portefeuille.
- Corrélation moyenne **{avg_corr:.2f}** → {"diversification limitée" if avg_corr > 0.5 else "bonne diversification"}.
- Il fait tourner **{turn:.0%}** du portefeuille ({int((w_now > 1e-4).sum())} positions actives).
""")
    if ca["n_at_cap"]:
        st.caption(f"⛔ {ca['n_at_cap']} actif(s) **au plafond** {params['w_max']:.0%} — "
                   "c'est la contrainte, pas le modèle, qui limite la concentration.")
with right:
    st.plotly_chart(C.movements_bar(delta, top=10), use_container_width=True)
