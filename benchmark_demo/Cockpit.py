"""Cockpit — page d'accueil de la démonstration (vue d'ensemble du portefeuille).

Point d'entrée Streamlit. Lancer depuis `benchmark_demo/` :
    streamlit run Cockpit.py

Application en 3 pages, centrée sur le PROCESSUS DE DÉCISION :
    • Cockpit (cette page)  — univers, hypothèses, paramètres expliqués.
    • Simulation            — le portefeuille observe, optimise, rebalance (en direct + replay).
    • Analyse               — performance, risque, marché, régimes + lectures automatiques.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import streamlit as st

from components import theme as T
from components.sidebar import render_sidebar, require_committed, get_bundle, health_banner
from components import charts as C
from core import services
from core.registry import label as _lbl

st.set_page_config(page_title="Portfolio Cockpit", page_icon="", layout="wide",
                   initial_sidebar_state="expanded")
T.inject_css(st)

params, committed = render_sidebar()

st.markdown("<div class='demo-kicker'>Portfolio Lab · Cockpit</div>", unsafe_allow_html=True)
st.title("Vue d'ensemble du portefeuille")
st.caption("Univers, hypothèses et réglages du modèle — avant de simuler.")

params = require_committed(committed)
bundle = get_bundle(params)
health_banner(bundle, params)
st.divider()

left, right = st.columns([0.46, 0.54])
with left:
    st.markdown("### Hypothèses du modèle")
    st.markdown(f"""
| Levier | Choix |
|---|---|
| **Univers** | {_lbl(params['universe_kind'])} — {bundle.n_effective} actifs |
| **Période** | {params['start']} → {params['end']} ({_lbl(params['frequency'])}) |
| **Rendement μ** | {_lbl(params['mu_name'])} |
| **Covariance Σ** | {_lbl(params['cov_name'])} |
| **Méthode** | {_lbl(params['method_name'])} |
| **Poids max / action** | {params['w_max']:.0%} |
| **Frais** | {params['tc_bps']:.0f} bps · taux sans risque {params['rf']:.1%} |
""")
    st.info("**Markowitz suppose** des rendements i.i.d. et des μ, Σ stables. La démo met ces "
            "hypothèses à l'épreuve **hors échantillon**.", icon="💡")

with right:
    st.markdown("### Univers par secteur")
    st.caption("Chaque tuile est une action ; cliquez un secteur pour zoomer.")
    sectors = services.asset_sectors(tuple(bundle.prices.columns))
    st.plotly_chart(C.universe_treemap(sectors), use_container_width=True)

st.divider()

# --- Comprendre les réglages (remplace l'ancien « aperçu de la performance ») --------------
# Choix UX (audit §5) : explication DIRECTEMENT dans la page (Option A) plutôt que des
# tooltips — en soutenance, l'écran est projeté et les tooltips ne se voient pas. Les
# tooltips restent malgré tout sur les widgets de la sidebar pour aider à la manipulation.
st.markdown("### Détail des réglages")
st.caption("Ce que change chaque paramètre, et pourquoi il existe.")


def _setting_card(icon: str, title: str, body: str) -> str:
    return (f"<div class='demo-card' style='height:100%'>"
            f"<div style='font-size:20px'>{icon} <b>{title}</b></div>"
            f"<div style='color:{T.MUTED};font-size:.96rem;line-height:1.5;margin-top:6px'>{body}</div>"
            f"</div>")


cards = [
    (" ", "Historique observé",
     "Le passé que le modèle regarde pour estimer rendement et risque. <b>Long</b> : "
     "estimations stables mais lentes à réagir. <b>Court</b> : réactif mais bruité. "
     "C'est la « période d'entraînement », recalée à chaque rebalancement (walk-forward)."),
    (" ", "Fréquence de rebalancement",
     "Tous les combien le portefeuille est recalculé. <b>Souvent</b> : suit le marché de "
     "près mais paie plus de frais (turnover ↑). <b>Rarement</b> : économe mais dérive."),
    (" ", "Frais par transaction",
     "Chaque achat/vente coûte (courtage, écart bid-ask). 10 bps = 0,10 % du montant "
     "échangé. Sans eux, un backtest <b>surestime</b> systématiquement la performance."),
    (" ", "Poids maximum par action",
     "Empêche le modèle de tout miser sur un titre. "
     "Trop bas avec peu "
     "d'actifs → tout devient équipondéré (le bandeau vous l'indique)."),
    (" ", "Fréquence des données",
     "<b>Quotidien</b> : plus de points, plus de bruit. <b>Hebdomadaire</b> : bon "
     "compromis (défaut). <b>Mensuel</b> : très lisse mais peu d'observations."),
    (" ", "Méthode de portefeuille",
     "L'objectif optimisé. <b>Équipondéré</b> ignore μ/Σ (référence). <b>Risque minimum</b> "
     "ne dépend que de Σ. <b>Markowitz</b> dépend de μ ET Σ (le plus ambitieux)."),
]
for row_start in range(0, len(cards), 3):
    cols = st.columns(3)
    for col, (icon, title, body) in zip(cols, cards[row_start:row_start + 3]):
        col.markdown(_setting_card(icon, title, body), unsafe_allow_html=True)

st.markdown("")

