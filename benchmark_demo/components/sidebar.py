"""Barre latérale GLOBALE : le « pupitre du gérant ».

Rend les widgets de configuration (univers, période, fréquence, μ, Σ, méthode, w_max,
coûts, rf, fenêtres), stocke les paramètres dans `st.session_state`, et expose un helper
`get_bundle()` qui lance (en cache) le backtest partagé par toutes les pages. Affiche aussi
le garde-fou de dégénérescence N×w_max hérité de l'audit.
"""
from __future__ import annotations

import datetime as _dt

import streamlit as st

from core import registry as reg
from core import services
from core.constraints import degeneracy_check
from components import theme as T


_PPY = {"daily": 252, "weekly": 52, "monthly": 12}

# Presets : règlent d'un coup tous les réglages « avancés ». Les clés correspondent aux
# `key=` des widgets ; on les écrit dans session_state via un callback (avant re-render).
_PRESETS: dict[str, dict] = {
    "Standard": dict(cfg_univ="StratifiedBySector", cfg_n=40, cfg_method="Mean Variance",
                     cfg_start=_dt.date(2015, 1, 1), cfg_end=_dt.date(2024, 12, 31),
                     cfg_freq="weekly", cfg_mu="SampleMean", cfg_cov="Ledoit-Wolf",
                     cfg_wmax=0.10, cfg_tc=10.0, cfg_rf=0.04, cfg_win=52, cfg_hold=1),
    "Démo rapide": dict(cfg_univ="StratifiedBySector", cfg_n=40, cfg_method="Mean Variance",
                        cfg_start=_dt.date(2018, 1, 1), cfg_end=_dt.date(2024, 12, 31),
                        cfg_freq="weekly", cfg_mu="James-Stein", cfg_cov="Ledoit-Wolf",
                        cfg_wmax=0.15, cfg_tc=10.0, cfg_rf=0.04, cfg_win=52, cfg_hold=1),
    "Recherche": dict(cfg_univ="StratifiedBySector", cfg_n=60, cfg_method="Mean Variance",
                      cfg_start=_dt.date(2012, 1, 1), cfg_end=_dt.date(2024, 12, 31),
                      cfg_freq="weekly", cfg_mu="James-Stein", cfg_cov="Ledoit-Wolf",
                      cfg_wmax=0.08, cfg_tc=10.0, cfg_rf=0.04, cfg_win=104, cfg_hold=1),
}


def _apply_preset() -> None:
    """Callback du selectbox de preset : écrit les réglages dans session_state (avant render)."""
    preset = st.session_state.get("cfg_preset")
    for k, v in _PRESETS.get(preset, {}).items():
        st.session_state[k] = v


def _estimate_periods(start: _dt.date, end: _dt.date, frequency: str) -> int:
    """Estimation grossière du nombre d'observations sur la période (pour le garde-fou fenêtre)."""
    years = max((end - start).days, 1) / 365.25
    return int(years * _PPY.get(frequency, 52))


def _seed_defaults() -> None:
    """Amorce une seule fois les valeurs par défaut (preset Standard) dans session_state.

    Permet ensuite d'utiliser les widgets avec UNIQUEMENT `key=` (sans `value=`/`index=`),
    ce qui évite l'avertissement Streamlit « default value + Session State » et laisse les
    presets piloter tous les widgets en écrivant leur clé interne.
    """
    if st.session_state.get("_cfg_seeded"):
        return
    for k, v in _PRESETS["Standard"].items():
        st.session_state.setdefault(k, v)
    st.session_state.setdefault("cfg_synth", False)
    st.session_state["_cfg_seeded"] = True


def _persist_widget_state() -> None:
    """Conserve les réglages d'une PAGE à l'autre.

    Piège Streamlit multipage : l'état d'un widget identifié par `key` est purgé dès que la
    page qui le porte n'est plus à l'écran (les `cfg_*` reviendraient alors aux défauts à
    chaque navigation). En réécrivant chaque clé sur elle-même AVANT de réinstancier les
    widgets, on « promeut » la valeur en entrée de session_state durable : elle survit au
    changement de page. (Parade documentée et sans effet de bord.)
    """
    for k in list(st.session_state.keys()):
        if k.startswith("cfg_"):
            st.session_state[k] = st.session_state[k]


def render_sidebar() -> dict:
    """Sidebar structurée : Preset → essentiel → ▶ Lancer (visible) → 3 groupes repliables
    (Données / Modèle / Protocole). Renvoie (params, committed) avec les MÊMES clés qu'avant.

    Les selectbox affichent un libellé HUMAIN (`format_func`) mais stockent la clé INTERNE :
    presets et `params` continuent de manipuler les identifiants techniques.
    """
    _seed_defaults()
    _persist_widget_state()
    st.sidebar.markdown("##  Configuration")

    # --- Preset (règle tout l'avancé d'un coup) ---
    st.sidebar.selectbox("Préréglage", list(_PRESETS), index=0, key="cfg_preset",
                         on_change=_apply_preset,
                         help="« Démo rapide » pour une présentation, « Recherche » pour un "
                              "historique long. Vous pouvez ensuite ajuster chaque réglage.")

    # --- Essentiel (toujours visible) ---
    universe_kind = st.sidebar.selectbox("Univers", reg.UNIVERSE_KINDS, key="cfg_univ",
                                         format_func=reg.label,
                                         help="Comment on choisit les actifs du portefeuille.")
    n = st.sidebar.slider("Nombre d'actifs", 10, 120, key="cfg_n", step=5,
                          disabled=(universe_kind == "All"))
    method_name = st.sidebar.selectbox("Méthode de portefeuille", list(reg.PORTFOLIO_METHODS),
                                       key="cfg_method", format_func=reg.label,
                                       help="L'objectif que l'optimiseur cherche à atteindre.")
    st.sidebar.caption(reg.describe_method(method_name))

    # --- Bouton RUN : REMONTÉ ici, toujours visible (avant les expanders). ---
    run_clicked = st.sidebar.button("▶  Lancer la simulation", type="primary",
                                    use_container_width=True)

    # --- Groupe 1 : Données ---
    with st.sidebar.expander(" Données", expanded=False):
        c1, c2 = st.columns(2)
        start_d = c1.date_input("Début", key="cfg_start", format="YYYY-MM-DD",
                                min_value=_dt.date(2005, 1, 1), max_value=_dt.date(2025, 1, 1))
        end_d = c2.date_input("Fin", key="cfg_end", format="YYYY-MM-DD",
                              min_value=_dt.date(2005, 6, 1), max_value=_dt.date(2025, 1, 1))
        frequency = st.selectbox("Fréquence des données", ["weekly", "daily", "monthly"],
                                 key="cfg_freq", format_func=reg.label,
                                 help="Hebdomadaire = bon compromis bruit/observations (défaut).")
        use_synthetic = st.toggle("Données simulées (sans internet)", key="cfg_synth",
                                  help="Décoché (défaut) : vraies données yfinance. "
                                       "Coché : prix simulés, utile en salle sans réseau.")

    # --- Groupe 2 : Modèle ---
    with st.sidebar.expander(" Modèle", expanded=False):
        mu_name = st.selectbox("Prévision de rendement (μ)", list(reg.MU_ESTIMATORS),
                               key="cfg_mu", format_func=reg.label,
                               help="Comment on estime le rendement attendu de chaque actif.")
        cov_name = st.selectbox("Modèle de risque (Σ)", list(reg.COV_ESTIMATORS),
                                key="cfg_cov", format_func=reg.label,
                                help="Comment on estime la covariance (le risque) des actifs.")
        w_max = st.slider("Poids maximum par action", 0.02, 1.0, key="cfg_wmax", step=0.01,
                          help="Plafond par actif (noté w_max). Protège contre les paris "
                               "extrêmes de Markowitz.")

    # --- Groupe 3 : Protocole ---
    with st.sidebar.expander(" Protocole de backtest", expanded=False):
        est_window = st.number_input("Historique observé (périodes)", 12, 756, step=4, key="cfg_win",
                                     help="Le passé que le modèle regarde pour estimer μ et Σ "
                                          "(la « fenêtre d'estimation », recalée à chaque rebalancement).")
        holding = st.number_input("Fréquence de rebalancement (périodes)", 1, 60, step=1, key="cfg_hold",
                                  help="Tous les combien le portefeuille est recalculé.")
        # Garde-fou fenêtre : estime le nombre de rebalancements possibles sur la période.
        T_est = _estimate_periods(start_d, end_d, frequency)
        n_rebal = max(0, (T_est - int(est_window)) // max(int(holding), 1))
        if int(est_window) >= T_est:
            st.error(f"Fenêtre ({est_window}) ≥ historique disponible (~{T_est} périodes) : "
                     "aucun rebalancement possible. Réduisez la fenêtre ou élargissez la période.")
        elif int(est_window) > 0.5 * T_est:
            st.warning(f"Fenêtre longue : ~{n_rebal} rebalancements seulement sur cette période.")
        else:
            st.caption(f"≈ {T_est} périodes disponibles · ~{n_rebal} rebalancements.")
        tc_bps = st.slider("Frais par transaction (bps)", 0.0, 50.0, key="cfg_tc", step=1.0,
                           help="10 bps = 0,10 % du montant échangé. Sans eux, un backtest "
                                "surestime la performance.")
        rf = st.slider("Taux sans risque annuel", 0.0, 0.08, key="cfg_rf", step=0.005,
                       help="Sert à calculer le Sharpe (rendement EXCÉDENTAIRE / risque).")

    params = dict(universe_kind=universe_kind, n=int(n),
                  start=start_d.isoformat(), end=end_d.isoformat(),
                  frequency=frequency, mu_name=mu_name, cov_name=cov_name,
                  method_name=method_name, w_max=float(w_max), tc_bps=float(tc_bps),
                  rf=float(rf), est_window=int(est_window), holding=int(holding),
                  use_synthetic=bool(use_synthetic))
    st.session_state["params"] = params

    # --- Validation au Run : on COMMIT sauf si la config est manifestement invalide. ---
    st.sidebar.divider()
    if run_clicked:
        if int(est_window) >= _estimate_periods(start_d, end_d, frequency):
            st.sidebar.error("Config invalide : fenêtre trop longue pour la période. Non lancée.")
        else:
            st.session_state["committed_params"] = params
            if int(est_window) < int(n) and cov_name == "Empirical":
                st.sidebar.warning("Σ empirique avec fenêtre < nombre d'actifs : matrice "
                                   "probablement singulière. Préférez Ledoit-Wolf.")
    committed = st.session_state.get("committed_params")
    if committed is None:
        st.sidebar.caption("Réglez les paramètres puis lancez.")
    elif committed != params:
        st.sidebar.caption("⚠️ Paramètres modifiés — relancez pour appliquer.")
    else:
        st.sidebar.caption("✅ Simulation à jour.")
    return params, committed


def require_committed(committed: dict | None) -> dict:
    """Bloque la page tant que l'utilisateur n'a pas cliqué « Lancer la simulation ».

    Renvoie les paramètres VALIDÉS (ceux du dernier Run), de sorte que l'affichage reflète
    la simulation réellement calculée — pas les réglages en cours non encore appliqués.
    """
    if committed is None:
        st.info("👈 Réglez les paramètres dans la barre latérale, puis cliquez "
                "**▶ Lancer la simulation**.")
        st.stop()
    return committed


@st.cache_data(show_spinner="Backtest walk-forward en cours…")
def _cached_bundle(params_tuple: tuple) -> services.BacktestBundle:
    p = dict(params_tuple)
    return services.run_backtest(**p)


def get_bundle(params: dict) -> services.BacktestBundle:
    """Lance (ou récupère en cache) le backtest correspondant aux paramètres courants."""
    return _cached_bundle(tuple(sorted(params.items())))


def health_banner(bundle: services.BacktestBundle, params: dict) -> None:
    """Bandeau de contrôles : univers effectif, DIVERSIFICATION EFFECTIVE (nb effectif
    d'actifs = 1/Herfindahl), fallbacks. Avertit si le portefeuille est (quasi) dégénéré."""
    import numpy as np
    W = bundle.weights
    hhi = float((W ** 2).sum(axis=1).mean()) if len(W) else np.nan
    eff_n = (1.0 / hhi) if hhi and hhi > 0 else np.nan
    dg = degeneracy_check(bundle.n_effective, params["w_max"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Univers effectif", f"{bundle.n_effective} actifs",
              help="Titres réellement disponibles après nettoyage.")
    c2.metric("Diversification effective", f"{eff_n:.0f} actifs" if eff_n == eff_n else "—",
              help="Nombre effectif de positions (1/Herfindahl). Plus haut = mieux réparti.")
    fb = int(bundle.metrics.get("Fallback count", 0))
    c3.metric("Fallbacks", fb, help="Fenêtres où l'optimisation était infaisable.")
    if not dg["ok"]:
        st.warning("Diversification contrainte : l'espace admissible est très réduit "
                   f"(plafond {params['w_max']:.0%} sur {bundle.n_effective} actifs) — "
                   "les portefeuilles tendent vers l'équipondéré. Augmentez le nombre d'actifs.")


def page_header(kicker: str, title: str, subtitle: str = "") -> None:
    """En-tête de page homogène (kicker cyan + titre + sous-titre)."""
    T.inject_css(st)
    st.markdown(f"<div class='demo-kicker'>{kicker}</div>", unsafe_allow_html=True)
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


__all__ = ["render_sidebar", "require_committed", "get_bundle", "health_banner", "page_header"]
