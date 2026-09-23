"""Fabriques de figures Plotly (pures, sans Streamlit) pour toutes les pages de la démo.

Chaque fonction prend des objets pandas/numpy et renvoie une `go.Figure` déjà stylée par le
thème institutionnel. La logique de calcul vit dans `core/` ; ici on ne fait que *dessiner*.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import theme as T


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _top_assets(weights: pd.DataFrame, top: int) -> list[str]:
    """Garde les `top` actifs au poids maximal dans le temps (lisibilité)."""
    order = weights.abs().max().sort_values(ascending=False)
    return list(order.head(top).index)


def _empty(msg: str = "Aucune donnée") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(color=T.MUTED, size=14))
    return T.apply(fig, height=300)


# --------------------------------------------------------------------------- #
#  Page 3 — Performance                                                        #
# --------------------------------------------------------------------------- #
#: couleur dédiée au benchmark « poids aléatoires » (violet, distinct du cyan/gris/orange).
RANDOM_COLOR = "#A855F7"


def nav_chart(history: pd.DataFrame, *, log: bool = True,
              index_nav: pd.Series | None = None,
              random_nav: pd.Series | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.index, y=history["benchmark_nav"], name="Équipondéré 1/N",
                             line=dict(color=T.BENCH, width=1.6, dash="dash")))
    if index_nav is not None and len(index_nav):
        fig.add_trace(go.Scatter(x=index_nav.index, y=index_nav.values, name="S&P 500 (indice)",
                                 line=dict(color=T.WARN, width=1.6, dash="dot")))
    if random_nav is not None and len(random_nav):
        fig.add_trace(go.Scatter(x=random_nav.index, y=random_nav.values, name="Poids aléatoires",
                                 line=dict(color=RANDOM_COLOR, width=1.4, dash="dashdot")))
    fig.add_trace(go.Scatter(x=history.index, y=history["strategy_nav"], name="Stratégie",
                             line=dict(color=T.ACCENT, width=2.4)))
    fig.update_yaxes(type="log" if log else "linear")
    return T.apply(fig, height=420, title="Valeur du portefeuille (base 1)")


def drawdown_chart(drawdown: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown * 100, name="Drawdown",
                             fill="tozeroy", line=dict(color=T.NEG, width=1),
                             fillcolor="rgba(239,68,68,0.25)"))
    fig.update_yaxes(ticksuffix="%")
    return T.apply(fig, height=260, title="Drawdown")


def rolling_chart(rolling: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Volatilité glissante (annualisée)", "Sharpe glissant"))
    fig.add_trace(go.Scatter(x=rolling.index, y=rolling["rolling_vol"] * 100,
                             line=dict(color=T.WARN, width=1.8), name="Vol"), row=1, col=1)
    fig.add_trace(go.Scatter(x=rolling.index, y=rolling["rolling_sharpe"],
                             line=dict(color=T.ACCENT, width=1.8), name="Sharpe"), row=2, col=1)
    fig.add_hline(y=0, line=dict(color=T.GRID, width=1), row=2, col=1)
    fig.update_yaxes(ticksuffix="%", row=1, col=1)
    fig.update_layout(showlegend=False)
    return T.apply(fig, height=420)


# --------------------------------------------------------------------------- #
#  Page 4 — Allocations                                                        #
# --------------------------------------------------------------------------- #
def weights_stacked_area(weights: pd.DataFrame, *, top: int = 12) -> go.Figure:
    cols = _top_assets(weights, top)
    others = weights.drop(columns=cols, errors="ignore").sum(axis=1)
    fig = go.Figure()
    for i, c in enumerate(cols):
        fig.add_trace(go.Scatter(x=weights.index, y=weights[c] * 100, name=c, mode="lines",
                                 stackgroup="w", line=dict(width=0.5),
                                 fillcolor=T.CATEGORICAL[i % len(T.CATEGORICAL)]))
    if (others > 1e-9).any():
        fig.add_trace(go.Scatter(x=weights.index, y=others * 100, name="Autres", mode="lines",
                                 stackgroup="w", line=dict(width=0.5), fillcolor=T.MUTED))
    fig.update_yaxes(ticksuffix="%", range=[0, 100])
    return T.apply(fig, height=420, title="Composition du portefeuille dans le temps")


def sankey_rebalance(weights: pd.DataFrame, date_from, date_to, *, top: int = 10) -> go.Figure:
    cols = _top_assets(weights, top)
    a = weights.loc[date_from, cols] if date_from in weights.index else weights.iloc[0][cols]
    b = weights.loc[date_to, cols] if date_to in weights.index else weights.iloc[-1][cols]
    labels = [f"{c} (t)" for c in cols] + [f"{c} (t+1)" for c in cols]
    src, tgt, val = [], [], []
    for i, c in enumerate(cols):
        # Flux : le min(a,b) reste, le reste se réalloue proportionnellement (vue simplifiée).
        keep = float(min(a[c], b[c]))
        if keep > 1e-4:
            src.append(i); tgt.append(len(cols) + i); val.append(keep)
    # Réallocation du surplus vers les actifs qui montent.
    inc = {c: max(float(b[c] - a[c]), 0.0) for c in cols}
    dec = {c: max(float(a[c] - b[c]), 0.0) for c in cols}
    tot_inc = sum(inc.values()) or 1.0
    for i, c in enumerate(cols):
        if dec[c] > 1e-4:
            for j, d in enumerate(cols):
                if inc[d] > 1e-4:
                    src.append(i); tgt.append(len(cols) + j)
                    val.append(dec[c] * inc[d] / tot_inc)
    fig = go.Figure(go.Sankey(
        node=dict(label=labels, pad=14, thickness=14,
                  color=(T.CATEGORICAL * 3)[:len(labels)]),
        link=dict(source=src, target=tgt, value=val,
                  color="rgba(40,192,199,0.25)")))
    return T.apply(fig, height=420, title="Flux de rebalancement (t → t+1)")


def _sectors_for(tickers, sectors: pd.Series | None) -> list[str]:
    if sectors is None:
        return ["Portefeuille"] * len(tickers)
    return [str(sectors.get(t, "Autre")) for t in tickers]


def _hierarchy(weights_row: pd.Series, sectors: pd.Series | None):
    """Construit (labels, parents, values) hiérarchiques cohérents avec branchvalues='total' :
    chaque nœud SECTEUR a pour valeur la SOMME de ses actifs (sinon Plotly n'affiche rien)."""
    w = weights_row[weights_row > 1e-4]
    secs = _sectors_for(w.index, sectors)
    sector_total: dict[str, float] = {}
    for s, v in zip(secs, w.values):
        sector_total[s] = sector_total.get(s, 0.0) + float(v) * 100
    labels = list(sector_total) + list(w.index)
    parents = [""] * len(sector_total) + secs
    values = list(sector_total.values()) + [float(v) * 100 for v in w.values]
    return labels, parents, values


def sunburst_allocation(weights_row: pd.Series, sectors: pd.Series | None = None) -> go.Figure:
    labels, parents, values = _hierarchy(weights_row, sectors)
    if not labels:
        return _empty("Aucune position")
    fig = go.Figure(go.Sunburst(labels=labels, parents=parents, values=values,
                                branchvalues="total", insidetextorientation="radial",
                                marker=dict(colors=list(range(len(labels))),
                                            colorscale="Teal", line=dict(color=T.BG, width=1))))
    return T.apply(fig, height=440, title="Répartition secteur → actif")


def treemap_allocation(weights_row: pd.Series, sectors: pd.Series | None = None) -> go.Figure:
    labels, parents, values = _hierarchy(weights_row, sectors)
    if not labels:
        return _empty("Aucune position")
    fig = go.Figure(go.Treemap(labels=labels, parents=parents, values=values,
                               branchvalues="total", textinfo="label+value+percent parent",
                               marker=dict(colors=list(range(len(labels))), colorscale="Teal",
                                           line=dict(color=T.BG, width=1))))
    return T.apply(fig, height=440, title="Treemap des poids (secteur → actif)")


def universe_treemap(sectors: pd.Series) -> go.Figure:
    """Treemap à DEUX niveaux de l'univers : secteur → actifs (équipondérés visuellement).

    Chaque actif est une tuile nominative sous son secteur ; cliquer un secteur zoome
    nativement (Plotly). Le hover donne secteur + ticker. Contrairement à un treemap de
    secteurs seuls, l'utilisateur VOIT les titres et peut explorer l'univers d'un clic.
    """
    s = pd.Series(sectors).fillna("Autre")
    if s.empty:
        return _empty("Univers vide")
    sector_counts = s.value_counts()
    # Niveau 1 : secteurs (valeur = nb d'actifs). Niveau 2 : actifs (valeur = 1 chacun).
    labels = list(sector_counts.index) + list(s.index)
    parents = [""] * len(sector_counts) + list(s.values)
    values = list(sector_counts.values) + [1] * len(s)
    # Couleur héritée du secteur parent (palette catégorielle stable).
    sec_color = {sec: T.CATEGORICAL[i % len(T.CATEGORICAL)]
                 for i, sec in enumerate(sector_counts.index)}
    colors = [sec_color[sec] for sec in sector_counts.index] + [sec_color[sec] for sec in s.values]
    fig = go.Figure(go.Treemap(
        labels=labels, parents=parents, values=values, branchvalues="total",
        textinfo="label", textfont=dict(size=13),
        marker=dict(colors=colors, line=dict(color=T.BG, width=1.5)),
        hovertemplate="<b>%{label}</b><br>%{parent}<extra></extra>",
        tiling=dict(pad=2), pathbar=dict(visible=True, thickness=18)))
    return T.apply(fig, height=320, title=None)


def weights_bar(weights_row: pd.Series, *, top: int = 15) -> go.Figure:
    w = weights_row[weights_row > 1e-4].sort_values(ascending=True).tail(top)
    fig = go.Figure(go.Bar(x=w.values * 100, y=list(w.index), orientation="h",
                           marker=dict(color=T.ACCENT)))
    fig.update_xaxes(ticksuffix="%")
    return T.apply(fig, height=420, title="Poids actuels")


def movements_bar(delta: pd.Series, *, top: int = 10) -> go.Figure:
    """Barres divergentes des plus gros CHANGEMENTS de poids vs le rebalancement précédent
    (achats en vert, ventes en rouge). Montre la DÉCISION, pas l'état (déjà dans le donut)."""
    d = delta[delta.abs() > 1e-4]
    if d.empty:
        return _empty("Aucun mouvement (allocation inchangée)")
    d = d.reindex(d.abs().sort_values(ascending=False).index).head(top).sort_values()
    colors = [T.POS if v > 0 else T.NEG for v in d.values]
    fig = go.Figure(go.Bar(x=d.values * 100, y=list(d.index), orientation="h",
                           marker=dict(color=colors),
                           hovertemplate="%{y} : %{x:+.1f} pt<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=T.MUTED, width=1))
    fig.update_xaxes(ticksuffix="%", title="Δ poids (points)")
    return T.apply(fig, height=380, title="Mouvements vs semaine précédente")


# --------------------------------------------------------------------------- #
#  Page 2 — Animation « gérant en direct »                                     #
# --------------------------------------------------------------------------- #
def manager_animation(history: pd.DataFrame, weights: pd.DataFrame, *, top: int = 8,
                      max_frames: int = 60, index_nav: pd.Series | None = None,
                      random_nav: pd.Series | None = None) -> go.Figure:
    """Animation épurée « gérant en direct » : à gauche la VALEUR qui se remplit (aire), à
    droite un DONUT d'allocation qui se déforme en douceur. Frames sous-échantillonnées
    (≤ `max_frames`) pour une simulation accélérée et fluide. Play/Pause + slider intégrés.

    Trace, à côté de la stratégie : le 1/N (équipondéré), le S&P 500 (si `index_nav`) et un
    portefeuille à poids aléatoires (si `random_nav`). Une légende identifie chaque courbe.
    """
    if weights.empty:
        return _empty("Pas de rebalancements à animer")
    all_dates = list(weights.index)
    stride = max(1, len(all_dates) // max_frames)
    dates = all_dates[::stride]
    if all_dates[-1] not in dates:
        dates.append(all_dates[-1])

    nav = history["strategy_nav"]
    bench = history["benchmark_nav"]
    sp = (index_nav.reindex(nav.index).ffill() if index_nav is not None and len(index_nav)
          else pd.Series(dtype=float))
    has_sp = bool(len(sp.dropna()))
    rnd = (random_nav.reindex(nav.index).ffill() if random_nav is not None and len(random_nav)
           else pd.Series(dtype=float))
    has_rnd = bool(len(rnd.dropna()))

    # Couleur STABLE par actif (sur l'union des titres jamais détenus > 1 %) : un actif
    # garde sa couleur d'une date à l'autre. On affiche les positions RÉELLES du jour.
    held_ever = [c for c in weights.columns if (weights[c] > 0.01).any()]
    amap = {a: T.CATEGORICAL[i % len(T.CATEGORICAL)] for i, a in enumerate(held_ever)}

    def donut(d, k: int = 12):
        wd = weights.loc[d]
        head = wd[wd > 0.01].sort_values(ascending=False).head(k)
        autres = max(float(wd.sum() - head.sum()), 0.0)
        labels = list(head.index) + (["Autres"] if autres > 1e-3 else [])
        values = list(head.values * 100) + ([autres * 100] if autres > 1e-3 else [])
        colors = [amap.get(a, T.MUTED) for a in head.index] + ([T.MUTED] if autres > 1e-3 else [])
        return labels, values, colors

    fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38],
                        horizontal_spacing=0.06,
                        subplot_titles=("Valeur du portefeuille", "Positions détenues"),
                        specs=[[{"type": "scatter"}, {"type": "domain"}]])

    d0 = dates[0]
    nav0, bench0 = nav.loc[:d0], bench.loc[:d0]
    sp0 = sp.loc[:d0] if has_sp else pd.Series(dtype=float)
    rnd0 = rnd.loc[:d0] if has_rnd else pd.Series(dtype=float)
    lab0, val0, col0 = donut(d0)
    fig.add_trace(go.Scatter(x=bench0.index, y=bench0.values, name="Équipondéré 1/N",
                             line=dict(color=T.BENCH, width=1.6, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=sp0.index, y=sp0.values, name="S&P 500 (indice)",
                             line=dict(color=T.WARN, width=1.4, dash="dot"),
                             showlegend=has_sp), row=1, col=1)
    fig.add_trace(go.Scatter(x=rnd0.index, y=rnd0.values, name="Poids aléatoires",
                             line=dict(color=RANDOM_COLOR, width=1.4, dash="dashdot"),
                             showlegend=has_rnd), row=1, col=1)
    fig.add_trace(go.Scatter(x=nav0.index, y=nav0.values, name="Stratégie", fill="tozeroy",
                             line=dict(color=T.ACCENT, width=2.8),
                             fillcolor="rgba(40,192,199,0.18)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[d0], y=[float(nav.loc[d0])], mode="markers", name="Rebalancement",
                             marker=dict(color=T.BG, size=7, symbol="diamond",
                                         line=dict(color=T.ACCENT, width=1.5)), showlegend=False,
                             hovertemplate="Rebalancement<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Pie(labels=lab0, values=val0, hole=0.62, sort=True,
                         direction="clockwise", marker=dict(colors=col0, line=dict(color=T.BG, width=1.5)),
                         textinfo="label", textposition="inside", insidetextorientation="horizontal",
                         textfont=dict(size=12, color=T.BG),
                         hovertemplate="%{label} : %{value:.1f}%<extra></extra>"), row=1, col=2)

    frames = []
    for d in dates:
        navd, benchd = nav.loc[:d], bench.loc[:d]
        spd = sp.loc[:d] if has_sp else pd.Series(dtype=float)
        rndd = rnd.loc[:d] if has_rnd else pd.Series(dtype=float)
        lab, val, colr = donut(d)
        mk = [x for x in dates if x <= d]
        frames.append(go.Frame(name=str(pd.Timestamp(d).date()), data=[
            go.Scatter(x=benchd.index, y=benchd.values),
            go.Scatter(x=spd.index, y=spd.values),
            go.Scatter(x=rndd.index, y=rndd.values),
            go.Scatter(x=navd.index, y=navd.values),
            go.Scatter(x=mk, y=[float(nav.loc[x]) for x in mk]),
            go.Pie(labels=lab, values=val, marker=dict(colors=colr, line=dict(color=T.BG, width=1.5))),
        ], traces=[0, 1, 2, 3, 4, 5]))
    fig.frames = frames

    _mins = [nav.min(), bench.min()] + ([sp.min()] if has_sp else []) + ([rnd.min()] if has_rnd else [])
    _maxs = [nav.max(), bench.max()] + ([sp.max()] if has_sp else []) + ([rnd.max()] if has_rnd else [])
    lo = float(min(_mins)) * 0.97
    hi = float(max(_maxs)) * 1.04
    fig.update_yaxes(title_text=None, range=[lo, hi], row=1, col=1)
    fig.update_xaxes(range=[nav.index.min(), nav.index.max()], row=1, col=1)

    # Play et Pause dans DEUX menus séparés : Plotly applique sinon son style « bouton actif »
    # (fond blanc délavé) au bouton sélectionné. `showactive=False` + un menu par bouton
    # supprime cet état et permet de brander le Play à l'accent cyan (cohérent avec « Lancer
    # la simulation » de la sidebar).
    play = dict(label="▶  Lancer", method="animate",
                args=[None, dict(frame=dict(duration=260, redraw=True), fromcurrent=True,
                                 transition=dict(duration=240, easing="cubic-in-out"))])
    pause = dict(label="⏸  Pause", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="v", x=0.01, y=0.99, xanchor="left", yanchor="top",
                    bgcolor="rgba(14,17,23,0.55)", bordercolor=T.GRID, borderwidth=1,
                    font=dict(size=11, color=T.TEXT)),
        updatemenus=[
            dict(type="buttons", direction="left", x=0.0, y=1.22, xanchor="left",
                 showactive=False, pad=dict(r=8, t=4),
                 bgcolor=T.ACCENT, bordercolor=T.ACCENT,
                 font=dict(color=T.BG, size=14, family="Inter"), buttons=[play]),
            dict(type="buttons", direction="left", x=0.135, y=1.22, xanchor="left",
                 showactive=False, pad=dict(r=8, t=4),
                 bgcolor=T.PANEL, bordercolor=T.GRID,
                 font=dict(color=T.TEXT, size=14, family="Inter"), buttons=[pause]),
        ],
        sliders=[dict(active=0, y=-0.04, x=0.0, len=1.0, pad=dict(t=6),
                      currentvalue=dict(prefix="Semaine du ", font=dict(color=T.ACCENT, size=15)),
                      font=dict(size=11, color=T.MUTED),
                      steps=[dict(method="animate", label=f.name,
                                  args=[[f.name], dict(mode="immediate",
                                        frame=dict(duration=0, redraw=True))]) for f in frames])],
    )
    return T.apply(fig, height=500)


def nav_locator(history: pd.DataFrame, weights: pd.DataFrame, selected_date,
                index_nav: pd.Series | None = None,
                random_nav: pd.Series | None = None) -> go.Figure:
    """NAV avec TICKS de rebalancement et la date sélectionnée surlignée (situe le curseur
    temporel sur la courbe). Compact : accompagne le slider de la page Simulation."""
    nav = history["strategy_nav"]
    bench = history["benchmark_nav"]
    rdates = list(weights.index)
    stride = max(1, len(rdates) // 80)
    ticks = rdates[::stride]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bench.index, y=bench.values, name="Équipondéré 1/N",
                             line=dict(color=T.BENCH, width=1.2, dash="dash")))
    if index_nav is not None and len(index_nav):
        fig.add_trace(go.Scatter(x=index_nav.index, y=index_nav.values, name="S&P 500 (indice)",
                                 line=dict(color=T.WARN, width=1.2, dash="dot")))
    if random_nav is not None and len(random_nav):
        fig.add_trace(go.Scatter(x=random_nav.index, y=random_nav.values, name="Poids aléatoires",
                                 line=dict(color=RANDOM_COLOR, width=1.2, dash="dashdot")))
    fig.add_trace(go.Scatter(x=nav.index, y=nav.values, name="Stratégie",
                             line=dict(color=T.ACCENT, width=2.2)))
    fig.add_trace(go.Scatter(x=ticks, y=[float(nav.loc[d]) for d in ticks], mode="markers",
                             name="Rebalancements", marker=dict(color=T.BG, size=5, symbol="diamond",
                             line=dict(color=T.ACCENT, width=1)), hoverinfo="x"))
    if selected_date is not None:
        sd = pd.Timestamp(selected_date)
        fig.add_vline(x=sd, line=dict(color=T.WARN, width=2))
        if sd in nav.index:
            fig.add_trace(go.Scatter(x=[sd], y=[float(nav.loc[sd])], mode="markers",
                          name="Sélection", marker=dict(color=T.WARN, size=12, symbol="circle",
                          line=dict(color=T.BG, width=1.5))))
    fig.update_layout(showlegend=True,
                      legend=dict(orientation="h", y=1.02, x=0.0, font=dict(size=10),
                                  bgcolor="rgba(0,0,0,0)"))
    return T.apply(fig, height=240, title=None)


# --------------------------------------------------------------------------- #
#  Page 5 — Marché                                                             #
# --------------------------------------------------------------------------- #
def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=corr.values, x=list(corr.columns), y=list(corr.index),
                               zmin=-1, zmax=1, colorscale="RdBu_r",
                               colorbar=dict(title="ρ")))
    return T.apply(fig, height=520, title="Matrice de corrélation")


def distribution_hist(returns: pd.Series, x_norm=None, y_norm=None) -> go.Figure:
    r = pd.Series(returns).dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=r.values, histnorm="probability density", name="Rendements",
                               marker=dict(color="rgba(40,192,199,0.55)"), nbinsx=60))
    if x_norm is not None and len(x_norm):
        fig.add_trace(go.Scatter(x=x_norm, y=y_norm, name="Loi normale ajustée",
                                 line=dict(color=T.WARN, width=2)))
    return T.apply(fig, height=380, title="Distribution des rendements")


def acf_bar(ap: dict, *, which: str = "acf") -> go.Figure:
    vals, conf = ap[which], ap["conf"]
    fig = go.Figure(go.Bar(x=ap["lags"], y=vals, marker=dict(color=T.ACCENT)))
    fig.add_hline(y=conf, line=dict(color=T.NEG, width=1, dash="dot"))
    fig.add_hline(y=-conf, line=dict(color=T.NEG, width=1, dash="dot"))
    fig.add_hline(y=0, line=dict(color=T.GRID, width=1))
    return T.apply(fig, height=300, title=("ACF" if which == "acf" else "PACF"))


def rolling_corr_chart(avg_corr: pd.Series) -> go.Figure:
    fig = go.Figure(go.Scatter(x=avg_corr.index, y=avg_corr.values, name="Corr. moyenne",
                               line=dict(color=T.ACCENT, width=1.8), fill="tozeroy",
                               fillcolor="rgba(40,192,199,0.12)"))
    return T.apply(fig, height=300, title="Corrélation moyenne glissante (co-mouvement)")


def add_regime_bands(fig: go.Figure, spans: list[dict]) -> go.Figure:
    """Superpose des bandes verticales colorées (bull/bear/crise) sur un axe temporel."""
    for s in spans:
        fig.add_vrect(x0=s["start"], x1=s["end"], line_width=0,
                      fillcolor=T.REGIME_COLORS.get(s["regime"], "rgba(0,0,0,0)"), layer="below")
    return fig


# --------------------------------------------------------------------------- #
#  Page 6 — Comparaison                                                        #
# --------------------------------------------------------------------------- #
def nav_overlay(navs: dict) -> go.Figure:
    fig = go.Figure()
    for i, (name, nav) in enumerate(navs.items()):
        fig.add_trace(go.Scatter(x=nav.index, y=nav.values, name=name,
                                 line=dict(width=2, color=T.CATEGORICAL[i % len(T.CATEGORICAL)])))
    fig.update_yaxes(type="log")
    return T.apply(fig, height=420, title="Valeur comparée (échelle log)")


def risk_return_scatter(metrics: pd.DataFrame) -> go.Figure:
    x = metrics["Vol annualized (%)"]
    y = metrics["CAGR (%)"]
    size = metrics["Max drawdown (%)"].abs().clip(lower=1)
    fig = go.Figure(go.Scatter(
        x=x, y=y, mode="markers+text", text=metrics.index, textposition="top center",
        marker=dict(size=size, sizemode="area", sizeref=2.0 * size.max() / 40**2,
                    color=T.CATEGORICAL[:len(metrics)], line=dict(color=T.TEXT, width=1)),
        textfont=dict(color=T.MUTED)))
    fig.update_xaxes(title="Volatilité annualisée (%)")
    fig.update_yaxes(title="CAGR (%)")
    return T.apply(fig, height=440, title="Risque / rendement (bulle = drawdown)")


def bar_compare(series: pd.Series, title: str, *, suffix: str = "") -> go.Figure:
    s = series.sort_values()
    fig = go.Figure(go.Bar(x=s.values, y=list(s.index), orientation="h",
                           marker=dict(color=T.CATEGORICAL[:len(s)])))
    if suffix:
        fig.update_xaxes(ticksuffix=suffix)
    return T.apply(fig, height=320, title=title)


__all__ = [name for name in dir() if not name.startswith("_") and name not in {"go", "np", "pd", "T", "make_subplots"}]
