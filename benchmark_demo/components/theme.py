"""Thème visuel « desk institutionnel » (sombre, minimaliste) pour Plotly + Streamlit.

Inspiré des plateformes type Bloomberg / Aladdin / Koyfin : fond sombre, accent cyan,
typographie sobre, grilles discrètes. Centralise palette et template Plotly pour un rendu
homogène sur toutes les pages.
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# --------------------------------------------------------------------------- #
#  Palette                                                                     #
# --------------------------------------------------------------------------- #
BG       = "#0E1117"      # fond (aligné sur le dark theme Streamlit)
PANEL    = "#1B2130"      # cartes / panneaux (contraste suffisant avec le fond)
GRID     = "#2C3342"
TEXT     = "#F2F4F8"      # texte quasi-blanc : lisibilité maximale
MUTED    = "#AEB6C7"      # texte secondaire / axes : volontairement clair (lisible)
ACCENT   = "#28C0C7"      # cyan : la stratégie
BENCH    = "#8B93A7"      # gris : le benchmark 1/N
POS      = "#22C55E"      # vert : gains
NEG      = "#EF4444"      # rouge : pertes / drawdown
WARN     = "#F59E0B"

# Séquence catégorielle (méthodes, actifs) — lisible sur fond sombre.
CATEGORICAL = ["#28C0C7", "#6366F1", "#F59E0B", "#22C55E", "#EF4444",
               "#A855F7", "#14B8A6", "#EC4899", "#84CC16", "#0EA5E9"]

# Couleurs de régime (bandes Page 5/7).
REGIME_COLORS = {"bull": "rgba(34,197,94,0.10)", "bear": "rgba(239,68,68,0.12)",
                 "crise": "rgba(245,158,11,0.16)"}


# --------------------------------------------------------------------------- #
#  Template Plotly                                                             #
# --------------------------------------------------------------------------- #
def _build_template() -> go.layout.Template:
    t = go.layout.Template()
    t.layout = go.Layout(
        paper_bgcolor=BG, plot_bgcolor=PANEL,
        font=dict(color=TEXT, family="Inter, system-ui, sans-serif", size=15),
        title=dict(font=dict(size=19, color=TEXT), x=0.02, xanchor="left"),
        colorway=CATEGORICAL,
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, color=TEXT,
                   tickfont=dict(size=13, color=MUTED), title=dict(font=dict(size=14, color=MUTED))),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, color=TEXT,
                   tickfont=dict(size=13, color=MUTED), title=dict(font=dict(size=14, color=MUTED))),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT, size=13)),
        margin=dict(l=55, r=22, t=56, b=44),
        hoverlabel=dict(bgcolor=PANEL, font=dict(color=TEXT, size=13)),
    )
    return t


pio.templates["institutional"] = _build_template()


def apply(fig: go.Figure, *, height: int | None = None, title: str | None = None) -> go.Figure:
    """Applique le template institutionnel à une figure et harmonise la mise en page."""
    fig.update_layout(template="institutional")
    if height:
        fig.update_layout(height=height)
    if title is not None:
        fig.update_layout(title=title)
    return fig


# --------------------------------------------------------------------------- #
#  CSS Streamlit (injecté une fois par page)                                   #
# --------------------------------------------------------------------------- #
CSS = """
<style>
  .stApp { background-color: #0E1117; }
  section[data-testid="stSidebar"] { background-color: #1B2130; }

  /* Texte courant : clair et un peu plus grand pour la lisibilité en soutenance. */
  .stApp, .stMarkdown, p, li, span, label, div[data-testid="stMarkdownContainer"] {
      color: #F2F4F8;
  }
  .stMarkdown p, .stMarkdown li { font-size: 1.02rem; line-height: 1.6; }

  /* Légendes (st.caption) : nettement plus lisibles qu'un gris terne. */
  div[data-testid="stCaptionContainer"], .stCaption, small { color: #B9C1D2 !important; }

  /* Cartes de métriques : contraste et grand chiffre. */
  div[data-testid="stMetric"] {
      background: #1B2130; border: 1px solid #2C3342; border-radius: 12px;
      padding: 14px 16px;
  }
  div[data-testid="stMetricLabel"] p { color: #AEB6C7 !important; font-size: .9rem; }
  div[data-testid="stMetricValue"] { color: #F2F4F8; font-size: 2rem; }

  /* Titres lumineux. */
  h1 { color: #FFFFFF; font-weight: 700; }
  h2, h3 { color: #EAF6F7; letter-spacing: .2px; }

  /* Onglets et tableaux plus lisibles. */
  button[data-baseweb="tab"] { font-size: 1rem; }
  div[data-testid="stDataFrame"] { border: 1px solid #2C3342; border-radius: 10px; }

  .demo-kicker { color:#28C0C7; font-weight:700; text-transform:uppercase;
                 letter-spacing:1.8px; font-size:12.5px; }
  .demo-card { background:#1B2130; border:1px solid #2C3342; border-radius:12px;
               padding:16px 18px; }
</style>
"""


def inject_css(st_module) -> None:
    """Injecte le CSS sombre (appeler en tête de chaque page)."""
    st_module.markdown(CSS, unsafe_allow_html=True)


__all__ = ["apply", "inject_css", "CSS", "BG", "PANEL", "GRID", "TEXT", "MUTED",
           "ACCENT", "BENCH", "POS", "NEG", "WARN", "CATEGORICAL", "REGIME_COLORS"]
