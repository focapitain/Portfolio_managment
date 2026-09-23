"""Composants d'interface HTML (hors Plotly) : timeline de décision, cartes d'insight.

Rendus via `st.markdown(..., unsafe_allow_html=True)`. Styles INLINE (robustes quel que soit
l'ordre d'injection du CSS), alignés sur le thème sombre institutionnel.
"""
from __future__ import annotations

from . import theme as T


def decision_timeline(steps: list[dict]) -> str:
    """Frise horizontale du PROCESSUS DE DÉCISION.

    `steps` : liste de {icon, label, value?} affichés en cartes reliées par des flèches.
    Le pipeline canonique : Observation → Estimation μ → Estimation Σ → Optimisation →
    Rebalancement → Mise à jour du portefeuille.
    """
    cards = []
    for s in steps:
        val = s.get("value", "")
        val_html = (f"<div style='color:{T.ACCENT};font-weight:700;font-size:12.5px;"
                    f"margin-top:3px'>{val}</div>") if val else ""
        cards.append(
            f"<div style='flex:1;min-width:118px;background:{T.PANEL};border:1px solid {T.GRID};"
            f"border-radius:10px;padding:10px 8px;text-align:center'>"
            f"<div style='font-size:20px;line-height:1'>{s['icon']}</div>"
            f"<div style='color:{T.TEXT};font-weight:600;font-size:12px;margin-top:5px'>{s['label']}</div>"
            f"{val_html}</div>"
        )
    arrow = (f"<div style='align-self:center;color:{T.ACCENT};font-size:17px;"
             f"padding:0 2px'>→</div>")
    row = arrow.join(cards)
    return (f"<div style='display:flex;align-items:stretch;gap:2px;overflow-x:auto;"
            f"padding:4px 0 8px 0'>{row}</div>")


def render_insights(st_module, insights: list[dict]) -> None:
    """Affiche les interprétations automatiques (couleur selon le niveau)."""
    icon = {"good": "✅", "warn": "⚠️", "info": "💡"}
    for ins in insights:
        lvl = ins.get("level", "info")
        txt = f"{icon.get(lvl, '•')} {ins['text']}"
        (st_module.success if lvl == "good" else
         st_module.warning if lvl == "warn" else st_module.info)(txt)


__all__ = ["decision_timeline", "render_insights"]
