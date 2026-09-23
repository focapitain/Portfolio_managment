"""Journalisation reproductible des runs de notebooks (return / covariance / markowitz).

Objectif : à chaque exécution d'un notebook, produire dans `results/` un compte rendu
HORODATÉ (Markdown, PDF optionnel) qui consigne :
  - TOUS les paramètres réellement utilisés (période demandée ET couverte, univers demandé
    ET effectif après nettoyage, fenêtres, coûts, rf, estimateurs μ/Σ, objectif, w_max…) ;
  - les TABLES de résultats déjà calculées dans le notebook ;
  - les FIGURES (PNG embarqués) ;
  - des CONTRÔLES DE SANTÉ automatiques (dont le piège de dégénérescence N×w_max et les
    fallbacks), pour qu'un rapport soit interprétable seul, des mois plus tard.

Usage minimal (une seule cellule en fin de notebook) :
    from portfolio_lab.reporting import RunReport
    rep = RunReport("return")
    rep.capture(prices=prices, selector=SELECTOR, requested_n=20, period=(START, END),
                bt_cfg=BT_CFG, base_config=base, use_synthetic=USE_SYNTHETIC,
                mean_estimators=mean_estimators, cov_estimators={"Σ figée": COV_FIXED})
    rep.add_table("Qualité prédictive (μ)", pred[cols])
    rep.add_results(res)                      # tables de métriques d'un dict de backtests
    rep.add_figure(fig, "nav")                # optionnel
    print(rep.save())                         # -> results/return__AAAA-MM-JJ_HHMMSS.md
"""
from __future__ import annotations

import datetime as _dt
import platform as _platform
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
#  Description lisible d'un objet (estimateur, optimiseur, sélecteur…)         #
# --------------------------------------------------------------------------- #
def describe(obj: Any) -> str:
    """Renvoie une description compacte `ClasseName(param=valeur, …)` d'un objet.

    Lit les attributs publics de l'instance (`__dict__`) ; idéal pour les estimateurs
    et sélecteurs du framework, qui exposent leurs hyperparamètres en attributs simples.
    """
    if obj is None:
        return "—"
    cls = type(obj).__name__
    params = {}
    for k, v in vars(obj).items() if hasattr(obj, "__dict__") else []:
        if k.startswith("_"):
            continue
        params[k] = v
    if not params:
        return cls
    inside = ", ".join(f"{k}={_short(v)}" for k, v in params.items())
    return f"{cls}({inside})"


def _short(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _df_to_md(df: pd.DataFrame) -> str:
    """DataFrame -> Markdown (via tabulate si dispo, sinon bloc de code aligné)."""
    try:
        return df.to_markdown()
    except Exception:
        return "```\n" + df.to_string() + "\n```"


# --------------------------------------------------------------------------- #
#  Le rapport                                                                  #
# --------------------------------------------------------------------------- #
class RunReport:
    """Accumulateur de sections Markdown + paramètres, écrit dans `results/`.

    Chaque rapport est identifié par `<notebook>__<horodatage>` ; les figures vont dans
    `results/figs/`. Appeler `capture(...)` une fois (paramètres), puis `add_table` /
    `add_figure` / `add_note` au fil du notebook, enfin `save()`.
    """

    def __init__(self, notebook: str, *, results_dir: str = "results",
                 title: Optional[str] = None):
        self.notebook = notebook
        self.timestamp = _dt.datetime.now()
        stamp = self.timestamp.strftime("%Y-%m-%d_%H%M%S")
        self.run_id = f"{notebook}__{stamp}"
        self.title = title or f"Rapport de run — notebook `{notebook}`"

        self.dir = Path(results_dir)
        self.fig_dir = self.dir / "figs"
        self.dir.mkdir(parents=True, exist_ok=True)

        self._params: dict[str, Any] = {}        # libellé -> valeur (affichés en table)
        self._sections: list[str] = []           # blocs Markdown dans l'ordre d'ajout
        self._checks: list[tuple[str, bool, str]] = []  # (libellé, ok, détail)
        self._n_effective: Optional[int] = None
        self._w_max: Optional[float] = None

    # ---------------------------------------------------------------- params #
    def param(self, label: str, value: Any) -> "RunReport":
        """Ajoute un paramètre libre à la table de tête."""
        self._params[label] = value
        return self

    def capture(self, *,
                prices: Optional[pd.DataFrame] = None,
                selector: Any = None,
                requested_n: Optional[int] = None,
                period: Optional[tuple[str, str]] = None,
                bt_cfg: Any = None,
                base_config: Any = None,
                mean_estimators: Optional[dict] = None,
                cov_estimators: Optional[dict] = None,
                use_synthetic: Optional[bool] = None,
                w_max: Optional[float] = None,
                extra: Optional[dict] = None) -> "RunReport":
        """Extrait automatiquement les paramètres clés des objets du notebook."""
        if use_synthetic is not None:
            self.param("Source de données", "SYNTHÉTIQUE" if use_synthetic else "réelle (yfinance)")

        if period is not None:
            self.param("Période demandée", f"{period[0]} → {period[1]}")

        if prices is not None:
            n = int(prices.shape[1])
            self._n_effective = n
            self.param("Univers EFFECTIF (après nettoyage)", f"{n} actifs")
            self.param("Couverture réelle des prix",
                       f"{prices.index[0].date()} → {prices.index[-1].date()} "
                       f"({prices.shape[0]} observations)")
            self.param("Actifs retenus", ", ".join(map(str, prices.columns)))

        if selector is not None:
            self.param("Sélecteur d'univers", describe(selector))
            req = requested_n
            if req is None:
                req = getattr(selector, "n", getattr(selector, "n_total", None))
            if req is not None:
                self.param("Univers DEMANDÉ", f"{req} actifs (cible du sélecteur)")
                if self._n_effective is not None and self._n_effective < req:
                    self._checks.append((
                        "Univers effectif = univers demandé", False,
                        f"demandé {req}, obtenu {self._n_effective} "
                        f"(stratification arrondie et/ou actifs sans historique complet supprimés)."))

        if bt_cfg is not None:
            self.param("Fenêtre d'estimation", f"{getattr(bt_cfg, 'estimation_window', '?')} périodes")
            self.param("Période de détention (rebalancement)", f"{getattr(bt_cfg, 'holding_period', '?')} périodes")
            self.param("Fréquence", getattr(bt_cfg, "frequency", "?"))
            self.param("Coûts de transaction", f"{getattr(bt_cfg, 'transaction_cost_bps', '?')} bps")
            self.param("Taux sans risque (annuel)", f"{getattr(bt_cfg, 'risk_free_annual', 0.0):.2%}")
            self.param("Fallback", getattr(bt_cfg, "fallback", "?"))

        if base_config is not None:
            self.param("Objectif d'optimisation", getattr(base_config, "objective", "?"))
            self.param("Optimiseur", describe(getattr(base_config, "optimizer", None)))
            self.param("Long-only", getattr(base_config, "long_only", "?"))
            wmax = getattr(base_config, "w_max", None)
            self._w_max = wmax
            self.param("Plafond par actif (w_max)", f"{wmax}" if wmax is not None else "—")
            self.param("Mode de rendement", getattr(base_config, "return_mode", "?"))
            self.param("Aversion au risque (max_utility)", getattr(base_config, "risk_aversion", "?"))
            if getattr(base_config, "cov_estimator", None) is not None:
                self.param("Estimateur Σ (config de base)", describe(base_config.cov_estimator))
            if getattr(base_config, "mean_estimator", None) is not None:
                self.param("Estimateur μ (config de base)", describe(base_config.mean_estimator))

        # w_max explicite (ex. Markowitz : il vit dans la ConstraintSet, pas dans une config).
        if w_max is not None and self._w_max is None:
            self._w_max = w_max
            self.param("Plafond par actif (w_max)", f"{w_max}")

        if mean_estimators:
            self.param("Estimateurs μ comparés",
                       "; ".join(f"{k} = {describe(v)}" for k, v in mean_estimators.items()))
        if cov_estimators:
            self.param("Estimateurs Σ comparés",
                       "; ".join(f"{k} = {describe(v)}" for k, v in cov_estimators.items()))

        if extra:
            for k, v in extra.items():
                self.param(k, v)

        # Contrôle de dégénérescence automatique (N × w_max).
        if self._n_effective is not None and self._w_max is not None and self._w_max < 1.0:
            budget = self._n_effective * self._w_max
            ok = budget > 1.2
            detail = (f"N×w_max = {self._n_effective}×{self._w_max} = {budget:.2f}. "
                      + ("> 1.2 : l'optimiseur a du jeu." if ok else
                         "≤ 1.2 : espace admissible (quasi) réduit à l'équipondéré — "
                         "les objectifs dépendant de μ/Σ ne se différencient plus. "
                         "Augmenter N ou relever w_max."))
            self._checks.append(("Portefeuille non dégénéré (N×w_max > 1.2)", ok, detail))
        return self

    # ----------------------------------------------------------- contenus #
    def section(self, title: str, body_md: str = "") -> "RunReport":
        self._sections.append(f"## {title}\n\n{body_md}".rstrip() + "\n")
        return self

    def add_note(self, text_md: str) -> "RunReport":
        self._sections.append(text_md.rstrip() + "\n")
        return self

    def add_table(self, title: str, df: pd.DataFrame, *, note: str = "",
                  round_to: Optional[int] = 4) -> "RunReport":
        """Ajoute un DataFrame comme table Markdown sous un titre."""
        d = df.round(round_to) if (round_to is not None) else df
        block = f"### {title}\n\n{_df_to_md(d)}\n"
        if note:
            block += f"\n_{note}_\n"
        self._sections.append(block)
        return self

    def add_results(self, results: dict, *, columns: Optional[list[str]] = None,
                    title: str = "Métriques de backtest (OOS)") -> "RunReport":
        """Construit la table de métriques d'un dict {nom: WalkForwardBacktestResult}.

        Capte aussi automatiquement le total de fallbacks (signal de dégénérescence).
        """
        m = pd.DataFrame({name: res.metrics for name, res in results.items()}).T
        cols = [c for c in (columns or m.columns) if c in m.columns]
        self.add_table(title, m[cols] if cols else m)
        if "Fallback count" in m.columns:
            tot = float(m["Fallback count"].sum())
            ok = tot == 0
            self._checks.append((
                f"Aucun fallback — {title}", ok,
                f"{int(tot)} fallback(s) cumulés sur cette table."
                + ("" if ok else " Fenêtres baissières (tous μ < rf) ou contraintes trop serrées.")))
        return self

    def add_figure(self, fig, name: str, *, caption: str = "", dpi: int = 110) -> "RunReport":
        """Sauve une figure matplotlib en PNG dans results/figs et l'embarque."""
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{self.run_id}__{name}.png"
        path = self.fig_dir / fname
        try:
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
        except Exception as exc:  # pragma: no cover
            self.add_note(f"_(figure '{name}' non sauvegardée : {exc})_")
            return self
        rel = f"figs/{fname}"
        cap = f"\n\n*{caption}*" if caption else ""
        self._sections.append(f"![{name}]({rel}){cap}\n")
        return self

    def add_check(self, label: str, ok: bool, detail: str = "") -> "RunReport":
        self._checks.append((label, bool(ok), detail))
        return self

    def add_if(self, namespace: dict, var: str, title: str, *,
               columns: Optional[list[str]] = None) -> "RunReport":
        """Ajoute `namespace[var]` au rapport SI elle existe, en dispatchant par type.

        - DataFrame / Series  -> table ;
        - dict de résultats de backtest (objets à attribut `.metrics`) -> table de métriques.
        Silencieux si la variable est absente : la cellule de reporting reste robuste même
        si certaines sections du notebook n'ont pas été exécutées.
        """
        if var not in namespace:
            return self
        obj = namespace[var]
        try:
            if isinstance(obj, pd.Series):
                self.add_table(title, obj.to_frame())
            elif isinstance(obj, pd.DataFrame):
                cols = [c for c in (columns or obj.columns) if c in obj.columns]
                self.add_table(title, obj[cols] if cols else obj)
            elif isinstance(obj, dict) and obj and all(hasattr(v, "metrics") for v in obj.values()):
                self.add_results(obj, columns=columns, title=title)
            elif isinstance(obj, dict):
                self.add_table(title, pd.DataFrame(obj).T)
        except Exception as exc:  # pragma: no cover
            self.add_note(f"_(section '{title}' ignorée : {exc})_")
        return self

    # ------------------------------------------------------------- rendu #
    def _render_md(self) -> str:
        lines: list[str] = [f"# {self.title}", ""]
        lines.append(f"- **Run** : `{self.run_id}`")
        lines.append(f"- **Généré le** : {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"- **Python** : {_platform.python_version()} "
                     f"| numpy {np.__version__} | pandas {pd.__version__}")
        try:
            import portfolio_lab as _pl
            lines.append(f"- **portfolio_lab** : v{getattr(_pl, '__version__', '?')}")
        except Exception:
            pass
        lines.append("")

        # Paramètres
        lines.append("## Paramètres du run\n")
        if self._params:
            pdf = pd.DataFrame({"Valeur": {k: str(v) for k, v in self._params.items()}})
            pdf.index.name = "Paramètre"
            lines.append(_df_to_md(pdf))
        else:
            lines.append("_Aucun paramètre capturé._")
        lines.append("")

        # Contrôles de santé
        lines.append("## Contrôles de santé\n")
        if self._checks:
            rows = [{"Contrôle": lbl, "Statut": "✅ OK" if ok else "⚠️ ATTENTION",
                     "Détail": det} for lbl, ok, det in self._checks]
            lines.append(_df_to_md(pd.DataFrame(rows).set_index("Contrôle")))
        else:
            lines.append("_Aucun contrôle enregistré._")
        lines.append("")

        # Sections de contenu
        for block in self._sections:
            lines.append(block)
        return "\n".join(lines).rstrip() + "\n"

    def to_markdown(self) -> str:
        """Renvoie le rapport en Markdown SANS écrire de fichier.

        Utile pour un téléchargement direct (ex. bouton Streamlit) ou un test : on récupère
        la chaîne sans toucher au disque.
        """
        return self._render_md()

    def save(self, *, fmt: str = "md") -> str:
        """Écrit le rapport. `fmt` : 'md' (défaut), 'pdf' ou 'both'.

        Le PDF est tenté via `pandoc` s'il est installé ; sinon on retombe proprement sur
        le Markdown (toujours écrit) en le signalant.
        """
        md = self._render_md()
        md_path = self.dir / f"{self.run_id}.md"
        md_path.write_text(md, encoding="utf-8")
        out = str(md_path)

        if fmt in ("pdf", "both"):
            pdf_path = self.dir / f"{self.run_id}.pdf"
            if _to_pdf(md_path, pdf_path):
                out = str(pdf_path) if fmt == "pdf" else f"{md_path} ; {pdf_path}"
            else:
                print("[reporting] PDF indisponible (pandoc absent) — Markdown écrit à la place.")
        return out


def _to_pdf(md_path: Path, pdf_path: Path) -> bool:
    """Tente une conversion Markdown -> PDF via pandoc. Renvoie True si réussi."""
    import shutil
    import subprocess
    if shutil.which("pandoc") is None:
        return False
    try:
        subprocess.run(["pandoc", str(md_path), "-o", str(pdf_path)],
                       check=True, capture_output=True)
        return pdf_path.exists()
    except Exception:
        return False


__all__ = ["RunReport", "describe"]
