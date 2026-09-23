# Démonstration interactive — Portfolio Lab

Application **Streamlit** pédagogique et visuelle qui montre, en direct, comment un portefeuille
de Markowitz est estimé, optimisé, backtesté et comment il réagit au marché. Elle **réutilise**
le package `portfolio_lab` **sans le modifier** : toute la logique de recherche reste dans les
modules d'origine, la démo n'ajoute qu'une couche de présentation.

## Lancer

```bash
# depuis la racine du projet
pip install -r requirements.txt -r benchmark_demo/requirements-demo.txt
cd benchmark_demo
streamlit run Cockpit.py
```

Par défaut, la démo utilise les **vraies données yfinance** (réseau requis). En secours sans
réseau (salle de cours), cochez **Données simulées (sans internet)** dans le groupe « Données ».
Choisissez un **Préréglage** (Démo rapide / Standard / Recherche), ajustez si besoin, puis
cliquez **▶ Lancer la simulation** : rien ne se calcule avant ce clic, et le déroulement est
tracé dans le terminal.

## Architecture

```
benchmark_demo/
├── Cockpit.py              # COCKPIT (accueil : univers, hypothèses, réglages expliqués)
├── .streamlit/config.toml  # thème dark officiel (lisibilité)
├── core/                   # COUCHE SERVICE (testable sans navigateur)
│   ├── registry.py         # nom ↔ objet portfolio_lab (+ libellés humains DISPLAY_NAMES)
│   ├── services.py         # @st.cache : load_prices, load_index_nav (S&P 500), run_backtest
│   │                       #   (+ benchmark poids aléatoires), decision_table, compare_methods
│   ├── analytics.py        # rolling metrics, régimes, distributions, corrélations, interprétations
│   ├── constraints.py      # contraintes activées + diversification effective
│   └── presets.py          # scénarios historiques (replay) + univers long-historique
├── components/             # UI réutilisable
│   ├── theme.py            # thème sombre « desk institutionnel » (Plotly + CSS)
│   ├── charts.py           # figures Plotly (animation, donut, locator, mouvements…) ;
│   │                       #   overlay 1/N + S&P 500 + poids aléatoires, avec légendes
│   ├── ui.py               # timeline de décision + cartes d'insight (HTML)
│   └── sidebar.py          # config (presets + 3 groupes), bouton Run, bandeau de santé
└── pages/                  # 2 pages (+ le Cockpit = Cockpit.py)
    1_Simulation · 2_Analyse
```

**Principe** : l'UI n'appelle jamais `portfolio_lab` directement, seulement `core/services`
(qui met tout en cache). Le backtest n'est lancé **que sur clic du bouton ▶ Lancer** ; un même
résultat caché alimente les pages. Navigation instantanée et cohérente. La traçabilité du
déroulement s'affiche dans le **terminal** (logger `demo │`).

## Les 3 pages

1. **Cockpit** (accueil) — univers (**treemap secteur → actifs**, cliquable), hypothèses, et
   section **« Comprendre les réglages »** (chaque paramètre expliqué : impact + raison d'être).
2. **Simulation** — le cœur : NAV animée (lecture auto) comparée à **trois références
   identifiées par une légende** — Équipondéré 1/N, **S&P 500 (indice)** et **Poids aléatoires**
   (référence « zéro intelligence », Dirichlet) — avec rebalancements marqués, slider temporel,
   **panneau « Décision du portefeuille »** (volatilité estimée, corrélation, turnover,
   rebalancement) et **frise** Observation → Estimation μ → Estimation Σ → Optimisation →
   Rebalancement → Mise à jour. **Mode replay** : COVID, bear 2022, bull 2017.
3. **Analyse** — performance (NAV comparée 1/N · S&P 500 · poids aléatoires ; drawdown,
   vol/Sharpe glissants), distribution (skew/kurtosis),
   corrélations, **régimes** (bull/bear/forte vol) sur les graphiques, et **export d'un rapport
   Markdown** horodaté (paramètres, métriques OOS, contrôles de santé).

> Sidebar structurée : **Préréglage** (Démo rapide / Standard / Recherche) · **Univers ·
> Nombre d'actifs · Méthode · ▶ Lancer** (toujours visible) ; le reste dans 3 groupes repliables
> **📊 Données · 🧪 Modèle · ⏱️ Protocole**. Les selectbox affichent des **libellés clairs**
> (« Diversifié par secteur », « Markowitz (rendement/risque) »…). Le bandeau affiche la
> **Diversification effective** et alerte si l'espace admissible est trop contraint.
