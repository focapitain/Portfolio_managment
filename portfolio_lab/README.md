# portfolio_lab

Plateforme **modulaire** de recherche en optimisation de portefeuille, construite sur la
théorie moderne (Markowitz) et conçue pour le **benchmarking reproductible hors échantillon**
sur l'univers S&P 500.

Le projet sépare proprement trois préoccupations : (1) **estimer** les ingrédients de Markowitz
(rendement espéré μ, covariance Σ), (2) **optimiser** un portefeuille sous contraintes, (3)
**évaluer** le tout *hors échantillon* (walk-forward) avec des métriques honnêtes. Chaque famille
de choix suit le **patron Strategy** : ajouter une méthode = ajouter une classe.

---

## Installation

```bash
pip install -r requirements.txt
# depuis la racine du projet :
python -c "import portfolio_lab as pl; print(pl.__version__)"   # -> 2.0.0
```

Les notebooks ajoutent eux-mêmes la racine au `sys.path` ; lancez `jupyter lab` depuis la
racine **ou** depuis `notebooks/` (le chemin est géré dans la première cellule).

---

## Architecture

```
portfolio_lab/
├── portfolio_lab/                # le package importable
│   ├── __init__.py               # API publique (tout est ré-exporté ici)
│   ├── config.py                 # ExperimentConfig : tous les leviers en un dataclass
│   ├── data.py                   # constituants S&P 500, download yfinance, nettoyage,
│   │                             #   + make_synthetic_prices (mode hors-ligne / CI)
│   ├── frequencies.py            # daily/weekly/monthly : resampling & annualisation
│   ├── returns.py                # rendements simples / log
│   ├── universe.py               # SÉLECTEURS d'univers (Strategy) :
│   │                             #   AllUniverse, TopN (=FirstNAlphabetical), RandomN,
│   │                             #   StratifiedBySector
│   ├── estimators.py             # ESTIMATEURS (Strategy) :
│   │                             #   μ : SampleMean, JamesSteinMean, EWMAMean,
│   │                             #       RollingMean, PCAFactorMean
│   │                             #   Σ : Empirical, Rolling, EWMA, IEWMA, LedoitWolf,
│   │                             #       OAS, ConstantCorrelation
│   ├── optimizers.py             # OPTIMISEURS (Strategy) :
│   │                             #   AnalyticLagrangian, CVXPY, Scipy(SLSQP),
│   │                             #   DifferentialEvolution + baselines de risque
│   │                             #   (EqualWeight, RandomWeight, InverseVolatility,
│   │                             #    RiskParity, MaxDiversification)
│   ├── markowitz.py              # Benchmark INSTITUTIONNEL : ConstraintSet/PenaltySet,
│   │                             #   MarkowitzPlusPlus (QP/SOCP unifié), make_variant,
│   │                             #   harnais walk-forward dédié, calibration, sensibilité
│   ├── frontier.py               # frontière efficiente (target_return sur une grille)
│   ├── experiment.py             # run_experiment(config) -> ExperimentResult (IN-sample)
│   ├── backtest.py               # backtest walk-forward OOS (Sharpe net de rf, coûts
│   │                             #   symétriques, fallback compté, anti look-ahead)
│   ├── backtest_adapter.py       # PONT ExperimentConfig → backtest.walk_forward_backtest
│   ├── benchmarking.py           # moteur partagé des notebooks (run_oos_benchmark, tables)
│   ├── cov_eval.py               # évaluation STATISTIQUE de Σ (loglik, RMSE, conditionnement,
│   │                             #   stabilité, régime crise/hors-crise) — hors échantillon
│   ├── mean_eval.py              # évaluation PRÉDICTIVE de μ (IC de Spearman, hit-rate…)
│   ├── metrics.py                # Sortino, Calmar, VaR, CVaR, tail_metrics
│   ├── plotting.py               # visualisations in-sample
│   └── reporting.py              # RunReport : journalise chaque run dans results/ (Markdown
│                                 #   horodaté + paramètres + contrôles de santé)
├── notebooks/
│   ├── covariance.ipynb          # benchmark des estimateurs de Σ (2 axes : statistique + portef.)
│   ├── markowitz.ipynb           # 5 formulations de Markowitz (standard→++), calibration, régimes
│   ├── return.ipynb              # benchmark des estimateurs de μ (IC, robustesse, impact portef.)
│   └── simulation.py             # SIMULATION à choc injecté : quel estimateur de Σ détecte
│                                 #   le choc de volatilité le plus vite (délai de détection)
├── benchmark_demo/               # démo Streamlit (overlay 1/N + S&P 500 + poids aléatoires)
├── test_pipeline.py              # tests de bout en bout (hors-ligne, synthétique)
├── requirements.txt
└── README.md                     # (projet) ce package est documenté ici
```

### Principe de conception

`ExperimentConfig` centralise tous les leviers : **une expérience = un objet**.

**Rigueur expérimentale.** Tous les benchmarks partent d'**un** `base_config` et de **données
uniques**, et ne font varier **qu'une dimension à la fois** via `dataclasses.replace`. Tout le
reste (univers, fréquence, fenêtres, coûts) est donc contrôlé par construction.

**Conventions d'unités.** Les estimateurs renvoient des grandeurs **par période** ;
l'annualisation (×`periods_per_year`, hypothèse i.i.d.) est faite **une seule fois**, juste avant
l'optimiseur (`backtest_adapter` / `experiment`). μ, Σ, rf entrent **annualisés** dans `solve()`.

---

## Workflow de recherche

```
constituants S&P 500  ──►  univers (Strategy)  ──►  prix ajustés (yfinance)  ──►  rendements
        │                                                                            │
        │                                                       μ̂ = mean_estimator.estimate
        │                                                       Σ̂ = cov_estimator.estimate
        │                                                                            │
        │                                              annualisation (×ppy, i.i.d.)  │
        ▼                                                                            ▼
   évaluation                                                         optimiseur.solve(μ,Σ,…)
   ├─ μ : mean_eval (IC, hit-rate)                                            │
   └─ Σ : cov_eval (loglik, conditionnement)                                 ▼
                                                          backtest walk-forward OOS (coûts, rf)
                                                                              │
                                                                              ▼
                                              métriques (Sharpe net, Sortino, MaxDD, turnover…)
```

### Backtest hors échantillon (anti look-ahead)

`backtest.py` découpe l'historique en fenêtres glissantes (estimation `estimation_window` →
détention `holding_period`), ré-optimise à chaque pas, applique des coûts de transaction sur le
turnover, et compare à un benchmark 1/N **qui paie aussi ses coûts**.

**Points de correction (audit) intégrés :**
- **Sharpe net du taux sans risque** : `Sharpe = (CAGR − r_f) / vol`.
- **Coûts symétriques** : la stratégie *et* le benchmark 1/N sont taxés sur leur turnover.
- **Fallback visible** : un `max_sharpe` infaisable (aucun excès de rendement positif sur la
  fenêtre) bascule en equal-weight **compté** (`Fallback count`), jamais silencieux.
- **Anti look-ahead** : la sélection des actifs disponibles se fait **par fenêtre** (pas de
  `dropna` global qui présupposerait les survivants).

> **Limite résiduelle assumée — biais de survivance.** L'univers vient de la composition
> **actuelle** du S&P 500 (`get_sp500_constituents`). Une correction *point-in-time* nécessite des
> données externes (CRSP/Norgate) et reste **hors périmètre**. De plus, `TopN` échantillonne par
> **ordre alphabétique du ticker** (PAS par capitalisation — voir l'alias `FirstNAlphabetical`).

```python
import portfolio_lab as pl
from portfolio_lab.backtest import WalkForwardBacktestConfig

cfg = pl.ExperimentConfig(
    universe=pl.StratifiedBySector(60),
    cov_estimator=pl.LedoitWolfCovariance(),
    objective="max_sharpe", optimizer=pl.CVXPYOptimizer(),
    w_max=0.10, risk_free_annual=0.04,
)
prices = pl.clean_prices(pl.download_prices(
    cfg.universe.select(pl.get_sp500_constituents()), "2012-01-01", "2025-12-31"))
res = pl.run_walk_forward(prices, cfg,
        backtest_config=WalkForwardBacktestConfig(estimation_window=252, holding_period=21,
                                                  transaction_cost_bps=10.0, risk_free_annual=0.04))
print(res.metrics)
```

---

## Les 3 notebooks de benchmark

Chaque notebook suit la même trame : **question → hypothèse → variables contrôlées → backtest OOS
→ tables → visualisations → interprétation → exemple concret (100 000 €) → décision**. Une cellule
finale écrit un **rapport horodaté** dans `results/` (cf. *Journalisation*).

| Notebook | Dimension étudiée | Reste figé sur |
|----------|-------------------|----------------|
| `covariance.ipynb` | estimateur de **Σ** (Empirical, Rolling, EWMA, IEWMA, Ledoit-Wolf, OAS, Constant-Corr) — axes *statistique* (`cov_eval`) **et** *portefeuille* | μ = SampleMean ; objectif `min_variance` puis `max_sharpe` |
| `return.ipynb` | estimateur de **μ** (SampleMean, RollingMean, EWMAMean, JamesStein, PCAFactor) — axes *prédictif* (`mean_eval`, IC) **et** *portefeuille* | Σ = Ledoit-Wolf ; objectif `max_sharpe` |
| `markowitz.ipynb` | **formulation** (Standard, Linéaire, Opérationnelle, Robuste, Markowitz++) + calibration + analyse par régime | μ, Σ fixés ; univers fixé |

> En complément, **`simulation.py`** (script, hors-ligne) génère des rendements synthétiques à
> **choc de volatilité injecté** (date connue) et mesure le **délai de détection** de chaque
> estimateur de Σ : EWMA/IEWMA réagissent nettement plus vite que l'empirique ou Ledoit-Wolf.

> ⚠️ **Garde-fou de dégénérescence.** Avec un plafond `w_max`, l'espace admissible
> `{w : Σw=1, 0≤w≤w_max}` se réduit à l'équipondéré dès que **N × w_max ≈ 1** : tous les objectifs
> donnent alors le même portefeuille. Visez **N × w_max ≥ 2** (idéalement ≥ 3). `RunReport` le
> vérifie automatiquement.

### Mode hors-ligne

Chaque notebook expose `USE_SYNTHETIC` en tête. `True` génère des prix synthétiques (GBM corrélé
par secteur) pour tourner **sans réseau** (CI / démo) ; `False` (défaut) télécharge yfinance.

---

## Journalisation des runs (`results/`)

À chaque exécution complète d'un notebook, la cellule finale crée un rapport Markdown horodaté
dans `results/` via `portfolio_lab.reporting.RunReport` :

- **paramètres réellement utilisés** (période demandée *et* couverte, univers demandé *et*
  effectif, fenêtres, coûts, rf, estimateurs μ/Σ, objectif, w_max) ;
- **tables de résultats** déjà calculées + figures optionnelles ;
- **contrôles de santé automatiques** : univers effectif < demandé, **dégénérescence N×w_max**,
  fallbacks `max_sharpe`.

```python
from portfolio_lab.reporting import RunReport
rep = RunReport("return", results_dir="results")
rep.capture(prices=prices, selector=SELECTOR, period=(START, END),
            bt_cfg=BT_CFG, base_config=base, mean_estimators=mean_estimators)
rep.add_results(res, columns=cols_perf)
print(rep.save())                 # -> results/return__AAAA-MM-JJ_HHMMSS.md
```

`results/` est **non versionné** (cf. `.gitignore`) : ce sont des artefacts régénérables.

---

## Tests

```bash
python test_pipeline.py    # hors-ligne (données synthétiques)
```

Valide : estimateurs PSD, convergence des optimiseurs, adaptateur backtest, moteur de
benchmarking, split passé/futur, nouveaux estimateurs (EWMA-Mean, IEWMA, RollingMean, PCAFactor),
modules `cov_eval` / `mean_eval`, et les 5 formulations de `markowitz.py`.

---


`portfolio_lab` — version 2.0.0.
