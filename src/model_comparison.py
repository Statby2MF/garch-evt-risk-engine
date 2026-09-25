"""
Comparaison finale des modèles de VaR.

Ce module agrège les résultats de :
    - benchmark_models  (4 VaR concurrentes)
    - dynamic_var       (VaR GARCH-EVT)
    - backtesting       (Kupiec, Christoffersen, DQ)

Il produit :
    - Un DataFrame récapitulatif (tableau final)
    - Un graphique comparatif de toutes les VaR vs rendements
    - Un graphique des violations
    - Un graphique du score Bâle III

Auteur : Statby2Mf
Projet : GARCH-EVT Risk Engine (M2 Statistique, UGB Saint-Louis)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtesting import backtest_table
from src.benchmark_models import (
    var_historical,
    var_normal,
    var_garch_normal,
)
from src.dynamic_var import rolling_var_es


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Pipeline de comparaison
# ---------------------------------------------------------------------------

def compute_all_models(
    returns: pd.Series,
    p: float = 0.99,
    window: int = 1000,
    refit_every: int = 50,
    verbose: bool = True,
) -> dict[str, pd.Series]:
    """
    Calcule la VaR de tous les modèles (GARCH-EVT + 3 concurrents).

    Parameters
    ----------
    returns : pd.Series
        Log-rendements en %.
    p : float
        Niveau de confiance.
    window : int
        Fenêtre glissante.
    refit_every : int
        Fréquence de refit.
    verbose : bool

    Returns
    -------
    dict of {model_name: var_series}
    """
    models = {}

    if verbose:
        print("📊 Calcul de la VaR Historique…")
    models["Historique"] = var_historical(returns, window=window, p=p)

    if verbose:
        print("📊 Calcul de la VaR Normale…")
    models["Normale"] = var_normal(returns, window=window, p=p)

    if verbose:
        print("📊 Calcul de la VaR GARCH-Normal…")
    models["GARCH-Normal"] = var_garch_normal(
        returns, window=window, p=p, refit_every=refit_every
    )

    if verbose:
        print("📊 Calcul de la VaR GARCH-EVT (le plus long)…")
    result_evt = rolling_var_es(
        returns,
        window=window,
        p=p,
        refit_every=refit_every,
        model_type="garch",
        dist="ged",
        threshold_quantile=0.90,
        verbose=False,
    )
    models["GARCH-EVT"] = result_evt.var

    return models


def build_comparison_table(
    returns: pd.Series,
    models: dict[str, pd.Series],
    p: float = 0.99,
) -> pd.DataFrame:
    """
    Construit le tableau comparatif final avec :
        - Statistiques de VaR (moyenne, min, max)
        - Backtest (violations, taux)
        - Tests Bâle III (Kupiec, IND, CC)
        - Score composite

    Returns
    -------
    pd.DataFrame
    """
    table = backtest_table(returns, models, p=p)

    # Ajout des stats descriptives VaR
    stats = []
    for name, var in models.items():
        v = var.dropna()
        stats.append({
            "model": name,
            "var_mean": v.mean(),
            "var_min": v.min(),
            "var_max": v.max(),
        })
    stats_df = pd.DataFrame(stats)

    table = table.merge(stats_df, on="model", how="left")

    # Score Bâle III
    table["score_bale"] = table["kupiec_p"] + table["cc_p"]

    # Nombre de tests Bâle III passés
    table["n_tests_passed"] = (
        (table["kupiec_p"] > 0.05).astype(int) +
        (table["ind_p"] > 0.05).astype(int) +
        (table["cc_p"] > 0.05).astype(int)
    )

    # Tri par score décroissant
    table = table.sort_values("score_bale", ascending=False).reset_index(drop=True)
    table.insert(0, "rank", range(1, len(table) + 1))

    return table


# ---------------------------------------------------------------------------
# Graphiques
# ---------------------------------------------------------------------------

def plot_var_comparison(
    returns: pd.Series,
    models: dict[str, pd.Series],
    save_path: Path | None = None,
    figsize: tuple = (14, 7),
) -> plt.Figure:
    """
    Graphique : rendements + toutes les VaR superposées.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Rendements en arrière-plan
    ax.plot(returns.index, returns.values, color="lightgray",
            linewidth=0.5, label="Rendements", alpha=0.7)

    # VaR des modèles
    colors = {
        "Historique": "#1f77b4",
        "Normale": "#ff7f0e",
        "GARCH-Normal": "#2ca02c",
        "GARCH-EVT": "#d62728",
    }
    for name, var in models.items():
        color = colors.get(name, None)
        lw = 1.5 if name == "GARCH-EVT" else 1.0
        ax.plot(var.index, var.values, color=color, linewidth=lw, label=name)

    ax.set_title("Comparaison des VaR — p = 99%", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rendement / VaR (%)")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"💾 Graphique sauvegardé : {save_path}")

    return fig


def plot_violations(
    returns: pd.Series,
    models: dict[str, pd.Series],
    save_path: Path | None = None,
    figsize: tuple = (14, 10),
) -> plt.Figure:
    """
    Graphique en grille : violations de chaque modèle.
    """
    n_models = len(models)
    fig, axes = plt.subplots(n_models, 1, figsize=figsize, sharex=True)
    if n_models == 1:
        axes = [axes]

    for ax, (name, var) in zip(axes, models.items()):
        mask = var.notna()
        r = returns[mask]
        v = var[mask]
        violations = r < v

        # Rendements
        ax.plot(r.index, r.values, color="lightgray", linewidth=0.5)

        # VaR
        ax.plot(v.index, v.values, color="blue", linewidth=1.0, label="VaR")

        # Violations en rouge
        viol_dates = r[violations].index
        viol_values = r[violations].values
        ax.scatter(viol_dates, viol_values, color="red", s=30, zorder=5,
                   label=f"Violations ({violations.sum()})")

        ax.set_title(f"{name} — {violations.sum()} violations "
                     f"({violations.sum()/len(r):.2%})",
                     fontsize=11, fontweight="bold")
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Date")
    fig.suptitle("Violations de VaR par modèle (p = 99%)",
                 fontsize=14, fontweight="bold", y=1.00)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"💾 Graphique sauvegardé : {save_path}")

    return fig


def plot_scores(
    table: pd.DataFrame,
    save_path: Path | None = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Bar chart : score Bâle III par modèle.
    """
    fig, ax = plt.subplots(figsize=figsize)

    models = table["model"].values
    scores = table["score_bale"].values

    # Couleur spéciale pour GARCH-EVT
    colors = ["#d62728" if m == "GARCH-EVT" else "#4c72b0" for m in models]

    bars = ax.barh(models, scores, color=colors, edgecolor="black", linewidth=0.5)

    # Annotations
    for bar, score in zip(bars, scores):
        width = bar.get_width()
        ax.text(width + 0.03, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Score Bâle III (Kupiec + Christoffersen CC)", fontsize=11)
    ax.set_title("Classement des modèles — Score Bâle III",
                 fontsize=13, fontweight="bold")
    ax.axvline(x=0.10, color="gray", linestyle="--", alpha=0.5,
               label="Seuil minimal (α = 0.05 × 2 tests)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"💾 Graphique sauvegardé : {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Sauvegarde
# ---------------------------------------------------------------------------

def save_results(
    table: pd.DataFrame,
    models: dict[str, pd.Series],
    output_dir: Path = DATA_DIR,
) -> None:
    """Sauvegarde le tableau et les VaR en CSV."""
    table_path = output_dir / "model_comparison.csv"
    table.to_csv(table_path, index=False)
    print(f"💾 Tableau sauvegardé : {table_path}")

    var_df = pd.DataFrame(models)
    var_path = output_dir / "var_series.csv"
    var_df.to_csv(var_path)
    print(f"💾 VaR séries sauvegardées : {var_path}")


# ---------------------------------------------------------------------------
# Test rapide
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.data_manager import get_returns

    print("🔍 Module de comparaison finale des modèles\n")
    print("=" * 80)

    btc = get_returns("BTC-USD")
    print(f"📊 BTC : {len(btc)} observations\n")

    p = 0.99
    window = 1000
    refit_every = 50

    # 1. Calcul de toutes les VaR
    print("=" * 80)
    print("ÉTAPE 1 — Calcul des VaR (peut prendre 3-4 min)")
    print("=" * 80)
    models = compute_all_models(
        btc, p=p, window=window, refit_every=refit_every, verbose=True
    )

    # 2. Tableau comparatif
    print("\n" + "=" * 80)
    print("ÉTAPE 2 — Tableau comparatif final")
    print("=" * 80)
    table = build_comparison_table(btc, models, p=p)

    # Affichage compact
    display_cols = [
        "rank", "model", "n_viol", "rate",
        "kupiec_p", "ind_p", "cc_p",
        "score_bale", "n_tests_passed",
    ]
    print("\n" + table[display_cols].to_string(index=False))

    # 3. Graphiques
    print("\n" + "=" * 80)
    print("ÉTAPE 3 — Génération des graphiques")
    print("=" * 80)

    plot_var_comparison(btc, models,
                        save_path=REPORTS_DIR / "var_comparison.png")
    plot_violations(btc, models,
                    save_path=REPORTS_DIR / "violations.png")
    plot_scores(table,
                save_path=REPORTS_DIR / "scores_bale.png")

    # 4. Sauvegarde CSV
    print("\n" + "=" * 80)
    print("ÉTAPE 4 — Sauvegarde des résultats")
    print("=" * 80)
    save_results(table, models)

    # 5. Synthèse
    print("\n" + "=" * 80)
    print("🏆 SYNTHÈSE FINALE")
    print("=" * 80)
    best = table.iloc[0]
    print(f"\n🥇 MEILLEUR MODÈLE : {best['model']}")
    print(f"   Taux de violations  : {best['rate']:.4%} (cible : {1-p:.2%})")
    print(f"   Score Bâle III      : {best['score_bale']:.3f}")
    print(f"   Tests Bâle III      : {best['n_tests_passed']}/3 passés")

    if best["model"] == "GARCH-EVT":
        print("\n   ✅ GARCH-EVT confirme sa supériorité empirique !")

    print("\n" + "=" * 80)
    print("✅ Comparaison terminée. Résultats dans reports/ et data/")
    print("=" * 80)
