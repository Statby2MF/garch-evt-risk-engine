"""
Script de téléchargement des données de marché — VERSION PRO.

Ce script télécharge les cryptos UNE PAR UNE (au lieu de tout d'un coup),
ce qui permet de conserver l'historique complet de chaque actif même si
certains n'existent que depuis une date récente (ex: SOL depuis avril 2020).

Pipeline :
    1. Téléchargement individuel par ticker
    2. Concaténation sur l'union des dates
    3. Nettoyage (suppression jours fériés, forward-fill sur weekends)
    4. Suppression de la période pré-listing de chaque actif
    5. Sauvegarde en CSV
    6. Rapport de qualité détaillé

Usage :
    python download_data.py
    python download_data.py --force     # retélécharge même si CSV existe

Auteur : Statby2Mf
Projet : GARCH-EVT Risk Engine (M2 Statistique, UGB Saint-Louis)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

PRICES_FILE = DATA_DIR / "prices.csv"

#: Univers crypto analysé.
CRYPTOS = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "BNB-USD": "Binance Coin",
    "SOL-USD": "Solana",
    "XRP-USD": "Ripple",
}

#: Date de début souhaitée.
#: Chaque actif démarrera à sa propre date de première cotation si plus tardive.
START_DATE = "2018-01-01"


# ---------------------------------------------------------------------------
# Téléchargement individuel
# ---------------------------------------------------------------------------

def download_single_ticker(
    ticker: str,
    start: str = START_DATE,
    end: str | None = None,
) -> pd.Series:
    """
    Télécharge un seul ticker et renvoie sa série de prix.

    Parameters
    ----------
    ticker : str
        Ticker Yahoo Finance (ex: "BTC-USD").
    start : str
        Date de début souhaitée.
    end : str or None
        Date de fin. None = aujourd'hui.

    Returns
    -------
    pd.Series
        Série nommée d'après le ticker, indexée par dates.
    """
    print(f"   → {ticker}… ", end="", flush=True)

    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        raise RuntimeError(f"Aucune donnée renvoyée pour {ticker}")

    # yfinance peut renvoyer un DataFrame multi-index → on force "Close"
    if isinstance(df.columns, pd.MultiIndex):
        prices = df["Close"][ticker]
    else:
        prices = df["Close"]

    prices = prices.dropna().sort_index()
    prices.name = ticker

    print(f"{len(prices)} obs ({prices.index[0].date()} → {prices.index[-1].date()})")
    return prices


def download_all_tickers(
    tickers: list[str],
    start: str = START_DATE,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Télécharge chaque ticker individuellement puis concatène.

    Returns
    -------
    pd.DataFrame
        Prix de clôture, index = union des dates, colonnes = tickers.
    """
    print(f"📥 Téléchargement individuel de {len(tickers)} actifs…")

    series_list = [download_single_ticker(t, start, end) for t in tickers]

    # Concaténation sur l'union des dates (outer join)
    prices = pd.concat(series_list, axis=1, join="outer")
    prices = prices.sort_index()
    return prices


# ---------------------------------------------------------------------------
# Nettoyage intelligent
# ---------------------------------------------------------------------------

def clean_prices(prices: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Nettoyage pro, respectant les dates de listing de chaque actif.

    Étapes :
        1. Forward-fill limité aux weekends / jours fériés (max 3 jours)
        2. Suppression des dates ANTÉRIEURES au premier prix de chaque actif
           (important : ne pas ffill avant le premier prix !)
        3. Suppression des dates postérieures au dernier prix (si actif mort)

    Returns
    -------
    (prices_clean, stats)
    """
    stats = {}
    n_before = int(prices.isna().sum().sum())

    # Étape 1 : forward-fill limité (max 3 jours = weekend + 1)
    # ⚠️ On NE forward-fill PAS avant le premier prix (sinon on invente des données)
    prices = prices.ffill(limit=3)

    # Étape 2 : pour chaque actif, supprime les dates avant son premier prix
    cleaned_cols = []
    for col in prices.columns:
        s = prices[col].dropna()
        if len(s) > 0:
            # On garde seulement les dates >= premier prix de l'actif
            mask = prices.index >= s.index[0]
            cleaned_cols.append(prices.loc[mask, col])
    prices = pd.concat(cleaned_cols, axis=1)

    # Étape 3 : garde uniquement les dates où AU MOINS un actif a un prix
    # (pour ne pas avoir de lignes complètement vides)
    prices = prices.dropna(how="all")

    # Étape 4 : on garde les dates communes où TOUS les actifs existent
    # ⚠️ CHOIX : à partir d'ici, on commence à la date où le dernier actif apparaît
    #            → garantit un dataset "rectangulaire" sans NaN
    first_valid = prices.apply(lambda s: s.first_valid_index())
    last_valid = prices.apply(lambda s: s.last_valid_index())
    common_start = max(first_valid)
    common_end = min(last_valid)
    prices = prices.loc[common_start:common_end]

    # Étape 5 : supprime les NaN résiduels (rares)
    prices = prices.dropna()

    n_after = int(prices.isna().sum().sum())

    stats = {
        "n_before_nan": n_before,
        "n_after_nan": n_after,
        "first_valid_index": first_valid.to_dict(),
        "last_valid_index": last_valid.to_dict(),
        "common_start": common_start,
        "common_end": common_end,
    }
    return prices, stats


# ---------------------------------------------------------------------------
# Rapport de qualité
# ---------------------------------------------------------------------------

def print_quality_report(prices: pd.DataFrame, stats: dict) -> None:
    """Affiche un rapport de qualité détaillé."""
    print("\n" + "=" * 70)
    print("📊  RAPPORT DE QUALITÉ DES DONNÉES")
    print("=" * 70)
    print(f"Période commune  : {prices.index[0].date()} → {prices.index[-1].date()}")
    print(f"Observations     : {len(prices)} jours")
    print(f"Actifs           : {prices.shape[1]}")
    print(f"NaN avant        : {stats['n_before_nan']}")
    print(f"NaN après        : {stats['n_after_nan']}")
    print("-" * 70)
    print("📅 Historique par actif (avant alignement) :")
    for col in stats["first_valid_index"]:
        start = stats["first_valid_index"][col]
        end = stats["last_valid_index"][col]
        print(f"   {col:10s} : {start.date()} → {end.date()}")
    print("-" * 70)
    print("💰 Statistiques par actif (prix) :")
    print(prices.describe().round(2).to_string())
    print("=" * 70)


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Téléchargement des données crypto")
    parser.add_argument("--force", action="store_true",
                        help="Retélécharge même si le CSV existe")
    args = parser.parse_args()

    if PRICES_FILE.exists() and not args.force:
        print(f"ℹ️  {PRICES_FILE} existe déjà. Utilise --force pour retélécharger.")
        return

    # 1. Téléchargement individuel
    prices_raw = download_all_tickers(list(CRYPTOS.keys()))

    # 2. Nettoyage
    prices, stats = clean_prices(prices_raw)

    # 3. Sauvegarde
    prices.to_csv(PRICES_FILE)
    print(f"\n💾 Sauvegardé : {PRICES_FILE}")

    # 4. Rapport
    print_quality_report(prices, stats)


if __name__ == "__main__":
    main()
