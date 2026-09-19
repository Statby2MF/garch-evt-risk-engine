"""
Script de téléchargement des données de marché.

Ce script télécharge les prix de clôture ajustés des cryptomonnaies
via l'API Yahoo Finance (yfinance) et les sauvegarde dans data/prices.csv.

Usage :
    python download_data.py

Auteur : Statby2Mf
Projet : GARCH-EVT Risk Engine (M2 Statistique, UGB Saint-Louis)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Dossier racine du projet (le fichier est à la racine)
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

PRICES_FILE = DATA_DIR / "prices.csv"

#: Univers crypto analysé.
#: Clé = ticker Yahoo Finance, valeur = nom lisible (pour les graphiques).
CRYPTOS = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "BNB-USD": "Binance Coin",
    "SOL-USD": "Solana",
    "XRP-USD": "Ripple",
}

#: Date de début de l'historique.
START_DATE = "2018-01-01"


# ---------------------------------------------------------------------------
# Fonctions
# ---------------------------------------------------------------------------

def download_prices(
    tickers: list[str],
    start: str = START_DATE,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Télécharge les prix de clôture ajustés des tickers donnés.

    Parameters
    ----------
    tickers : list of str
        Liste de tickers Yahoo Finance (ex: ["BTC-USD", "ETH-USD"]).
    start : str
        Date de début (format "YYYY-MM-DD").
    end : str or None
        Date de fin. None = aujourd'hui.

    Returns
    -------
    pd.DataFrame
        Prix de clôture — index = dates, colonnes = tickers.
    """
    print(f"📥 Téléchargement de {len(tickers)} actifs depuis {start}…")
    print(f"   Tickers : {', '.join(tickers)}")

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,   # ajuste splits & dividendes
        progress=False,
        group_by="column",
    )["Close"]

    # Cas particulier : un seul ticker → yfinance renvoie une Series
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name=tickers[0])

    raw = raw.sort_index()
    print(f"✅ Téléchargé : {raw.shape[0]} jours × {raw.shape[1]} actifs")
    return raw


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les prix : supprime jours entièrement vides + forward-fill.

    Parameters
    ----------
    prices : pd.DataFrame
        Prix bruts issus de yfinance.

    Returns
    -------
    pd.DataFrame
        Prix nettoyés.
    """
    n_before = int(prices.isna().sum().sum())

    # Supprime les jours où TOUT est NaN (jours non cotés)
    prices = prices.dropna(how="all")
    # Forward-fill (comble les trous ponctuels) puis supprime les NaN restants
    prices = prices.ffill().dropna()

    n_after = int(prices.isna().sum().sum())

    print(f"🧹 Nettoyage : {n_before} NaN → {n_after} NaN")
    return prices


def quality_report(prices: pd.DataFrame) -> None:
    """
    Affiche un rapport de qualité des données à l'écran.

    Parameters
    ----------
    prices : pd.DataFrame
        Prix nettoyés.
    """
    print("\n" + "=" * 60)
    print("📊  RAPPORT DE QUALITÉ DES DONNÉES")
    print("=" * 60)
    print(f"Période          : {prices.index[0].date()} → {prices.index[-1].date()}")
    print(f"Observations     : {len(prices)} jours")
    print(f"Actifs           : {prices.shape[1]}")
    print(f"NaN résiduels    : {int(prices.isna().sum().sum())}")
    print("-" * 60)
    print("Statistiques par actif (prix de clôture) :")
    print(prices.describe().round(2).to_string())
    print("=" * 60)


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main() -> None:
    """Pipeline complet : téléchargement → nettoyage → sauvegarde → rapport."""
    tickers = list(CRYPTOS.keys())

    # 1. Téléchargement
    prices = download_prices(tickers)

    # 2. Nettoyage
    prices = clean_prices(prices)

    # 3. Sauvegarde
    prices.to_csv(PRICES_FILE)
    print(f"💾 Sauvegardé : {PRICES_FILE}")

    # 4. Rapport
    quality_report(prices)


if __name__ == "__main__":
    main()
