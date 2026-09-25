"""
Data manager du GARCH-EVT Risk Engine.

Ce module fournit une API haut-niveau pour :
    - Charger les prix depuis le CSV local
    - Nettoyer les données
    - Calculer les log-rendements (en %)
    - Fournir un rapport de qualité

Il est utilisé par tous les autres modules du projet (GARCH, EVT, dashboard).

Auteur : Statby2Mf
Projet : GARCH-EVT Risk Engine (M2 Statistique, UGB Saint-Louis)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Racine du projet (src/ est un sous-dossier)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PRICES_FILE = DATA_DIR / "prices.csv"

#: Univers crypto — identique à download_data.py
CRYPTOS: dict[str, str] = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "BNB-USD": "Binance Coin",
    "SOL-USD": "Solana",
    "XRP-USD": "Ripple",
}


# ---------------------------------------------------------------------------
# Structure de rapport de qualité
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataQualityReport:
    """
    Rapport synthétique de qualité des données.

    Un `dataclass` est une classe Python qui génère automatiquement
    __init__, __repr__ et __eq__. Idéal pour transporter des données
    structurées de manière lisible.
    """
    n_obs: int
    n_assets: int
    start: pd.Timestamp
    end: pd.Timestamp
    missing_before_ffill: int
    missing_after_ffill: int

    def __str__(self) -> str:
        return (
            f"DataQualityReport(\n"
            f"  Période        : {self.start.date()} → {self.end.date()}\n"
            f"  Observations   : {self.n_obs} jours × {self.n_assets} actifs\n"
            f"  NaN avant      : {self.missing_before_ffill}\n"
            f"  NaN après      : {self.missing_after_ffill}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

def load_prices() -> pd.DataFrame:
    """
    Charge les prix depuis le CSV local.

    Returns
    -------
    pd.DataFrame
        Prix de clôture, index = dates, colonnes = tickers.

    Raises
    ------
    FileNotFoundError
        Si data/prices.csv n'existe pas.
        → Il faut d'abord lancer : python download_data.py
    """
    if not PRICES_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {PRICES_FILE}\n"
            f"→ Lance d'abord : python download_data.py"
        )
    prices = pd.read_csv(PRICES_FILE, index_col=0, parse_dates=True)
    return prices


# ---------------------------------------------------------------------------
# Nettoyage & qualité
# ---------------------------------------------------------------------------

def clean_prices(prices: pd.DataFrame) -> tuple[pd.DataFrame, DataQualityReport]:
    """
    Nettoie les prix et renvoie un rapport de qualité.

    Parameters
    ----------
    prices : pd.DataFrame
        Prix bruts.

    Returns
    -------
    (prices_clean, report)
    """
    n_before = int(prices.isna().sum().sum())

    # Supprime les jours où tout est NaN
    prices = prices.dropna(how="all")
    # Forward-fill puis supprime les NaN résiduels (début d'historique)
    prices = prices.ffill().dropna()

    n_after = int(prices.isna().sum().sum())

    report = DataQualityReport(
        n_obs=len(prices),
        n_assets=prices.shape[1],
        start=prices.index[0],
        end=prices.index[-1],
        missing_before_ffill=n_before,
        missing_after_ffill=n_after,
    )
    return prices, report


# ---------------------------------------------------------------------------
# Rendements
# ---------------------------------------------------------------------------

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les log-rendements en POURCENTAGE (× 100).

    Formule mathématique :
        r_t = 100 × ln(P_t / P_{t-1})

    Pourquoi × 100 ?
        - Le solveur de la bibliothèque `arch` (GARCH) converge mieux
          avec des rendements d'ordre 1 (≈ 1) qu'ordre 0.01.
        - N'affecte PAS les résultats statistiques : il suffit de se
          souvenir que les VaR/ES seront en % aussi.

    Returns
    -------
    pd.DataFrame
        Log-rendements en %, index = dates.
    """
    prices, _ = clean_prices(prices)
    returns = 100 * np.log(prices / prices.shift(1))
    return returns.dropna()


# ---------------------------------------------------------------------------
# API haut-niveau
# ---------------------------------------------------------------------------

def get_returns(ticker: str) -> pd.Series:
    """
    Renvoie les log-rendements (%) d'un actif unique.

    Parameters
    ----------
    ticker : str
        Ex : "BTC-USD".

    Returns
    -------
    pd.Series
        Série nommée d'après le ticker.
    """
    prices = load_prices()
    if ticker not in prices.columns:
        raise ValueError(
            f"Ticker '{ticker}' introuvable. "
            f"Disponibles : {list(prices.columns)}"
        )
    returns = compute_log_returns(prices[[ticker]])[ticker]
    return returns.rename(ticker)


def get_all_returns() -> pd.DataFrame:
    """Renvoie les log-rendements (%) de tous les actifs."""
    prices = load_prices()
    return compute_log_returns(prices)


def get_quality_report() -> DataQualityReport:
    """Renvoie le rapport de qualité des prix actuels."""
    prices = load_prices()
    _, report = clean_prices(prices)
    return report

def get_prices(ticker: str | None = None) -> pd.DataFrame | pd.Series:
    """
    Renvoie les prix nettoyés.

    Parameters
    ----------
    ticker : str or None
        Si fourni, renvoie une Series pour cet actif.
        Sinon, renvoie un DataFrame avec tous les actifs.

    Returns
    -------
    pd.DataFrame or pd.Series
    """
    prices = load_prices()
    prices, _ = clean_prices(prices)
    if ticker is not None:
        if ticker not in prices.columns:
            raise ValueError(f"Ticker '{ticker}' introuvable.")
        return prices[ticker]
    return prices
# ---------------------------------------------------------------------------
# Test rapide (uniquement si lancé directement)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("🔍 Test du data_manager\n")
    report = get_quality_report()
    print(report)

    print("\n📈 Aperçu BTC-USD :")
    btc = get_returns("BTC-USD")
    print(f"Observations : {len(btc)}")
    print(f"Pire jour    : {btc.min():.2f}% le {btc.idxmin().date()}")
    print(f"Meilleur     : {btc.max():.2f}% le {btc.idxmax().date()}")
    print(f"Moyenne      : {btc.mean():.3f}%")
    print(f"Écart-type   : {btc.std():.3f}%")
